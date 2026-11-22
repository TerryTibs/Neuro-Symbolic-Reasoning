# ============================================================
# SACRSN v37: BIG-TEXT READY + 2x VOCAB CODEBOOK + ZIPF ANNEALING
# Upgrades from v36:
#   1. Dynamic 2 * used_vocab codebook (up to 1024 on CUDA / 512 on CPU)
#   2. Diversity weight annealing (1.0x -> 0.25x) for Zipf-natural text
#   3. Auto Train/Val split on large text files + Validation checkpointing
#   4. CPU-resident corpus tensor + wider GPU streams (prevents CUDA OOM & speeds up big files)
#   5. Top-64 topology node cap + dynamic early stopping patience
# ============================================================

import os
import re
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter

# ==========================================
# 0. Determinism
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")

# ==========================================
# 1. Configuration
# ==========================================
CONFIG = {
    # Auto-Sizing Controls
    "auto_size": True,
    "min_symbols": 16,
    "max_symbols": 1024 if DEVICE == "cuda" else 512,
    "max_bpe_merges": 1536 if DEVICE == "cuda" else 768,
    "epoch_token_cap": 4096,        # Tokens swept per epoch across batched streams
    "diag_max_tokens": 2000,
    "val_split_min_chars": 12000,   # Enable held-out validation split above this size

    # Sequence & Model Capacity (auto-adjusted when auto_size=True)
    "batch_size": 4,
    "seq_len": 16,
    "bpe_merges": 32,
    "embedding_dim": 64,
    "n_symbols": 120,
    "stack_size": 16,
    "max_recursion_depth": 8,

    # Reasoning (ACT)
    "act_threshold": 0.99,
    "ponder_penalty": 0.0001,
    "use_stack": True,

    # Topology & Stability
    "commitment_cost": 0.25,
    "vq_weight": 1.0,
    "graph_bias_scale": 0.1,
    "symbol_consistency_weight": 0.01,
    "ethical_weight": 0.12,         # Strong token-level grammar signal
    "diversity_weight": 1.5,        # Anneals to 0.25x after warmup to match Zipf frequencies
    "vq_temp": 0.35,

    # Training Schedule (auto-adjusted when auto_size=True)
    "epochs": 2000,
    "warmup_epochs": 160,
    "log_every": 40,
    "sample_every": 200,
    "revive_every": 5,
    "early_stop_patience": 12,      # Stop after N log intervals without improvement post-50%
    "learning_rate": 1.5e-3,
    "weight_decay": 1e-5,
    "grad_clip": 1.0,
    "eps": 1e-6,
}

# ==========================================
# 2. Data, Fast Cached BPE Tokenizer & Auto-Configurator
# ==========================================
DATA_FILE = None  # Set to e.g. "my_book.txt" to test on big text files

TEXT_DATA = """True, without falsehood, certain and most true. 
That which is above is like to that which is below, 
and that which is below is like to that which is above.
The father of all perfection in the whole world is here.
Its force or power is entire if it be converted into earth."""

if DATA_FILE is not None and os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        TEXT_DATA = f.read()
    print(f"Loaded corpus from '{DATA_FILE}' ({len(TEXT_DATA):,} chars)")

class SimpleBPETokenizer:
    def __init__(self, text, num_merges=32, min_freq=2, min_tokens=None):
        self.pattern = re.compile(r' ?\w+|\n|[^\w\s]|[ \t\r]+')
        self.unk_token = "<UNK>"
        self.unk_id = 0
        self._encode_cache = {}

        base_chars = sorted(list(set(text)))
        self.vocab = [self.unk_token] + base_chars
        self.merges = []

        pre_tokens = self.pattern.findall(text)
        word_freqs = Counter(pre_tokens)

        if len(word_freqs) > 8000:
            learn_freqs = dict(word_freqs.most_common(8000))
            total_chars = sum(len(w) * f for w, f in word_freqs.items())
            learn_chars = sum(len(w) * f for w, f in learn_freqs.items())
            scaled_min_tokens = (
                int(min_tokens * (learn_chars / max(1, total_chars)))
                if min_tokens is not None
                else None
            )
        else:
            learn_freqs = dict(word_freqs)
            scaled_min_tokens = min_tokens

        splits = {word: list(word) for word in learn_freqs}
        total_tokens = sum(len(splits[w]) * freq for w, freq in learn_freqs.items())

        for _ in range(num_merges):
            if scaled_min_tokens is not None and total_tokens <= scaled_min_tokens:
                break

            pair_freqs = Counter()
            for word, freq in learn_freqs.items():
                syms = splits[word]
                for i in range(len(syms) - 1):
                    pair_freqs[(syms[i], syms[i + 1])] += freq

            if not pair_freqs:
                break

            best_pair, best_freq = pair_freqs.most_common(1)[0]
            if best_freq < min_freq:
                break

            if scaled_min_tokens is not None and (total_tokens - best_freq) < scaled_min_tokens:
                break

            merged_sym = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            if merged_sym not in self.vocab:
                self.vocab.append(merged_sym)

            p0, p1 = best_pair
            for word, freq in learn_freqs.items():
                syms = splits[word]
                if len(syms) < 2 or p0 not in syms or p1 not in syms:
                    continue
                new_syms = []
                i = 0
                merges_in_word = 0
                while i < len(syms):
                    if i < len(syms) - 1 and syms[i] == p0 and syms[i + 1] == p1:
                        new_syms.append(merged_sym)
                        i += 2
                        merges_in_word += 1
                    else:
                        new_syms.append(syms[i])
                        i += 1
                if merges_in_word > 0:
                    splits[word] = new_syms
                    total_tokens -= merges_in_word * freq

        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.vocab)}
        self.bpe_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.vocab_size = len(self.vocab)

    def _encode_word(self, word):
        cached = self._encode_cache.get(word)
        if cached is not None:
            return cached

        symbols = [ch if ch in self.token_to_id else self.unk_token for ch in word]
        while len(symbols) >= 2:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            best_pair = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if best_pair not in self.bpe_ranks:
                break
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                    new_symbols.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        result = [self.token_to_id.get(sym, self.unk_id) for sym in symbols]
        if len(self._encode_cache) < 50000:
            self._encode_cache[word] = result
        return result

    def encode(self, text):
        ids = []
        for chunk in self.pattern.findall(text):
            ids.extend(self._encode_word(chunk))
        return ids

    def decode(self, ids):
        return "".join(self.id_to_token.get(i, self.unk_token) for i in ids)

    @staticmethod
    def format_token(tok):
        return tok.replace("\n", "\\n").replace(" ", "·")


def auto_configure_from_text(text, config):
    char_count = len(text)
    unique_chars = len(set(text))

    # Automatic Train/Val split on larger texts (>= 12k chars)
    if char_count >= config["val_split_min_chars"]:
        val_chars = min(16000, int(0.08 * char_count))
        split_at = char_count - val_chars
        # Align split cleanly to whitespace if nearby
        ws_pos = text.find(" ", split_at)
        if ws_pos != -1 and (ws_pos - split_at) < 100:
            split_at = ws_pos + 1
        train_text = text[:split_at]
        val_text = text[split_at:]
    else:
        train_text = text
        val_text = None

    train_chars = len(train_text)
    if config.get("auto_size", True):
        target_merges = max(8, min(config["max_bpe_merges"], int(round(2.0 * math.sqrt(max(1, train_chars))))))
        min_pair_freq = max(2, int(math.log10(max(10, train_chars))))
        min_tokens_floor = max(32, int(0.40 * train_chars))
    else:
        target_merges = config["bpe_merges"]
        min_pair_freq = 2
        min_tokens_floor = None

    tok = SimpleBPETokenizer(
        train_text,
        num_merges=target_merges,
        min_freq=min_pair_freq,
        min_tokens=min_tokens_floor,
    )
    encoded_train = tok.encode(train_text)
    num_tokens = len(encoded_train)
    used_vocab = len(set(encoded_train))
    v_size = tok.vocab_size

    if num_tokens < 2:
        raise ValueError("TEXT_DATA is too short—provide at least a few words.")

    encoded_val = tok.encode(val_text) if val_text is not None else None
    if encoded_val is not None and len(encoded_val) < 16:
        encoded_val = None

    if config.get("auto_size", True):
        n_steps = num_tokens - 1

        # 1. Dynamic 2 * used_vocab Codebook Sizing (capped safely on tiny texts and hardware max)
        target_by_vocab = 2 * used_vocab
        target_by_tokens = max(config["min_symbols"], int(round(0.85 * num_tokens)))
        raw_symbols = min(target_by_vocab, target_by_tokens)
        snapped_symbols = int(round(raw_symbols / 8.0)) * 8
        n_symbols = max(config["min_symbols"], min(config["max_symbols"], snapped_symbols))

        # 2. GPU-Friendly Batched Streams (favor wider batches over long sequential Python loops)
        if n_steps < 48:
            batch_size = 1
        elif n_steps < 512:
            batch_size = min(4, max(2, n_steps // 32))
        elif n_steps < 4096:
            batch_size = 16 if DEVICE == "cuda" else 8
        else:
            batch_size = 64 if DEVICE == "cuda" else 16

        epoch_window = min(n_steps, config["epoch_token_cap"])
        stream_len = max(8, min(epoch_window, n_steps) // batch_size)
        adj_seq_len = max(4, min(32, max(8, stream_len // 2)))

        # 3. Dynamic Model Capacity
        raw_dim = int(round(8.0 * math.sqrt(max(16, v_size)) / 16.0)) * 16
        emb_dim = max(32, min(192, raw_dim))

        if n_steps < 80:
            stack_size = 8
        elif n_steps <= 2000:
            stack_size = 16
        else:
            stack_size = 24

        if n_steps <= 2000:
            rec_depth = 8
        elif n_steps <= 20000:
            rec_depth = 6
        else:
            rec_depth = 4

        adj_lr = float(1.5e-3 * math.sqrt(64.0 / emb_dim))
        adj_temp = float(0.35 * math.sqrt(64.0 / emb_dim))
        adj_wd = 1e-5 if n_steps < 4000 else 1e-4

        # 4. Step & Corpus-Pass Budgeted Schedule
        tokens_per_epoch = batch_size * stream_len
        chunks_per_epoch = max(1, math.ceil(stream_len / adj_seq_len))
        target_opt_steps = min(40000, int(round(4000 * (max(1, n_steps) / 150.0) ** 0.35)))
        epochs_by_steps = target_opt_steps // chunks_per_epoch
        # Ensure larger files still get >= 4 full passes if possible
        min_pass_epochs = int(math.ceil(4.0 * n_steps / max(1, tokens_per_epoch))) if n_steps > epoch_window else 150
        epochs = max(min_pass_epochs, min(2000, epochs_by_steps))
        epochs = max(150, min(2000, epochs))
        warmup_epochs = max(5, int(round(0.08 * epochs)))

        config["bpe_merges"] = len(tok.merges)
        config["n_symbols"] = n_symbols
        config["batch_size"] = batch_size
        config["seq_len"] = adj_seq_len
        config["embedding_dim"] = emb_dim
        config["stack_size"] = stack_size
        config["max_recursion_depth"] = rec_depth
        config["learning_rate"] = adj_lr
        config["weight_decay"] = adj_wd
        config["vq_temp"] = adj_temp
        config["epochs"] = epochs
        config["warmup_epochs"] = warmup_epochs
        config["log_every"] = max(1, epochs // 40)
        config["sample_every"] = max(1, epochs // 10)
        config["revive_every"] = max(1, min(5, epochs // 30))

    # Keep full corpus tensors on CPU to prevent GPU OOM on multi-MB files
    train_tensor = torch.tensor(encoded_train, dtype=torch.long)
    val_tensor = torch.tensor(encoded_val, dtype=torch.long) if encoded_val is not None else None

    n_steps = len(encoded_train) - 1
    bs = config["batch_size"]
    stream_len = max(1, min(n_steps, config["epoch_token_cap"]) // bs)
    est_passes = (config["epochs"] * bs * stream_len) / max(1, n_steps)

    print("=== Auto-Configuration Summary (SACRSN v37) ===")
    print(f"Corpus       : {char_count:,} chars ({unique_chars} base chars) | Train={len(encoded_train):,} toks"
          + (f" | Val={len(encoded_val):,} toks" if encoded_val is not None else " | Val=None (small text)"))
    print(
        f"Tokenizer    : Subword BPE | Vocab={v_size} (Used={used_vocab}) | Merges={len(tok.merges)}/{target_merges} "
        f"(min_freq>={min_pair_freq}) | Comp={train_chars / max(1, num_tokens):.2f}x"
    )
    print(
        f"Architecture : Dim={config['embedding_dim']} (2D={config['embedding_dim'] * 2}) | "
        f"Codebook={config['n_symbols']} (~{config['n_symbols'] / max(1, used_vocab):.2f}x used vocab) | "
        f"Stack={config['stack_size']} | MaxDepth={config['max_recursion_depth']}"
    )
    print(
        f"Schedule     : Streams={bs}x{stream_len} (Chunk={config['seq_len']}) | "
        f"Epochs={config['epochs']} (Warmup={config['warmup_epochs']}, ~{est_passes:.1f}x corpus passes)"
    )
    print(
        f"Hyperparams  : LR={config['learning_rate']:.5f} | WD={config['weight_decay']:.1e} | "
        f"VQTemp={config['vq_temp']:.3f}"
    )
    print(f"Sample tokens: {[tok.format_token(tok.id_to_token[i]) for i in encoded_train[:12]]}")
    print("===============================================")

    return tok, v_size, train_tensor, val_tensor


tokenizer, vocab_size, data_tensor, val_tensor = auto_configure_from_text(TEXT_DATA, CONFIG)

# ==========================================
# 3. Complex Primitives (Singularity-Free)
# ==========================================
class ComplexLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift_real = nn.Parameter(torch.zeros(dim))
        self.shift_imag = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        rms = torch.sqrt(torch.mean(z.real ** 2 + z.imag ** 2, dim=-1, keepdim=True) + 1e-4)
        z_real = (z.real / rms) * self.scale + self.shift_real
        z_imag = (z.imag / rms) * self.scale + self.shift_imag
        return torch.complex(z_real, z_imag)

class ModReLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        norm = torch.sqrt(z.real ** 2 + z.imag ** 2 + 1e-3)
        scale = F.relu(norm + self.bias) / norm
        return z * scale

class ComplexLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc_real = nn.Linear(dim, dim, bias=False)
        self.fc_imag = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.fc_real.weight)
        nn.init.orthogonal_(self.fc_imag.weight)

    def forward(self, z):
        return torch.complex(
            self.fc_real(z.real) - self.fc_imag(z.imag),
            self.fc_real(z.imag) + self.fc_imag(z.real)
        )

# ==========================================
# 4. Memory Modules
# ==========================================
class DifferentiableStack(nn.Module):
    def __init__(self, dim, size):
        super().__init__()
        self.dim = dim
        self.size = size

    def forward(self, z, memory, ptr, control):
        push, pop, noop = (
            control[:, 0].view(-1, 1),
            control[:, 1].view(-1, 1),
            control[:, 2].view(-1, 1),
        )

        ptr_up = torch.roll(ptr, 1, dims=1)
        ptr_down = torch.roll(ptr, -1, dims=1)
        new_ptr = (push * ptr_up) + (pop * ptr_down) + (noop * ptr)
        new_ptr = new_ptr / (new_ptr.sum(dim=1, keepdim=True) + CONFIG["eps"])

        z_flat = torch.cat([z.real, z.imag], dim=-1)
        write_mask = push * ptr_up
        write_val = write_mask.unsqueeze(2) * z_flat.unsqueeze(1)
        retain_mask = 1.0 - write_mask.unsqueeze(2)
        new_memory = write_val + (memory * retain_mask)

        read_mask = new_ptr.unsqueeze(2)
        read_flat = torch.sum(new_memory * read_mask, dim=1)
        read_z = torch.complex(read_flat[:, :self.dim], read_flat[:, self.dim:])

        return read_z, new_memory, new_ptr

class GraphMemoryVQ(nn.Module):
    def __init__(self, latent_dim, n_symbols):
        super().__init__()
        self.n_symbols = n_symbols

        self.codebook = nn.Parameter(torch.empty(n_symbols, latent_dim * 2))
        nn.init.normal_(self.codebook, mean=0.0, std=1.0)
        with torch.no_grad():
            rms = torch.sqrt(torch.mean(self.codebook ** 2, dim=-1, keepdim=True) + 1e-4)
            self.codebook.div_(rms)

        self.adjacency = nn.Parameter(torch.zeros(n_symbols, n_symbols))
        self.register_buffer("usage_count", torch.zeros(n_symbols))

    def forward(self, z, prev_symbol_idx=None, bias_scale=0.0):
        z_raw = torch.cat([z.real, z.imag], dim=-1)

        z_flat = z_raw / torch.sqrt(torch.mean(z_raw ** 2, dim=-1, keepdim=True) + 1e-4)
        cb_rms = torch.sqrt(torch.mean(self.codebook ** 2, dim=-1, keepdim=True) + 1e-4)
        cb_norm = self.codebook / cb_rms

        d_raw = (
            torch.sum(z_flat ** 2, dim=-1, keepdim=True)
            + torch.sum(cb_norm ** 2, dim=-1)
            - 2 * torch.matmul(z_flat, cb_norm.t())
        ) / z_flat.shape[-1]

        soft_probs = F.softmax(-d_raw / CONFIG["vq_temp"], dim=-1)

        d = d_raw
        if prev_symbol_idx is not None and bias_scale > 0.0:
            graph_prior = self.adjacency[prev_symbol_idx]
            bias = bias_scale * torch.sigmoid(graph_prior)
            d = d - bias

        min_indices = torch.argmin(d, dim=-1)

        if self.training:
            with torch.no_grad():
                self.usage_count.index_add_(
                    0,
                    min_indices.view(-1),
                    torch.ones_like(min_indices.view(-1), dtype=self.usage_count.dtype),
                )

        z_q = F.embedding(min_indices, self.codebook)

        loss_vq = F.mse_loss(z_q, z_flat.detach())
        loss_commit = F.mse_loss(z_q.detach(), z_flat)

        z_q_ste = z_flat + (z_q - z_flat).detach()
        z_complex = torch.complex(z_q_ste[..., :z.shape[-1]], z_q_ste[..., z.shape[-1]:])
        z_norm_complex = torch.complex(z_flat[..., :z.shape[-1]], z_flat[..., z.shape[-1]:])

        return (
            z_complex,
            z_norm_complex,
            loss_vq + CONFIG["commitment_cost"] * loss_commit,
            min_indices,
            soft_probs,
            z_flat.detach(),
        )

# ==========================================
# 5. Core Processor
# ==========================================
class ComplexAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.n_heads = 8 if dim % 8 == 0 else (4 if dim % 4 == 0 else 1)
        self.head_dim = dim // self.n_heads
        self.q_proj = ComplexLinear(dim)
        self.k_proj = ComplexLinear(dim)
        self.v_proj = ComplexLinear(dim)
        self.scale = (self.head_dim * 2) ** -0.5

    def forward(self, z):
        B = z.size(0)
        q = self.q_proj(z)
        k = self.k_proj(z)
        v = self.v_proj(z)

        q_flat = torch.cat(
            [q.real.view(B, self.n_heads, self.head_dim), q.imag.view(B, self.n_heads, self.head_dim)],
            dim=-1,
        )
        k_flat = torch.cat(
            [k.real.view(B, self.n_heads, self.head_dim), k.imag.view(B, self.n_heads, self.head_dim)],
            dim=-1,
        )

        attn_scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        v_real = torch.matmul(attn_weights, v.real.view(B, self.n_heads, self.head_dim)).reshape(B, self.dim)
        v_imag = torch.matmul(attn_weights, v.imag.view(B, self.n_heads, self.head_dim)).reshape(B, self.dim)
        return torch.complex(v_real, v_imag)

class EthicalConstraint(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prev_sym, curr_sym, adjacency):
        if prev_sym is None:
            return torch.tensor(0.0, device=adjacency.device)
        row_logits = adjacency[prev_sym]
        return F.cross_entropy(row_logits.view(-1, CONFIG["n_symbols"]), curr_sym.view(-1))

class AdaptiveRecursiveCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = ComplexLinear(dim)
        self.norm = ComplexLayerNorm(dim)
        self.act = ModReLU(dim)
        self.halt_linear = nn.Linear(dim * 2, 1)
        self.stack_ctrl = nn.Linear(dim * 2, 3)
        self.attention = ComplexAttention(dim)

        p0 = min(0.45, 1.5 / max(1, CONFIG["max_recursion_depth"]))
        halt_bias = float(math.log(p0 / (1.0 - p0)))
        nn.init.constant_(self.halt_linear.bias, halt_bias)

    def forward(self, z):
        z_proc = self.act(self.norm(self.linear(z)))
        z_proc = self.attention(z_proc)
        z_flat = torch.cat([z_proc.real, z_proc.imag], dim=-1)
        halt_prob = torch.sigmoid(self.halt_linear(z_flat))
        stack_probs = F.softmax(self.stack_ctrl(z_flat), dim=-1)
        return z_proc, halt_prob, stack_probs

# ==========================================
# 6. Master Model (UberCRSN)
# ==========================================
class UberCRSN(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.emb_mag = nn.Embedding(vocab_size, dim)
        self.emb_phase = nn.Parameter(torch.randn(vocab_size, dim))
        self.cell = AdaptiveRecursiveCell(dim)
        self.vq_layer = GraphMemoryVQ(dim, CONFIG["n_symbols"])
        self.decoder = nn.Linear(dim * 2, vocab_size)

        if CONFIG["use_stack"]:
            self.stack = DifferentiableStack(dim, CONFIG["stack_size"])

        self.ethics = EthicalConstraint()
        self.register_buffer("prev_sym_soft", torch.zeros(CONFIG["n_symbols"]))

    def embed(self, idx):
        r = self.emb_mag(idx)
        t = self.emb_phase[idx]
        return torch.complex(r * torch.cos(t), r * torch.sin(t))

    def forward(self, input_ids, hidden=None, prev_sym=None, bias_scale=None, return_stats=False):
        if bias_scale is None:
            bias_scale = CONFIG["graph_bias_scale"]

        batch_size = input_ids.size(0)
        z = self.embed(input_ids).squeeze(1)

        if hidden is None:
            z_prev = torch.zeros_like(z)
            stack_mem = torch.zeros(batch_size, CONFIG["stack_size"], self.dim * 2, device=z.device)
            stack_ptr = torch.zeros(batch_size, CONFIG["stack_size"], device=z.device)
            stack_ptr[:, 0] = 1.0
        else:
            z_prev, stack_mem, stack_ptr = hidden
            z = 0.5 * z + 0.5 * z_prev

        halting_probability = torch.zeros(batch_size, 1, device=z.device)
        remain = torch.ones(batch_size, 1, device=z.device)
        ponder_cost = torch.tensor(0.0, device=z.device)
        stack_history = []
        step_soft_probs = []
        step_z_flats = []

        z_weighted = torch.zeros_like(z)
        current_sym = prev_sym
        vq_loss_total = torch.tensor(0.0, device=z.device)
        act_steps = 0

        for t in range(CONFIG["max_recursion_depth"]):
            still_running = (halting_probability < CONFIG["act_threshold"]).float()
            if still_running.max().item() == 0.0:
                break

            act_steps += 1
            z_proc, p_halt, stack_ctrl = self.cell(z)

            if CONFIG["use_stack"]:
                stack_read, stack_mem, stack_ptr = self.stack(z_proc, stack_mem, stack_ptr, stack_ctrl)
                z_combined = z_proc + stack_read
                depth = torch.sum(stack_ptr * torch.arange(CONFIG["stack_size"], device=z.device), dim=1)
                stack_history.append(depth)
            else:
                z_combined = z_proc
                stack_history.append(torch.zeros(1, device=z.device))

            prior_sym = prev_sym if t == 0 else None
            step_bias = bias_scale if t == 0 else 0.0

            z_sym, z_norm, vq_loss, sym_idx, soft_probs, z_flat_det = self.vq_layer(
                z_combined, prior_sym, bias_scale=step_bias
            )
            step_soft_probs.append(soft_probs)
            step_z_flats.append(z_flat_det)
            current_sym = sym_idx

            z = 0.1 * z_norm + 0.9 * z_sym

            new_halted = ((halting_probability + p_halt * still_running) >= CONFIG["act_threshold"]).float() * still_running
            still_running_after = still_running - new_halted

            if t == CONFIG["max_recursion_depth"] - 1:
                p = remain * still_running
            else:
                p = p_halt * still_running_after + remain * new_halted

            halting_probability = halting_probability + p
            remain = remain - p

            z_weighted = z_weighted + (p * z)
            ponder_cost = ponder_cost + still_running.mean()
            vq_loss_total = vq_loss_total + vq_loss

        vq_loss_total = vq_loss_total / max(1, act_steps)
        ethical_loss = self.ethics(prev_sym, current_sym, self.vq_layer.adjacency)

        features = torch.cat([z_weighted.real, z_weighted.imag], dim=-1)
        logits = self.decoder(features)

        next_hidden = (z_weighted, stack_mem, stack_ptr)

        if len(stack_history) > 0:
            avg_stack = torch.stack(stack_history).mean()
        else:
            avg_stack = torch.tensor(0.0, device=z.device)

        if return_stats:
            return (
                logits,
                next_hidden,
                current_sym,
                ponder_cost,
                vq_loss_total,
                ethical_loss,
                avg_stack,
                torch.cat(step_soft_probs, dim=0),
                torch.cat(step_z_flats, dim=0),
            )

        return logits, next_hidden, current_sym, ponder_cost, vq_loss_total, ethical_loss, avg_stack

# ==========================================
# 7. Training & Validation Engine
# ==========================================
def build_epoch_streams(cpu_tensor, batch_size, epoch_cap, epoch):
    n_steps = len(cpu_tensor) - 1
    epoch_window = min(n_steps, epoch_cap)
    stream_len = max(1, epoch_window // batch_size)
    total_span = batch_size * stream_len

    if total_span > n_steps:
        stream_len = max(1, n_steps // batch_size)
        total_span = batch_size * stream_len

    slack = max(0, n_steps - total_span)
    if n_steps > epoch_cap and slack > 0:
        base = (epoch * total_span) % (slack + 1)
    elif slack > 0:
        base = random.randint(0, slack)
    else:
        base = 0

    offsets = [base + b * stream_len for b in range(batch_size)]
    streams = torch.stack([cpu_tensor[off : off + stream_len + 1] for off in offsets], dim=0)
    return streams.to(DEVICE)


@torch.no_grad()
def evaluate_val(model, val_cpu_tensor):
    if val_cpu_tensor is None or len(val_cpu_tensor) < 4:
        return None, None
    model.eval()
    n_val = len(val_cpu_tensor) - 1
    b_val = min(8, max(1, n_val // 64))
    s_len = min(128, n_val // b_val)
    streams = torch.stack(
        [val_cpu_tensor[b * s_len : b * s_len + s_len + 1] for b in range(b_val)], dim=0
    ).to(DEVICE)

    hidden, prev_sym = None, None
    total_loss = 0.0
    total_correct = 0
    for t in range(s_len):
        x = streams[:, t].unsqueeze(1)
        y = streams[:, t + 1]
        logits, hidden, prev_sym, _, _, _, _ = model(x, hidden, prev_sym)
        total_loss += F.cross_entropy(logits, y).item()
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())

    model.train()
    return total_loss / max(1, s_len), 100.0 * total_correct / max(1, b_val * s_len)


def train():
    model = UberCRSN(vocab_size, CONFIG["embedding_dim"]).to(DEVICE)

    codebook_params = [model.vq_layer.codebook]
    adj_params = [model.vq_layer.adjacency]
    other_params = [
        p for n, p in model.named_parameters()
        if n not in ("vq_layer.codebook", "vq_layer.adjacency")
    ]
    opt = torch.optim.AdamW(
        [
            {"params": other_params, "weight_decay": CONFIG["weight_decay"], "lr": CONFIG["learning_rate"]},
            {"params": codebook_params, "weight_decay": 0.0, "lr": CONFIG["learning_rate"]},
            {"params": adj_params, "weight_decay": 0.0, "lr": CONFIG["learning_rate"] * 3.0},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=CONFIG["epochs"], eta_min=max(1e-5, CONFIG["learning_rate"] * 0.01)
    )

    log_K = float(math.log(CONFIG["n_symbols"]))
    t_start = time.time()
    revived_total = 0
    min_active_converge = max(4, min(16, CONFIG["n_symbols"] // 4))
    best_metric = float("inf")
    best_state = None
    no_improve_logs = 0

    print("--- Training SACRSN v37 (Big-Text Ready + 2x Vocab Codebook + Zipf Annealing) ---")

    try:
        for epoch in range(CONFIG["epochs"]):
            model.train()
            streams = build_epoch_streams(
                data_tensor, CONFIG["batch_size"], CONFIG["epoch_token_cap"], epoch
            )
            bs, span_len = streams.shape
            stream_len = span_len - 1

            hidden = None
            prev_sym = None
            total_loss = 0.0
            total_pred_loss = 0.0
            total_vq = 0.0
            total_eth = 0.0
            total_div = 0.0
            total_ponder = 0.0
            total_stack = 0.0
            total_correct = 0
            total_toks = 0
            grad_norms = []

            progress = epoch / max(1, CONFIG["epochs"] - 1)
            warmup = min(1.0, epoch / max(1, CONFIG["warmup_epochs"]))
            curr_bias_scale = CONFIG["graph_bias_scale"] * warmup
            curr_eth_weight = CONFIG["ethical_weight"] * warmup
            entropy_weight = 0.01 * (1.0 - progress)

            # Anneal diversity from 1.0x -> 0.25x after first 25% so codebook reflects Zipf frequencies
            div_decay = 1.0 - 0.75 * max(0.0, (progress - 0.25) / 0.75)
            curr_div_weight = CONFIG["diversity_weight"] * div_decay

            revive_this_epoch = (epoch % CONFIG["revive_every"] == 0) and (epoch < int(0.45 * CONFIG["epochs"]))
            epoch_z_pool = []

            chunk_loss = 0.0
            chunk_probs = []
            steps_in_chunk = 0

            for t in range(stream_len):
                x = streams[:, t].unsqueeze(1)
                y = streams[:, t + 1]

                logits, hidden, sym_idx, ponder, vq_loss, eth_loss, avg_stack, soft_probs, z_flats = model(
                    x, hidden, prev_sym, bias_scale=curr_bias_scale, return_stats=True
                )

                h_z, h_mem, h_ptr = hidden
                hidden = (h_z.detach(), h_mem.detach(), h_ptr.detach())
                prev_sym = sym_idx.detach()

                loss_pred = F.cross_entropy(logits, y)
                loss_ponder = CONFIG["ponder_penalty"] * ponder

                probs = F.softmax(logits, dim=-1)
                loss_entropy = -entropy_weight * (-(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean())

                loss_vq = CONFIG["vq_weight"] * vq_loss
                loss_ethics = curr_eth_weight * eth_loss

                step_loss = loss_pred + loss_ponder + loss_vq + loss_entropy + loss_ethics
                chunk_loss = chunk_loss + step_loss
                chunk_probs.append(soft_probs)
                if revive_this_epoch and len(epoch_z_pool) < 128:
                    epoch_z_pool.append(z_flats)
                steps_in_chunk += 1

                total_pred_loss += loss_pred.item()
                total_vq += vq_loss.item()
                total_eth += eth_loss.item()
                total_ponder += ponder.item()
                total_stack += avg_stack.item()
                total_correct += int((logits.argmax(dim=-1) == y).sum().item())
                total_toks += bs

                if steps_in_chunk == CONFIG["seq_len"] or t == stream_len - 1:
                    probs_cat = torch.cat(chunk_probs, dim=0)
                    avg_probs = probs_cat.mean(dim=0)

                    marginal_entropy = -(avg_probs * torch.log(avg_probs + 1e-8)).sum()
                    norm_kl = (log_K - marginal_entropy) / max(1e-8, log_K)
                    sample_entropy = -(probs_cat * torch.log(probs_cat + 1e-8)).sum(dim=-1).mean()
                    norm_sample_ent = sample_entropy / max(1e-8, log_K)
                    loss_diversity = curr_div_weight * (norm_kl + 0.5 * norm_sample_ent)

                    total_chunk_loss = (chunk_loss / steps_in_chunk) + loss_diversity

                    opt.zero_grad()
                    total_chunk_loss.backward()
                    gnorm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), CONFIG["grad_clip"]
                    ).item()
                    grad_norms.append(gnorm)
                    opt.step()

                    total_loss += total_chunk_loss.item() * steps_in_chunk
                    total_div += loss_diversity.item() * steps_in_chunk
                    chunk_loss = 0.0
                    chunk_probs = []
                    steps_in_chunk = 0

            scheduler.step()

            avg_pred = total_pred_loss / stream_len

            with torch.no_grad():
                counts = model.vq_layer.usage_count.clone()
                active_syms = int((counts > 0).sum().item())
                dist = counts / (counts.sum() + 1e-10)
                model.prev_sym_soft.copy_(dist)
                entropy_val = -(dist * torch.log(dist + 1e-10)).sum()
                true_ppx = torch.exp(entropy_val).item()

                if revive_this_epoch and len(epoch_z_pool) > 0:
                    dead_mask = counts == 0
                    n_dead = int(dead_mask.sum().item())
                    if n_dead > 0:
                        pool = torch.cat(epoch_z_pool, dim=0)
                        rand_idx = torch.randint(0, pool.size(0), (n_dead,), device=DEVICE)
                        new_vecs = pool[rand_idx] + 0.05 * torch.randn(n_dead, pool.size(1), device=DEVICE)
                        new_vecs = new_vecs / torch.sqrt(torch.mean(new_vecs ** 2, dim=-1, keepdim=True) + 1e-4)
                        model.vq_layer.codebook.data[dead_mask] = new_vecs
                        revived_total += n_dead

                model.vq_layer.usage_count.zero_()

            is_last_epoch = (epoch == CONFIG["epochs"] - 1)
            if epoch % CONFIG["log_every"] == 0 or is_last_epoch:
                val_loss, val_acc = evaluate_val(model, val_tensor)
                monitor_loss = val_loss if val_loss is not None else avg_pred

                if epoch >= CONFIG["warmup_epochs"] and active_syms >= min_active_converge:
                    if monitor_loss < best_metric - 1e-3:
                        best_metric = monitor_loss
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        no_improve_logs = 0
                    elif progress >= 0.5:
                        no_improve_logs += 1

                avg_loss = total_loss / stream_len
                avg_vq = total_vq / stream_len
                avg_eth = total_eth / stream_len
                avg_div = total_div / stream_len
                avg_ponder = total_ponder / stream_len
                avg_stk = total_stack / stream_len
                acc_pct = 100.0 * total_correct / max(1, total_toks)
                tok_ppx = float(np.exp(min(avg_pred, 20.0)))
                top1_pct = float((counts.max() / (counts.sum() + 1e-10) * 100.0).item())
                strong_edges = int((F.softmax(model.vq_layer.adjacency, dim=-1) > 0.08).sum().item())
                avg_gnorm = sum(grad_norms) / max(1, len(grad_norms))
                dt = time.time() - t_start
                t_start = time.time()
                lr = scheduler.get_last_lr()[0]

                val_str = f" | Val:{val_loss:.3f} ({val_acc:4.1f}%)" if val_loss is not None else ""
                print(
                    f"Ep {epoch:04d} | Loss:{avg_loss:.3f} (Pred:{avg_pred:.3f} VQ:{avg_vq:.3f} "
                    f"Div:{avg_div:.3f} Eth:{avg_eth:.3f}) | Acc:{acc_pct:5.1f}% TokPPX:{tok_ppx:5.2f}{val_str} | "
                    f"ACT:{avg_ponder:.2f} Stk:{avg_stk:.2f} | "
                    f"Syms:{active_syms:3d}/{CONFIG['n_symbols']} SymPPX:{true_ppx:5.1f} Top1:{top1_pct:4.1f}% "
                    f"Rev:{revived_total:3d} Edges:{strong_edges:4d} | "
                    f"Grad:{avg_gnorm:.2f} LR:{lr:.5f} ({dt:.1f}s)"
                )
                revived_total = 0

                if epoch % CONFIG["sample_every"] == 0 or is_last_epoch:
                    model.eval()
                    with torch.no_grad():
                        seed_id = data_tensor[0].item()
                        gx = torch.tensor([[seed_id]], device=DEVICE)
                        gh, gp = None, None
                        sample_ids = [seed_id]
                        for _ in range(40):
                            g_logits, gh, gp, _, _, _, _ = model(gx, gh, gp)
                            gx = g_logits.argmax(dim=-1, keepdim=True)
                            sample_ids.append(gx.item())
                    print(f"   Sample: {repr(tokenizer.decode(sample_ids))}")
                    model.train()

                if avg_pred < 0.01 and active_syms >= min_active_converge:
                    print("\n--- PERFECT CONVERGENCE ---")
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_metric = avg_pred
                    break

                if progress >= 0.5 and no_improve_logs >= CONFIG["early_stop_patience"]:
                    print(f"\n--- EARLY STOPPING (plateaued for {no_improve_logs} checks) ---")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    if best_state is not None:
        model.load_state_dict(best_state)
        metric_name = "Val Loss" if val_tensor is not None else "Pred Loss"
        print(f"Restored best healthy checkpoint ({metric_name}: {best_metric:.4f}).")

    return model

# ==========================================
# 8. Visualization Suite
# ==========================================
def visualize_all(model):
    print("\n--- Generating Diagnostics & Images ---")
    model.eval()

    diag_steps = min(len(data_tensor) - 1, CONFIG["diag_max_tokens"])

    # 1. Semantic Mapping
    symbol_to_tok = defaultdict(list)
    hidden, prev_sym = None, None
    with torch.no_grad():
        for i in range(diag_steps):
            x = data_tensor[i].view(1, 1).to(DEVICE)
            _, hidden, prev_sym, _, _, _, _ = model(x, hidden, prev_sym)
            tok_str = tokenizer.format_token(tokenizer.id_to_token[data_tensor[i].item()])
            symbol_to_tok[prev_sym.item()].append(tok_str)

    node_labels = {}
    for sym_idx in range(CONFIG["n_symbols"]):
        tok_list = symbol_to_tok.get(sym_idx, [])
        if tok_list:
            most_common = max(set(tok_list), key=tok_list.count)
            node_labels[sym_idx] = f"{most_common}\n({len(tok_list)})"
        else:
            node_labels[sym_idx] = f"{sym_idx}"

    # 2. X-Ray Topology (caps at top 64 most active nodes for readability on big files)
    adj_probs = F.softmax(model.vq_layer.adjacency.detach(), dim=-1).cpu().numpy()
    G = nx.DiGraph()

    active_sorted = sorted(symbol_to_tok.keys(), key=lambda s: len(symbol_to_tok[s]), reverse=True)
    if CONFIG["n_symbols"] <= 64:
        nodes_to_plot = list(range(CONFIG["n_symbols"]))
    else:
        nodes_to_plot = active_sorted[:64] if active_sorted else list(range(min(64, CONFIG["n_symbols"])))

    node_set = set(nodes_to_plot)
    edge_thresh = 0.04
    max_out_edges = 4

    for i in nodes_to_plot:
        G.add_node(i)

    edges, weights = [], []
    for i in nodes_to_plot:
        row_candidates = [
            (j, float(adj_probs[i, j]))
            for j in nodes_to_plot
            if adj_probs[i, j] > edge_thresh and i != j
        ]
        row_candidates.sort(key=lambda pair: pair[1], reverse=True)
        for j, w in row_candidates[:max_out_edges]:
            G.add_edge(i, j, weight=w)
            edges.append((i, j))
            weights.append(w)

    plt.figure(figsize=(14, 14))
    if len(G.nodes()) > 0:
        try:
            pos = nx.spring_layout(G, k=0.25, iterations=60, seed=42)
        except Exception:
            pos = nx.circular_layout(G)

        node_count = len(G.nodes())
        node_size = max(250, min(850, 30000 // max(1, node_count)))
        font_size = max(6, min(9, 450 // max(1, node_count)))
        node_colors = ['#a0cbe2' if i in symbol_to_tok else '#ffe5e5' for i in G.nodes()]
        active_labels = {i: node_labels[i] for i in G.nodes() if i in node_set}

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, alpha=0.9)
        nx.draw_networkx_labels(G, pos, labels=active_labels, font_size=font_size, font_weight="bold")

        for (u, v), w in zip(edges, weights):
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], width=max(0.6, w * 5.0), alpha=min(0.85, max(0.2, w * 2.0)),
                edge_color='gray', arrowstyle='->', arrowsize=12
            )

    plt.title(f"1_semantic_topology (Active: {len(symbol_to_tok)}/{CONFIG['n_symbols']}, Shown: {len(G.nodes())})")
    plt.savefig("1_semantic_topology.png", dpi=150)
    print("Saved 1_semantic_topology.png")
    plt.close()

    # 3. Inference Scan
    hidden, prev_sym = None, None
    seed_id = data_tensor[0].item()
    x = torch.tensor([[seed_id]], device=DEVICE)
    stack_history, act_history, phase_reals, phase_imags = [], [], [], []
    gen_ids = [seed_id]

    print("Running Inference Scan...")
    for _ in range(100):
        with torch.no_grad():
            logits, hidden, prev_sym, ponder, _, _, s_hist = model(x, hidden, prev_sym)

            stack_history.append(s_hist.item())
            act_history.append(ponder.item())

            z = hidden[0].cpu().squeeze()
            if z.dim() > 0:
                phase_reals.append(z.real[0].item())
                phase_imags.append(z.imag[0].item())
            else:
                phase_reals.append(z.real.item())
                phase_imags.append(z.imag.item())

            probs = F.softmax(logits, dim=-1)
            next_ix = torch.multinomial(probs, 1)
            gen_ids.append(next_ix.item())
            x = next_ix

    print(f"Generated: {tokenizer.decode(gen_ids)}\n")

    # Stack MRI
    plt.figure(figsize=(12, 4))
    plt.plot(stack_history, color='purple', label='Stack Depth')
    plt.fill_between(range(len(stack_history)), stack_history, color='purple', alpha=0.1)
    plt.title("2_stack_mri (Memory Depth)")
    plt.savefig("2_stack_mri.png")
    plt.close()

    # ACT Profile
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(act_history)), act_history, color='orange')
    plt.ylim(0, CONFIG["max_recursion_depth"] + 0.5)
    plt.title("3_act_profile (Thinking Steps)")
    plt.savefig("3_act_profile.png")
    plt.close()

    # Phase Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(phase_reals, phase_imags, c=range(len(phase_reals)), cmap='plasma', alpha=0.5)
    plt.colorbar(label="Time")
    plt.title("4_phase_plot (Complex Trajectory)")
    plt.axis('equal')
    plt.savefig("4_phase_plot.png")
    plt.close()

# ==========================================
# 9. Logic Rule Extraction
# ==========================================
def extract_logic_rules(model, data_tensor):
    print("\n--- Extracting Explicit Logic Rules ---")
    model.eval()
    rule_book = defaultdict(list)
    hidden = None
    prev_sym = None
    diag_steps = min(len(data_tensor) - 1, CONFIG["diag_max_tokens"])
    min_rule_count = max(2, diag_steps // 100)

    with torch.no_grad():
        for i in range(diag_steps):
            x = data_tensor[i].view(1, 1).to(DEVICE)
            logits, hidden, sym_idx, ponder, _, _, _ = model(x, hidden, prev_sym)
            if prev_sym is not None:
                src = prev_sym.item()
                dst = sym_idx.item()
                rule_book[(src, dst)].append(ponder.item())
            prev_sym = sym_idx

    print(f"\n{'FROM':<6} | {'TO':<6} | {'COUNT':<6} | {'AVG STEPS':<10}")
    print("-" * 45)
    sorted_rules = sorted(rule_book.items(), key=lambda x: len(x[1]), reverse=True)
    printed = 0
    for (src, dst), ponders in sorted_rules:
        if len(ponders) >= min_rule_count:
            print(f"S_{src:<4} -> S_{dst:<4} | {len(ponders):<6} | {sum(ponders)/len(ponders):.2f}")
            printed += 1
            if printed >= 30:
                break

# ==========================================
# 10. Advanced Interaction
# ==========================================
def dream_mode(model):
    print("\n--- Dream Mode (Pure Symbolic Nucleus Walk) ---")
    adj_logits = model.vq_layer.adjacency.detach()
    model.eval()
    seed_id = data_tensor[0].item()
    x = torch.tensor([[seed_id]], device=DEVICE)
    with torch.no_grad():
        _, _, prev_sym, _, _, _, _ = model(x, None, None)
    curr_sym = prev_sym.item()
    output_ids = [seed_id]

    for _ in range(60):
        probs = F.softmax(adj_logits[curr_sym] / 0.7, dim=-1).cpu().numpy()
        sorted_idx = np.argsort(probs)[::-1]
        cum_probs = np.cumsum(probs[sorted_idx])
        cutoff = int(np.searchsorted(cum_probs, 0.90)) + 1
        keep_idx = sorted_idx[:max(1, cutoff)]

        filtered = np.zeros_like(probs)
        filtered[keep_idx] = probs[keep_idx]
        filtered = filtered / filtered.sum()
        next_sym = int(np.random.choice(len(filtered), p=filtered))

        with torch.no_grad():
            z_flat = model.vq_layer.codebook[next_sym].unsqueeze(0)
            logits = model.decoder(z_flat)
            tok_idx = torch.argmax(logits).item()

        output_ids.append(tok_idx)
        curr_sym = next_sym

    print(f"Dream Output: {tokenizer.decode(output_ids)}\n")

def anomaly_detector(model):
    print("\n--- Anomaly Detection Test ---")
    model.eval()

    sample_words = TEXT_DATA[:400].split()
    prefix = " ".join(sample_words[:min(7, len(sample_words))])
    anomaly_word = "banana" if "banana" not in TEXT_DATA.lower() else "xyzzy"
    corrupt_text = f"{prefix} {anomaly_word}" if prefix else anomaly_word
    print(f"Input: '{corrupt_text}'")

    input_ids = tokenizer.encode(corrupt_text)
    if len(input_ids) < 2:
        print("Input too short for anomaly scan.")
        return

    input_tensor = torch.tensor(input_ids, dtype=torch.long).to(DEVICE)
    hidden, prev_sym = None, None
    anomalies = []

    with torch.no_grad():
        for i in range(len(input_tensor) - 1):
            x = input_tensor[i].view(1, 1)
            _, hidden, prev_sym, _, _, eth_loss, _ = model(x, hidden, prev_sym)
            anomalies.append(eth_loss.item())

    x_labels = [tokenizer.format_token(tokenizer.id_to_token[idx]) for idx in input_ids[1:]]
    x_idx = np.arange(len(anomalies))

    plt.figure(figsize=(12, 4))
    plt.plot(x_idx, anomalies, color='skyblue', linestyle='-', linewidth=2, label='Topology Flow')
    plt.scatter(x_idx, anomalies, color='crimson', s=50, zorder=5, label='Anomaly Score')
    plt.xticks(x_idx, x_labels, rotation=35, ha='right', fontsize=8)

    plt.title("Topological Violation Score (Anomaly Detection)")
    plt.ylabel("Violation Magnitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("5_anomaly_detection.png")
    print("Saved 5_anomaly_detection.png")
    plt.close()

# ==========================================
# 11. Main Execution
# ==========================================
if __name__ == "__main__":
    FILENAME = "crsn_omni_model.pth"

    trained_model = train()

    print(f"\n--- Saving Model to {FILENAME} ---")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': CONFIG,
        'vocab': tokenizer.vocab,
        'merges': tokenizer.merges,
    }, FILENAME)
    print("Saved.")

    visualize_all(trained_model)
    extract_logic_rules(trained_model, data_tensor)

    dream_mode(trained_model)
    anomaly_detector(trained_model)

    try:
        from google.colab import files
        files.download(FILENAME)
        files.download("1_semantic_topology.png")
        files.download("2_stack_mri.png")
        files.download("3_act_profile.png")
        files.download("4_phase_plot.png")
        files.download("5_anomaly_detection.png")
    except ImportError:
        pass