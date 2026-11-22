# ============================================================
# SACRSN v40: RTX 5060 TI / WIN11 ACCELERATED EDITION
# Hardware Upgrades:
#   - Native (real, imag) tuples unlocking BF16 & TF32 Tensor Cores
#   - Zero per-step .item() CPU-GPU syncs in forward & train loop
#   - Fused AdamW + cudnn.benchmark + VRAM-aware 16GB auto-scaling
# Learning Upgrades:
#   - Multi-step TBPTT across full chunks (real stack credit assignment)
#   - Stack-Attentive 8-Head ComplexAttention over all stack_mem slots
#   - All 7 v39 Closed-Loop Self-Tuning Controllers & Telemetry
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
# 0. Hardware Setup & Determinism
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
VRAM_GB = (
    torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if DEVICE == "cuda"
    else 0.0
)
USE_BF16 = (DEVICE == "cuda") and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

if DEVICE == "cuda":
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

print(
    f"Hardware: {DEVICE} ({GPU_NAME}, {VRAM_GB:.1f} GB VRAM) | "
    f"BF16={USE_BF16} | TF32={DEVICE == 'cuda'}"
)

# ==========================================
# 1. Configuration
# ==========================================
HIGH_VRAM = (DEVICE == "cuda") and (VRAM_GB >= 12.0)

CONFIG = {
    # Auto-Sizing Controls (scales up automatically on >=12GB GPUs like 5060 Ti 16GB)
    "auto_size": True,
    "min_symbols": 16,
    "max_symbols": 2048 if HIGH_VRAM else (1024 if DEVICE == "cuda" else 512),
    "max_bpe_merges": 2048 if HIGH_VRAM else (1536 if DEVICE == "cuda" else 768),
    "max_dim": 256 if HIGH_VRAM else (192 if DEVICE == "cuda" else 128),
    "epoch_token_cap": 32768 if HIGH_VRAM else 4096,
    "diag_max_tokens": 2000,
    "val_split_min_chars": 12000,

    # Sequence & Model Capacity (auto-adjusted when auto_size=True)
    "batch_size": 4,
    "seq_len": 16,                  # Also acts as multi-step TBPTT horizon
    "bpe_merges": 32,
    "embedding_dim": 64,
    "n_symbols": 120,
    "stack_size": 16,
    "max_recursion_depth": 8,

    # Reasoning (ACT)
    "act_threshold": 0.99,
    "ponder_penalty": 0.0001,
    "use_stack": True,

    # Topology & Stability (initial values; self-tuned during training)
    "commitment_cost": 0.25,
    "vq_weight": 1.0,
    "graph_bias_scale": 0.1,
    "symbol_consistency_weight": 0.01,
    "ethical_weight": 0.12,
    "diversity_weight": 1.5,
    "vq_temp": 0.35,

    # Training Schedule (auto-adjusted when auto_size=True)
    "epochs": 2000,
    "warmup_epochs": 160,
    "log_every": 40,
    "sample_every": 200,
    "revive_every": 5,
    "lr_patience": 4,
    "early_stop_patience": 10,
    "learning_rate": 1.5e-3,
    "weight_decay": 1e-5,
    "grad_clip": 1.0,
    "eps": 1e-6,
    "comp_ratio": 1.75,
}

# ==========================================
# 2. Data, Rare-Pruned BPE Tokenizer & Auto-Configurator
# ==========================================
DATA_FILE = "data.txt"

DEFAULT_TEXT_DATA = """True, without falsehood, certain and most true. 
That which is above is like to that which is below, 
and that which is below is like to that which is above.
The father of all perfection in the whole world is here.
Its force or power is entire if it be converted into earth."""

TEXT_DATA = DEFAULT_TEXT_DATA
if DATA_FILE and os.path.isfile(DATA_FILE):
    try:
        with open(DATA_FILE, "r", encoding="utf-8-sig", errors="replace") as f:
            loaded_text = f.read()
        if len(loaded_text.strip()) >= 8:
            TEXT_DATA = loaded_text
            print(f"Loaded corpus from '{DATA_FILE}' ({len(TEXT_DATA):,} chars)")
        else:
            print(f"'{DATA_FILE}' is empty/too short — using built-in default text.")
    except OSError as e:
        print(f"Could not read '{DATA_FILE}' ({e}) — using built-in default text.")
else:
    print(f"'{DATA_FILE}' not found — using built-in default text.")

class SimpleBPETokenizer:
    def __init__(self, text, num_merges=32, min_freq=2, min_tokens=None, min_char_freq=1):
        self.pattern = re.compile(r' ?\w+|\n|[^\w\s]|[ \t\r]+')
        self.unk_token = "<UNK>"
        self.unk_id = 0
        self._encode_cache = {}

        char_freqs = Counter(text)
        base_chars = sorted([ch for ch, freq in char_freqs.items() if freq >= min_char_freq])
        if not base_chars:
            base_chars = sorted(list(char_freqs.keys()))
        self.valid_chars = set(base_chars)

        self.vocab = [self.unk_token] + base_chars
        self.merges = []

        pre_tokens = self.pattern.findall(text)
        word_freqs = Counter(pre_tokens)

        if len(word_freqs) > 10000:
            learn_freqs = dict(word_freqs.most_common(10000))
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

        splits = {
            word: [ch if ch in self.valid_chars else self.unk_token for ch in word]
            for word in learn_freqs
        }
        total_tokens = sum(len(splits[w]) * freq for w, freq in learn_freqs.items())

        for _ in range(num_merges):
            if scaled_min_tokens is not None and total_tokens <= scaled_min_tokens:
                break

            pair_freqs = Counter()
            for word, freq in learn_freqs.items():
                syms = splits[word]
                for i in range(len(syms) - 1):
                    if syms[i] != self.unk_token and syms[i + 1] != self.unk_token:
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

        symbols = [ch if ch in self.valid_chars else self.unk_token for ch in word]
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
        if len(self._encode_cache) < 100000:
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

    if char_count >= config["val_split_min_chars"]:
        val_chars = min(24000, int(0.08 * char_count))
        split_at = char_count - val_chars
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
        min_char_freq = 1 if train_chars <= 50000 else min(5, max(2, train_chars // 250000))
    else:
        target_merges = config["bpe_merges"]
        min_pair_freq = 2
        min_tokens_floor = None
        min_char_freq = 1

    tok = SimpleBPETokenizer(
        train_text,
        num_merges=target_merges,
        min_freq=min_pair_freq,
        min_tokens=min_tokens_floor,
        min_char_freq=min_char_freq,
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
        val_text = None

    comp_ratio = train_chars / max(1, num_tokens)

    if config.get("auto_size", True):
        n_steps = num_tokens - 1

        # 2.5 * used_vocab codebook up to hardware max_symbols (2048 on 16GB GPU)
        target_by_vocab = int(round(2.5 * used_vocab))
        target_by_tokens = max(config["min_symbols"], int(round(0.90 * num_tokens)))
        raw_symbols = min(target_by_vocab, target_by_tokens)
        snapped_symbols = int(round(raw_symbols / 8.0)) * 8
        n_symbols = max(config["min_symbols"], min(config["max_symbols"], snapped_symbols))

        if n_steps <= 4096:
            epoch_cap = n_steps
        else:
            max_cap = 32768 if HIGH_VRAM else (16384 if DEVICE == "cuda" else 4096)
            epoch_cap = max(4096, min(max_cap, n_steps // 8))

        # Feed Tensor Cores with wide parallel batches (512x64 on 16GB GPU)
        if n_steps < 48:
            batch_size = 1
        elif n_steps < 512:
            batch_size = min(4, max(2, n_steps // 32))
        elif n_steps < 4096:
            batch_size = 32 if DEVICE == "cuda" else 8
        elif n_steps < 16384:
            batch_size = 128 if DEVICE == "cuda" else 16
        else:
            batch_size = 512 if HIGH_VRAM else (256 if DEVICE == "cuda" else 16)

        epoch_window = min(n_steps, epoch_cap)
        stream_len = max(8, min(epoch_window, n_steps) // batch_size)
        if n_steps >= 4096:
            adj_seq_len = max(8, min(16, stream_len // 4))
        else:
            adj_seq_len = max(4, min(24, max(8, stream_len // 2)))

        raw_dim = int(round(8.0 * math.sqrt(max(16, v_size)) / 16.0)) * 16
        emb_dim = max(32, min(config["max_dim"], raw_dim))

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
        adj_grad_clip = float(1.0 * math.sqrt(emb_dim / 64.0))
        adj_ponder = float(0.0008 / max(1, rec_depth))
        adj_eth_weight = float(0.12 * math.sqrt(math.log(max(2, n_symbols)) / math.log(64.0)))
        adj_diag_tokens = min(n_steps, max(2000, 10 * n_symbols))

        tokens_per_epoch = batch_size * stream_len
        chunks_per_epoch = max(1, math.ceil(stream_len / adj_seq_len))
        max_ep_cap = 2000 if n_steps < 4096 else 1000
        target_opt_steps = min(40000, int(round(4000 * (max(1, n_steps) / 150.0) ** 0.35)))
        epochs_by_steps = target_opt_steps // chunks_per_epoch
        min_pass_epochs = int(math.ceil(4.0 * n_steps / max(1, tokens_per_epoch))) if n_steps > epoch_window else 150
        epochs = max(min_pass_epochs, min(max_ep_cap, epochs_by_steps))
        epochs = max(150, min(max_ep_cap, epochs))
        warmup_epochs = max(5, int(round(0.08 * epochs)))

        config["bpe_merges"] = len(tok.merges)
        config["n_symbols"] = n_symbols
        config["epoch_token_cap"] = epoch_cap
        config["batch_size"] = batch_size
        config["seq_len"] = adj_seq_len
        config["embedding_dim"] = emb_dim
        config["stack_size"] = stack_size
        config["max_recursion_depth"] = rec_depth
        config["learning_rate"] = adj_lr
        config["weight_decay"] = adj_wd
        config["grad_clip"] = adj_grad_clip
        config["ponder_penalty"] = adj_ponder
        config["ethical_weight"] = adj_eth_weight
        config["vq_temp"] = adj_temp
        config["diag_max_tokens"] = adj_diag_tokens
        config["epochs"] = epochs
        config["warmup_epochs"] = warmup_epochs
        config["log_every"] = max(1, epochs // 40)
        config["sample_every"] = max(1, epochs // 10)
        config["revive_every"] = max(1, min(5, epochs // 30))
        config["comp_ratio"] = comp_ratio

    train_tensor = torch.tensor(encoded_train, dtype=torch.long)
    val_tensor = torch.tensor(encoded_val, dtype=torch.long) if encoded_val is not None else None

    n_steps = len(encoded_train) - 1
    bs = config["batch_size"]
    stream_len = max(1, min(n_steps, config["epoch_token_cap"]) // bs)
    est_passes = (config["epochs"] * bs * stream_len) / max(1, n_steps)

    print("=== Auto-Configuration Summary (SACRSN v40) ===")
    print(f"Corpus       : {char_count:,} chars ({unique_chars} base chars, min_char_freq>={min_char_freq}) | Train={len(encoded_train):,} toks"
          + (f" | Val={len(encoded_val):,} toks" if encoded_val is not None else " | Val=None (small text)"))
    print(
        f"Tokenizer    : Subword BPE | Vocab={v_size} (Used={used_vocab}) | Merges={len(tok.merges)}/{target_merges} "
        f"(min_freq>={min_pair_freq}) | Comp={comp_ratio:.2f}x"
    )
    print(
        f"Architecture : Dim={config['embedding_dim']} (2D={config['embedding_dim'] * 2}) | "
        f"Codebook={config['n_symbols']} (~{config['n_symbols'] / max(1, used_vocab):.2f}x used vocab) | "
        f"Stack={config['stack_size']} | MaxDepth={config['max_recursion_depth']}"
    )
    print(
        f"Schedule     : Streams={bs}x{stream_len} (Cap={config['epoch_token_cap']:,}, TBPTT={config['seq_len']}) | "
        f"MaxEpochs={config['epochs']} (MinReady={config['warmup_epochs']}, ~{est_passes:.1f}x corpus passes)"
    )
    print(
        f"InitHyper    : LR={config['learning_rate']:.5f} (Plateau) | WD={config['weight_decay']:.1e} | "
        f"AutoClipInit={config['grad_clip']:.2f} | EthW={config['ethical_weight']:.3f} | InitTemp={config['vq_temp']:.3f}"
    )
    print(f"Sample tokens: {[tok.format_token(tok.id_to_token[i]) for i in encoded_train[:12]]}")
    print("===============================================")

    return tok, v_size, train_tensor, val_tensor, val_text


tokenizer, vocab_size, data_tensor, val_tensor, val_text_str = auto_configure_from_text(TEXT_DATA, CONFIG)

# ==========================================
# 3. Complex Primitives (Native Real-Tuple BF16-Ready)
# ==========================================
class ComplexLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift_real = nn.Parameter(torch.zeros(dim))
        self.shift_imag = nn.Parameter(torch.zeros(dim))

    def forward(self, z_r, z_i):
        rms = torch.sqrt(torch.mean(z_r.float() ** 2 + z_i.float() ** 2, dim=-1, keepdim=True) + 1e-4).to(z_r.dtype)
        out_r = (z_r / rms) * self.scale + self.shift_real
        out_i = (z_i / rms) * self.scale + self.shift_imag
        return out_r, out_i

class ModReLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, z_r, z_i):
        norm = torch.sqrt(z_r.float() ** 2 + z_i.float() ** 2 + 1e-3).to(z_r.dtype)
        scale = F.relu(norm + self.bias) / norm
        return z_r * scale, z_i * scale

class ComplexLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc_real = nn.Linear(dim, dim, bias=False)
        self.fc_imag = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.fc_real.weight)
        nn.init.orthogonal_(self.fc_imag.weight)

    def forward(self, z_r, z_i):
        return (
            self.fc_real(z_r) - self.fc_imag(z_i),
            self.fc_real(z_i) + self.fc_imag(z_r),
        )

# ==========================================
# 4. Memory Modules
# ==========================================
class DifferentiableStack(nn.Module):
    def __init__(self, dim, size):
        super().__init__()
        self.dim = dim
        self.size = size

    def forward(self, z_r, z_i, memory, ptr, control):
        # Keep stack state updates in FP32 across multi-step TBPTT
        ctrl_f = control.float()
        push = ctrl_f[:, 0:1]
        pop = ctrl_f[:, 1:2]
        noop = ctrl_f[:, 2:3]

        ptr_up = torch.roll(ptr, 1, dims=1)
        ptr_down = torch.roll(ptr, -1, dims=1)
        new_ptr = (push * ptr_up) + (pop * ptr_down) + (noop * ptr)
        new_ptr = new_ptr / (new_ptr.sum(dim=1, keepdim=True) + 1e-6)

        z_flat = torch.cat([z_r.float(), z_i.float()], dim=-1)
        write_mask = push * ptr_up
        write_val = write_mask.unsqueeze(2) * z_flat.unsqueeze(1)
        retain_mask = 1.0 - write_mask.unsqueeze(2)
        new_memory = write_val + (memory * retain_mask)

        read_mask = new_ptr.unsqueeze(2)
        read_flat = torch.sum(new_memory * read_mask, dim=1)
        read_r = read_flat[:, :self.dim].to(z_r.dtype)
        read_i = read_flat[:, self.dim:].to(z_i.dtype)

        return read_r, read_i, new_memory, new_ptr

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
        self.register_buffer("ema_d_std", torch.tensor(float(CONFIG["vq_temp"]) / 1.2))

    @property
    def live_temp(self):
        return float(torch.clamp(self.ema_d_std * 1.2, min=0.05, max=1.0).item())

    def forward(self, z_r, z_i, prev_symbol_idx=None, bias_scale=0.0, commit_cost=None, active_mask=None):
        if commit_cost is None:
            commit_cost = CONFIG["commitment_cost"]

        dim = z_r.shape[-1]
        # Keep VQ distances & losses in FP32 for exact geometry under BF16 autocast
        z_raw = torch.cat([z_r.float(), z_i.float()], dim=-1)
        cb_f = self.codebook.float()

        z_flat = z_raw / torch.sqrt(torch.mean(z_raw ** 2, dim=-1, keepdim=True) + 1e-4)
        cb_rms = torch.sqrt(torch.mean(cb_f ** 2, dim=-1, keepdim=True) + 1e-4)
        cb_norm = cb_f / cb_rms

        d_raw = (
            torch.sum(z_flat ** 2, dim=-1, keepdim=True)
            + torch.sum(cb_norm ** 2, dim=-1)
            - 2.0 * torch.matmul(z_flat, cb_norm.t())
        ) / z_flat.shape[-1]

        if self.training and d_raw.numel() > 1:
            with torch.no_grad():
                batch_d_std = d_raw.detach().std(dim=-1).mean().clamp(min=0.02, max=2.0)
                self.ema_d_std.mul_(0.95).add_(0.05 * batch_d_std)

        temp_tensor = torch.clamp(self.ema_d_std * 1.2, min=0.05, max=1.0)
        soft_probs = F.softmax(-d_raw / temp_tensor, dim=-1)

        d = d_raw
        if prev_symbol_idx is not None and bias_scale > 0.0:
            graph_prior = self.adjacency[prev_symbol_idx].float()
            bias = bias_scale * torch.sigmoid(graph_prior)
            d = d - bias

        min_indices = torch.argmin(d, dim=-1)

        if self.training:
            with torch.no_grad():
                if active_mask is None:
                    weights = torch.ones_like(min_indices.view(-1), dtype=self.usage_count.dtype)
                else:
                    weights = active_mask.view(-1).to(self.usage_count.dtype)
                self.usage_count.index_add_(0, min_indices.view(-1), weights)

        z_q = F.embedding(min_indices, cb_f)

        # Per-sample VQ loss so active_mask cleanly filters halted batch entries without syncs
        loss_vq_vec = torch.mean((z_q - z_flat.detach()) ** 2, dim=-1)
        loss_commit_vec = torch.mean((z_q.detach() - z_flat) ** 2, dim=-1)
        vq_loss_vec = loss_vq_vec + commit_cost * loss_commit_vec

        z_q_ste = z_flat + (z_q - z_flat).detach()
        out_dtype = z_r.dtype
        sym_r = z_q_ste[..., :dim].to(out_dtype)
        sym_i = z_q_ste[..., dim:].to(out_dtype)
        norm_r = z_flat[..., :dim].to(out_dtype)
        norm_i = z_flat[..., dim:].to(out_dtype)

        return (
            sym_r,
            sym_i,
            norm_r,
            norm_i,
            vq_loss_vec,
            min_indices,
            soft_probs,
            z_flat.detach(),
        )

# ==========================================
# 5. Core Processor (Stack-Attentive Hermitian Attention)
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

    def forward(self, z_r, z_i, stack_mem=None):
        B = z_r.size(0)
        q_r, q_i = self.q_proj(z_r, z_i)  # (B, D)

        # Attend over current state + all stack memory slots for associative recall
        if stack_mem is not None:
            mem_r = stack_mem[..., :self.dim].to(z_r.dtype)
            mem_i = stack_mem[..., self.dim:].to(z_i.dtype)
            all_r = torch.cat([z_r.unsqueeze(1), mem_r], dim=1)  # (B, S+1, D)
            all_i = torch.cat([z_i.unsqueeze(1), mem_i], dim=1)  # (B, S+1, D)
        else:
            all_r = z_r.unsqueeze(1)
            all_i = z_i.unsqueeze(1)

        M = all_r.size(1)
        k_r, k_i = self.k_proj(all_r, all_i)
        v_r, v_i = self.v_proj(all_r, all_i)

        H, hd = self.n_heads, self.head_dim
        q_h = torch.cat([q_r.view(B, H, 1, hd), q_i.view(B, H, 1, hd)], dim=-1)  # (B, H, 1, 2*hd)
        k_h = torch.cat(
            [k_r.view(B, M, H, hd).transpose(1, 2), k_i.view(B, M, H, hd).transpose(1, 2)],
            dim=-1,
        )  # (B, H, M, 2*hd)

        attn_scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * self.scale  # (B, H, 1, M)
        attn_weights = F.softmax(attn_scores.float(), dim=-1).to(v_r.dtype)

        v_r_h = v_r.view(B, M, H, hd).transpose(1, 2)  # (B, H, M, hd)
        v_i_h = v_i.view(B, M, H, hd).transpose(1, 2)

        out_r = torch.matmul(attn_weights, v_r_h).squeeze(2).reshape(B, self.dim)
        out_i = torch.matmul(attn_weights, v_i_h).squeeze(2).reshape(B, self.dim)

        return z_r + 0.5 * out_r, z_i + 0.5 * out_i

class EthicalConstraint(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prev_sym, curr_sym, adjacency):
        if prev_sym is None:
            return torch.zeros((), device=adjacency.device, dtype=torch.float32)
        row_logits = adjacency[prev_sym].float()
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

    def forward(self, z_r, z_i, stack_mem=None):
        l_r, l_i = self.linear(z_r, z_i)
        n_r, n_i = self.norm(l_r, l_i)
        a_r, a_i = self.act(n_r, n_i)
        p_r, p_i = self.attention(a_r, a_i, stack_mem)
        z_flat = torch.cat([p_r, p_i], dim=-1)
        halt_prob = torch.sigmoid(self.halt_linear(z_flat).float())
        stack_probs = F.softmax(self.stack_ctrl(z_flat).float(), dim=-1)
        return p_r, p_i, halt_prob, stack_probs

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
        self.register_buffer("sym_alpha", torch.tensor(0.65))

    def embed(self, idx):
        r = self.emb_mag(idx)
        t = self.emb_phase[idx]
        return r * torch.cos(t), r * torch.sin(t)

    def forward(self, input_ids, hidden=None, prev_sym=None, bias_scale=None, commit_cost=None, return_stats=False):
        if bias_scale is None:
            bias_scale = CONFIG["graph_bias_scale"]

        batch_size = input_ids.size(0)
        z_r, z_i = self.embed(input_ids)
        z_r = z_r.squeeze(1)
        z_i = z_i.squeeze(1)

        if hidden is None:
            stack_mem = torch.zeros(batch_size, CONFIG["stack_size"], self.dim * 2, device=z_r.device, dtype=torch.float32)
            stack_ptr = torch.zeros(batch_size, CONFIG["stack_size"], device=z_r.device, dtype=torch.float32)
            stack_ptr[:, 0] = 1.0
        else:
            prev_r, prev_i, stack_mem, stack_ptr = hidden
            z_r = 0.5 * z_r + 0.5 * prev_r.to(z_r.dtype)
            z_i = 0.5 * z_i + 0.5 * prev_i.to(z_i.dtype)

        halting_probability = torch.zeros(batch_size, 1, device=z_r.device, dtype=torch.float32)
        remain = torch.ones(batch_size, 1, device=z_r.device, dtype=torch.float32)
        ponder_cost = torch.zeros((), device=z_r.device, dtype=torch.float32)
        stack_depth_sum = torch.zeros((), device=z_r.device, dtype=torch.float32)
        stack_ctrl_sum = torch.zeros(3, device=z_r.device, dtype=torch.float32)

        step_soft_probs = []
        step_active_weights = []
        t0_z_flats = None

        z_w_r = torch.zeros(batch_size, self.dim, device=z_r.device, dtype=z_r.dtype)
        z_w_i = torch.zeros(batch_size, self.dim, device=z_i.device, dtype=z_i.dtype)
        current_sym = prev_sym
        vq_loss_sum = torch.zeros((), device=z_r.device, dtype=torch.float32)
        active_step_total = torch.zeros((), device=z_r.device, dtype=torch.float32)
        alpha = self.sym_alpha.to(z_r.dtype)

        if CONFIG["use_stack"]:
            slot_indices = torch.arange(CONFIG["stack_size"], device=z_r.device, dtype=torch.float32)

        for t in range(CONFIG["max_recursion_depth"]):
            still_running = (halting_probability < CONFIG["act_threshold"]).float()
            # Only perform early-break CPU sync during single-sample eval/inference, zero syncs in training
            if (not self.training) and batch_size == 1 and still_running.max().item() == 0.0:
                break

            active_mask = still_running.squeeze(-1)  # (B,)
            active_count = active_mask.sum()

            mem_arg = stack_mem if CONFIG["use_stack"] else None
            p_r, p_i, p_halt, stack_ctrl = self.cell(z_r, z_i, mem_arg)
            stack_ctrl_sum = stack_ctrl_sum + (stack_ctrl.detach() * still_running).sum(dim=0)

            if CONFIG["use_stack"]:
                read_r, read_i, new_mem, new_ptr = self.stack(p_r, p_i, stack_mem, stack_ptr, stack_ctrl)
                # Preserve exact memory state for halted batch entries
                run_3d = still_running.unsqueeze(-1)
                stack_mem = new_mem * run_3d + stack_mem * (1.0 - run_3d)
                stack_ptr = new_ptr * still_running + stack_ptr * (1.0 - still_running)
                comb_r = p_r + read_r
                comb_i = p_i + read_i
                depth_vec = torch.sum(stack_ptr * slot_indices, dim=1)
                stack_depth_sum = stack_depth_sum + (depth_vec * active_mask).sum()
            else:
                comb_r, comb_i = p_r, p_i

            prior_sym = prev_sym if t == 0 else None
            step_bias = bias_scale if t == 0 else 0.0

            sym_r, sym_i, norm_r, norm_i, vq_loss_vec, sym_idx, soft_probs, z_flat_det = self.vq_layer(
                comb_r, comb_i, prior_sym, bias_scale=step_bias, commit_cost=commit_cost, active_mask=active_mask
            )
            if t == 0:
                t0_z_flats = z_flat_det

            # Preserve symbol selection for halted batch entries
            if current_sym is None:
                current_sym = sym_idx
            else:
                active_bool = active_mask > 0.5
                current_sym = torch.where(active_bool, sym_idx, current_sym)

            step_soft_probs.append(soft_probs * still_running)
            step_active_weights.append(active_mask)

            z_r = (1.0 - alpha) * norm_r + alpha * sym_r
            z_i = (1.0 - alpha) * norm_i + alpha * sym_i

            new_halted = ((halting_probability + p_halt * still_running) >= CONFIG["act_threshold"]).float() * still_running
            still_running_after = still_running - new_halted

            if t == CONFIG["max_recursion_depth"] - 1:
                p = remain * still_running
            else:
                p = p_halt * still_running_after + remain * new_halted

            halting_probability = halting_probability + p
            remain = remain - p

            p_cast = p.to(z_r.dtype)
            z_w_r = z_w_r + (p_cast * z_r)
            z_w_i = z_w_i + (p_cast * z_i)
            ponder_cost = ponder_cost + still_running.mean()
            vq_loss_sum = vq_loss_sum + (vq_loss_vec * active_mask).sum()
            active_step_total = active_step_total + active_count

        denom = active_step_total.clamp(min=1.0)
        vq_loss_total = vq_loss_sum / denom
        avg_stack = stack_depth_sum / denom
        avg_stack_ctrl = stack_ctrl_sum / denom
        ethical_loss = self.ethics(prev_sym, current_sym, self.vq_layer.adjacency)

        features = torch.cat([z_w_r, z_w_i], dim=-1)
        logits = self.decoder(features)

        next_hidden = (z_w_r, z_w_i, stack_mem, stack_ptr)

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
                torch.cat(step_active_weights, dim=0),
                t0_z_flats,
                avg_stack_ctrl,
            )

        return logits, next_hidden, current_sym, ponder_cost, vq_loss_total, ethical_loss, avg_stack

# ==========================================
# 7. Training, Validation & Self-Tuning Engine
# ==========================================
def save_checkpoint(model, path):
    if path is None:
        return
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'vocab': tokenizer.vocab,
        'merges': tokenizer.merges,
    }, path)


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
    return streams.to(DEVICE, non_blocking=True)


@torch.no_grad()
def evaluate_val(model, val_cpu_tensor):
    if val_cpu_tensor is None or len(val_cpu_tensor) < 4:
        return None, None
    model.eval()
    n_val = len(val_cpu_tensor) - 1
    b_val = min(64, max(1, n_val // 64))
    s_len = min(64, n_val // b_val)
    streams = torch.stack(
        [val_cpu_tensor[b * s_len : b * s_len + s_len + 1] for b in range(b_val)], dim=0
    ).to(DEVICE)

    hidden, prev_sym = None, None
    total_loss_t = torch.zeros((), device=DEVICE)
    total_correct_t = torch.zeros((), dtype=torch.long, device=DEVICE)
    with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=USE_BF16):
        for t in range(s_len):
            x = streams[:, t].unsqueeze(1)
            y = streams[:, t + 1]
            logits, hidden, prev_sym, _, _, _, _ = model(x, hidden, prev_sym)
            total_loss_t += F.cross_entropy(logits.float(), y)
            total_correct_t += (logits.argmax(dim=-1) == y).sum()

    model.train()
    return total_loss_t.item() / max(1, s_len), 100.0 * total_correct_t.item() / max(1, b_val * s_len)


def train(save_path="crsn_omni_model.pth"):
    model = UberCRSN(vocab_size, CONFIG["embedding_dim"]).to(DEVICE)

    sym_scale = math.sqrt(max(1.0, CONFIG["n_symbols"] / 64.0))
    cb_lr = CONFIG["learning_rate"] * min(1.5, sym_scale)
    adj_lr = CONFIG["learning_rate"] * 3.0 * min(2.0, sym_scale)

    codebook_params = [model.vq_layer.codebook]
    adj_params = [model.vq_layer.adjacency]
    other_params = [
        p for n, p in model.named_parameters()
        if n not in ("vq_layer.codebook", "vq_layer.adjacency")
    ]
    param_groups = [
        {"params": other_params, "weight_decay": CONFIG["weight_decay"], "lr": CONFIG["learning_rate"]},
        {"params": codebook_params, "weight_decay": 0.0, "lr": cb_lr},
        {"params": adj_params, "weight_decay": 0.0, "lr": adj_lr},
    ]
    try:
        opt = torch.optim.AdamW(param_groups, fused=(DEVICE == "cuda"))
        fused_used = (DEVICE == "cuda")
    except Exception:
        opt = torch.optim.AdamW(param_groups)
        fused_used = False

    base_lrs = [pg["lr"] for pg in opt.param_groups]
    min_lrs = [max(1e-5, lr * 0.01) for lr in base_lrs]
    lr_warmup_epochs = max(3, min(15, CONFIG["epochs"] // 30))
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=CONFIG["lr_patience"],
        min_lr=min_lrs,
    )

    log_K = float(math.log(CONFIG["n_symbols"]))
    t_start = time.time()
    revived_total = 0
    last_ep_revived = 0
    min_active_converge = max(4, min(16, CONFIG["n_symbols"] // 4))
    best_metric = float("inf")
    best_state = None
    no_improve_logs = 0

    live_div_weight = CONFIG["diversity_weight"]
    live_commit_cost = CONFIG["commitment_cost"]
    live_ponder_penalty = CONFIG["ponder_penalty"]
    ema_gnorm = CONFIG["grad_clip"]
    readiness = 0.0
    ema_pred = None

    act_low = 1.25
    act_high = max(2.2, 0.65 * CONFIG["max_recursion_depth"])

    revive_accum_counts = torch.zeros(CONFIG["n_symbols"], device=DEVICE)
    revive_lookups_accum = 0
    min_revive_lookups = max(512, 8 * CONFIG["n_symbols"])
    epoch_z_pool = []
    tokens_since_log = 0
    clipped_chunks_since_log = 0
    total_chunks_since_log = 0

    print(f"--- Training SACRSN v40 (BF16={USE_BF16} | FusedAdamW={fused_used} | MultiStep-TBPTT) ---")

    try:
        for epoch in range(CONFIG["epochs"]):
            model.train()

            if epoch < lr_warmup_epochs:
                lr_scale = (epoch + 1) / float(lr_warmup_epochs)
                for pg, blr in zip(opt.param_groups, base_lrs):
                    pg["lr"] = blr * lr_scale

            streams = build_epoch_streams(
                data_tensor, CONFIG["batch_size"], CONFIG["epoch_token_cap"], epoch
            )
            bs, span_len = streams.shape
            stream_len = span_len - 1

            hidden = None
            prev_sym = None

            # Zero-sync GPU accumulators (read .item() once per epoch)
            total_loss_t = torch.zeros((), device=DEVICE)
            total_pred_t = torch.zeros((), device=DEVICE)
            total_vq_t = torch.zeros((), device=DEVICE)
            total_eth_t = torch.zeros((), device=DEVICE)
            total_div_t = torch.zeros((), device=DEVICE)
            total_ponder_t = torch.zeros((), device=DEVICE)
            total_stack_t = torch.zeros((), device=DEVICE)
            stack_ops_epoch = torch.zeros(3, device=DEVICE)
            total_correct_t = torch.zeros((), dtype=torch.long, device=DEVICE)
            self_loops_t = torch.zeros((), dtype=torch.long, device=DEVICE)
            trans_toks = 0
            total_toks = 0
            grad_norms = []

            progress = epoch / max(1, CONFIG["epochs"] - 1)
            entropy_weight = 0.01 * (1.0 - progress)

            live_bias_scale = float(torch.clamp(model.vq_layer.ema_d_std * 0.30, min=0.02, max=0.25).item()) * readiness
            live_eth_weight = CONFIG["ethical_weight"] * readiness

            in_revive_phase = epoch < int(0.50 * CONFIG["epochs"])

            chunk_loss = 0.0
            chunk_probs = []
            chunk_weights = []
            steps_in_chunk = 0

            for t in range(stream_len):
                x = streams[:, t].unsqueeze(1)
                y = streams[:, t + 1]

                with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=USE_BF16):
                    logits, hidden, sym_idx, ponder, vq_loss, eth_loss, avg_stack, soft_probs, active_w, t0_z_flats, avg_stk_ctrl = model(
                        x,
                        hidden,
                        prev_sym,
                        bias_scale=live_bias_scale,
                        commit_cost=live_commit_cost,
                        return_stats=True,
                    )

                    loss_pred = F.cross_entropy(logits.float(), y)
                    loss_ponder = live_ponder_penalty * ponder

                    probs = F.softmax(logits.float(), dim=-1)
                    loss_entropy = -entropy_weight * (-(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean())

                    loss_vq = CONFIG["vq_weight"] * vq_loss
                    loss_ethics = live_eth_weight * eth_loss

                    step_loss = loss_pred + loss_ponder + loss_vq + loss_entropy + loss_ethics

                if prev_sym is not None:
                    self_loops_t += (prev_sym == sym_idx).sum()
                    trans_toks += bs

                # Keep hidden attached across the TBPTT chunk so Stack learns multi-token credit
                prev_sym = sym_idx.detach()

                chunk_loss = chunk_loss + step_loss
                chunk_probs.append(soft_probs)
                chunk_weights.append(active_w)
                if in_revive_phase and len(epoch_z_pool) < 64:
                    epoch_z_pool.append(t0_z_flats)
                steps_in_chunk += 1

                total_pred_t += loss_pred.detach()
                total_vq_t += vq_loss.detach()
                total_eth_t += eth_loss.detach()
                total_ponder_t += ponder.detach()
                total_stack_t += avg_stack.detach()
                stack_ops_epoch += avg_stk_ctrl
                total_correct_t += (logits.detach().argmax(dim=-1) == y).sum()
                total_toks += bs
                tokens_since_log += bs

                if steps_in_chunk == CONFIG["seq_len"] or t == stream_len - 1:
                    probs_cat = torch.cat(chunk_probs, dim=0)
                    w_cat = torch.cat(chunk_weights, dim=0)
                    w_sum = w_cat.sum().clamp(min=1.0)

                    avg_probs = probs_cat.sum(dim=0) / w_sum
                    marginal_entropy = -(avg_probs * torch.log(avg_probs + 1e-8)).sum()
                    norm_kl = (log_K - marginal_entropy) / max(1e-8, log_K)
                    sample_entropy = -(probs_cat * torch.log(probs_cat + 1e-8)).sum() / w_sum
                    norm_sample_ent = sample_entropy / max(1e-8, log_K)
                    loss_diversity = live_div_weight * (norm_kl + 0.5 * norm_sample_ent)

                    total_chunk_loss = (chunk_loss / steps_in_chunk) + loss_diversity

                    clip_thresh = max(0.5, min(5.0, 2.5 * ema_gnorm))
                    opt.zero_grad(set_to_none=True)
                    total_chunk_loss.backward()
                    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_thresh).item()
                    if math.isfinite(gnorm):
                        ema_gnorm = 0.95 * ema_gnorm + 0.05 * min(gnorm, 10.0)
                        if gnorm > clip_thresh:
                            clipped_chunks_since_log += 1
                    total_chunks_since_log += 1
                    grad_norms.append(gnorm)
                    opt.step()

                    # Detach hidden ONLY at TBPTT chunk boundaries
                    h_r, h_i, h_mem, h_ptr = hidden
                    hidden = (h_r.detach(), h_i.detach(), h_mem.detach(), h_ptr.detach())

                    total_loss_t += total_chunk_loss.detach() * steps_in_chunk
                    total_div_t += loss_diversity.detach() * steps_in_chunk
                    chunk_loss = 0.0
                    chunk_probs = []
                    chunk_weights = []
                    steps_in_chunk = 0

            avg_pred = total_pred_t.item() / stream_len
            avg_vq = total_vq_t.item() / stream_len
            avg_ponder = total_ponder_t.item() / stream_len
            ema_pred = avg_pred if ema_pred is None else (0.80 * ema_pred + 0.20 * avg_pred)

            with torch.no_grad():
                counts = model.vq_layer.usage_count.clone()
                active_syms = int((counts > 0).sum().item())
                dist = counts / (counts.sum() + 1e-10)
                model.prev_sym_soft.copy_(dist)
                entropy_val = -(dist * torch.log(dist + 1e-10)).sum()
                true_ppx = torch.exp(entropy_val).item()
                top1_pct = float((counts.max() / (counts.sum() + 1e-10) * 100.0).item())

                revive_accum_counts.add_(counts)
                revive_lookups_accum += int(counts.sum().item())

                ep_revived = 0
                if (
                    in_revive_phase
                    and epoch % CONFIG["revive_every"] == 0
                    and revive_lookups_accum >= min_revive_lookups
                    and len(epoch_z_pool) > 0
                ):
                    dead_mask = revive_accum_counts == 0
                    n_dead = int(dead_mask.sum().item())
                    if n_dead > 0:
                        pool = torch.cat(epoch_z_pool, dim=0)
                        rand_idx = torch.randint(0, pool.size(0), (n_dead,), device=DEVICE)
                        new_vecs = pool[rand_idx] + 0.05 * torch.randn(n_dead, pool.size(1), device=DEVICE)
                        new_vecs = new_vecs / torch.sqrt(torch.mean(new_vecs ** 2, dim=-1, keepdim=True) + 1e-4)
                        model.vq_layer.codebook.data[dead_mask] = new_vecs
                        ep_revived = n_dead
                        revived_total += n_dead
                    revive_accum_counts.zero_()
                    revive_lookups_accum = 0
                    epoch_z_pool.clear()
                elif not in_revive_phase and len(epoch_z_pool) > 0:
                    epoch_z_pool.clear()

                last_ep_revived = ep_revived
                model.vq_layer.usage_count.zero_()

                # Closed-Loop Self-Tuning Updates
                util_ratio = true_ppx / max(1, CONFIG["n_symbols"])

                if util_ratio < 0.35 or top1_pct > 12.0:
                    live_div_weight = min(3.0, live_div_weight * 1.08)
                elif util_ratio > 0.55 and top1_pct < 6.0 and active_syms > 0.70 * CONFIG["n_symbols"]:
                    live_div_weight = max(0.03, live_div_weight * 0.96)

                if avg_vq > 0.65:
                    live_commit_cost = min(0.60, live_commit_cost * 1.04)
                elif avg_vq < 0.20 and avg_pred > 1.5:
                    live_commit_cost = max(0.08, live_commit_cost * 0.96)

                if avg_ponder > act_high:
                    live_ponder_penalty = min(0.01, live_ponder_penalty * 1.05)
                elif avg_ponder < act_low:
                    live_ponder_penalty = max(1e-6, live_ponder_penalty * 0.92)

                target_alpha = min(0.95, max(0.60, 0.95 - 0.55 * max(0.0, avg_vq - 0.25)))
                model.sym_alpha.mul_(0.85).add_(0.15 * target_alpha)

                u_score = min(1.0, util_ratio / 0.40)
                vq_score = min(1.0, max(0.0, (1.2 - avg_vq) / 0.6))
                rev_score = max(0.2, 1.0 - last_ep_revived / max(1.0, 0.25 * CONFIG["n_symbols"]))
                min_ramp = min(1.0, (epoch + 1) / max(5, CONFIG["warmup_epochs"] // 2))
                readiness = 0.85 * readiness + 0.15 * (u_score * vq_score * rev_score * min_ramp)

            is_last_epoch = (epoch == CONFIG["epochs"] - 1)
            if epoch % CONFIG["log_every"] == 0 or is_last_epoch:
                val_loss, val_acc = evaluate_val(model, val_tensor)
                monitor_loss = val_loss if val_loss is not None else ema_pred
                is_ready = (readiness >= 0.50 or epoch >= CONFIG["warmup_epochs"]) and active_syms >= min_active_converge

                if epoch >= lr_warmup_epochs and is_ready:
                    plateau_scheduler.step(monitor_loss)

                if is_ready:
                    if monitor_loss < best_metric - 1e-3:
                        best_metric = monitor_loss
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        save_checkpoint(model, save_path)
                        no_improve_logs = 0
                    elif progress >= 0.4:
                        no_improve_logs += 1

                avg_loss = total_loss_t.item() / stream_len
                avg_eth = total_eth_t.item() / stream_len
                avg_div = total_div_t.item() / stream_len
                avg_stk = total_stack_t.item() / stream_len
                stk_ops = (stack_ops_epoch / stream_len * 100.0).tolist()
                acc_pct = 100.0 * total_correct_t.item() / max(1, total_toks)
                tok_ppx = float(np.exp(min(avg_pred, 20.0)))
                loop_pct = 100.0 * self_loops_t.item() / max(1, trans_toks)
                strong_edges = int((F.softmax(model.vq_layer.adjacency.float(), dim=-1) > 0.08).sum().item())
                avg_gnorm = sum(grad_norms) / max(1, len(grad_norms))
                clip_pct = 100.0 * clipped_chunks_since_log / max(1, total_chunks_since_log)
                clip_thresh_now = max(0.5, min(5.0, 2.5 * ema_gnorm))
                dt = max(1e-4, time.time() - t_start)
                tok_per_sec = tokens_since_log / dt
                t_start = time.time()
                tokens_since_log = 0
                clipped_chunks_since_log = 0
                total_chunks_since_log = 0
                lr = opt.param_groups[0]["lr"]

                if val_loss is not None:
                    val_ppx = float(np.exp(min(val_loss, 20.0)))
                    val_str = f" | Val:{val_loss:.3f} ({val_acc:4.1f}% PPX:{val_ppx:5.1f})"
                else:
                    val_str = ""

                print(
                    f"Ep {epoch:04d} | Loss:{avg_loss:.3f} (Pred:{avg_pred:.3f} VQ:{avg_vq:.3f} "
                    f"Div:{avg_div:.3f} Eth:{avg_eth:.3f}) | Acc:{acc_pct:5.1f}% TokPPX:{tok_ppx:5.2f}{val_str} | "
                    f"{tok_per_sec:,.0f} tok/s ({dt:.1f}s)"
                )
                print(
                    f"        | ACT:{avg_ponder:.2f} Stk:{avg_stk:4.1f}({stk_ops[0]:2.0f}/{stk_ops[1]:2.0f}/{stk_ops[2]:2.0f}) | "
                    f"Syms:{active_syms:3d}/{CONFIG['n_symbols']} SymPPX:{true_ppx:5.1f} Top1:{top1_pct:4.1f}% "
                    f"Loop:{loop_pct:4.1f}% Rev:{revived_total:3d} Edges:{strong_edges:4d} | "
                    f"Grad:{avg_gnorm:.2f} LR:{lr:.5f}"
                )
                print(
                    f"        | Tune[Ready:{readiness*100:3.0f}% DivW:{live_div_weight:.2f} "
                    f"Temp:{model.vq_layer.live_temp:.2f} Cmt:{live_commit_cost:.2f} "
                    f"Alpha:{model.sym_alpha.item():.2f} PndP:{live_ponder_penalty:.5f} "
                    f"Clip:{clip_thresh_now:.2f}({clip_pct:2.0f}%)]"
                )
                revived_total = 0

                if epoch % CONFIG["sample_every"] == 0 or is_last_epoch:
                    model.eval()
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=USE_BF16):
                        seed_id = data_tensor[0].item()
                        gx = torch.tensor([[seed_id]], device=DEVICE)
                        gh, gp = None, None
                        sample_ids = [seed_id]
                        preview_toks = max(30, min(60, int(round(80 / max(1.0, CONFIG["comp_ratio"])))))
                        for _ in range(preview_toks):
                            g_logits, gh, gp, _, _, _, _ = model(gx, gh, gp)
                            gx = g_logits.argmax(dim=-1, keepdim=True)
                            sample_ids.append(gx.item())
                    print(f"   Sample: {repr(tokenizer.decode(sample_ids))}")
                    model.train()

                if avg_pred < 0.01 and is_ready:
                    print("\n--- PERFECT CONVERGENCE ---")
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_metric = avg_pred
                    save_checkpoint(model, save_path)
                    break

                if progress >= 0.4 and lr <= min_lrs[0] * 2.1 and no_improve_logs >= CONFIG["early_stop_patience"]:
                    print(f"\n--- EARLY STOPPING (LR converged & plateaued for {no_improve_logs} checks) ---")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    if best_state is not None:
        model.load_state_dict(best_state)
        metric_name = "Val Loss" if val_tensor is not None else "Pred Loss (EMA)"
        print(f"Restored best healthy checkpoint ({metric_name}: {best_metric:.4f}).")

    return model

# ==========================================
# 8. Visualization Suite
# ==========================================
def visualize_all(model):
    print("\n--- Generating Diagnostics & Images ---")
    model.eval()

    diag_steps = min(len(data_tensor) - 1, CONFIG["diag_max_tokens"])

    symbol_to_tok = defaultdict(list)
    hidden, prev_sym = None, None
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=USE_BF16):
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

    adj_probs = F.softmax(model.vq_layer.adjacency.detach().float(), dim=-1).cpu().numpy()
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

    hidden, prev_sym = None, None
    seed_id = data_tensor[0].item()
    x = torch.tensor([[seed_id]], device=DEVICE)
    stack_history, act_history, phase_reals, phase_imags = [], [], [], []
    gen_ids = [seed_id]
    scan_len = max(60, min(120, int(round(180 / max(1.0, CONFIG["comp_ratio"])))))

    print("Running Inference Scan...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=USE_BF16):
        for _ in range(scan_len):
            logits, hidden, prev_sym, ponder, _, _, s_hist = model(x, hidden, prev_sym)

            stack_history.append(s_hist.item())
            act_history.append(ponder.item())

            zr = hidden[0].detach().float().cpu().squeeze()
            zi = hidden[1].detach().float().cpu().squeeze()
            phase_reals.append(zr[0].item() if zr.dim() > 0 else zr.item())
            phase_imags.append(zi[0].item() if zi.dim() > 0 else zi.item())

            probs = F.softmax(logits.float(), dim=-1)
            next_ix = torch.multinomial(probs, 1)
            gen_ids.append(next_ix.item())
            x = next_ix

    act_counts = Counter(int(round(a)) for a in act_history)
    act_dist_str = ", ".join(
        f"{k} steps: {100.0 * v / len(act_history):.0f}%" for k, v in sorted(act_counts.items())
    )
    print(f"ACT Step Breakdown : {act_dist_str}")
    print(f"Stack Depth Range  : [{min(stack_history):.2f}, {max(stack_history):.2f}]")
    print(f"Generated          : {tokenizer.decode(gen_ids)}\n")

    plt.figure(figsize=(12, 4))
    plt.plot(stack_history, color='purple', label='Stack Depth')
    plt.fill_between(range(len(stack_history)), stack_history, color='purple', alpha=0.1)
    plt.title("2_stack_mri (Memory Depth)")
    plt.savefig("2_stack_mri.png")
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(act_history)), act_history, color='orange')
    plt.ylim(0, CONFIG["max_recursion_depth"] + 0.5)
    plt.title("3_act_profile (Thinking Steps)")
    plt.savefig("3_act_profile.png")
    plt.close()

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

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=USE_BF16):
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
    print("\n--- Dream Mode (Entropy-Targeted Symbolic Walk) ---")
    adj_logits = model.vq_layer.adjacency.detach().float()
    model.eval()
    seed_id = data_tensor[0].item()
    x = torch.tensor([[seed_id]], device=DEVICE)
    with torch.no_grad():
        _, _, prev_sym, _, _, _, _ = model(x, None, None)
    curr_sym = prev_sym.item()
    output_ids = [seed_id]

    target_branch_ppx = min(4.0, max(2.0, 0.04 * CONFIG["n_symbols"]))
    dream_temps = []
    dream_ppxs = []

    for _ in range(60):
        row_logits = adj_logits[curr_sym]
        lo, hi = 0.15, 2.50
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            p_mid = F.softmax(row_logits / mid, dim=-1)
            ent_mid = -(p_mid * torch.log(p_mid + 1e-10)).sum()
            ppx_mid = torch.exp(ent_mid).item()
            if ppx_mid < target_branch_ppx:
                lo = mid
            else:
                hi = mid
        best_temp = 0.5 * (lo + hi)

        probs_t = F.softmax(row_logits / best_temp, dim=-1)
        step_ent = -(probs_t * torch.log(probs_t + 1e-10)).sum()
        dream_temps.append(best_temp)
        dream_ppxs.append(torch.exp(step_ent).item())

        probs = probs_t.cpu().numpy()
        sorted_idx = np.argsort(probs)[::-1]
        cum_probs = np.cumsum(probs[sorted_idx])
        cutoff = int(np.searchsorted(cum_probs, 0.90)) + 1
        keep_idx = sorted_idx[:max(1, cutoff)]

        filtered = np.zeros_like(probs)
        filtered[keep_idx] = probs[keep_idx]
        filtered = filtered / filtered.sum()
        next_sym = int(np.random.choice(len(filtered), p=filtered))

        with torch.no_grad():
            z_flat = model.vq_layer.codebook[next_sym].unsqueeze(0).float()
            logits = model.decoder(z_flat)
            tok_idx = torch.argmax(logits).item()

        output_ids.append(tok_idx)
        curr_sym = next_sym

    print(
        f"Dream Stats  : AvgTemp={sum(dream_temps)/len(dream_temps):.2f} | "
        f"AvgBranchPPX={sum(dream_ppxs)/len(dream_ppxs):.2f} (Target={target_branch_ppx:.2f})"
    )
    print(f"Dream Output : {tokenizer.decode(output_ids)}\n")

def anomaly_detector(model):
    print("\n--- Anomaly Detection Test ---")
    model.eval()

    if val_text_str is not None and len(val_text_str) > 200:
        mid = len(val_text_str) // 4
        snippet = val_text_str[mid : mid + 400]
        words = snippet.split()
        if len(words) > 2:
            words = words[1:]
    else:
        words = TEXT_DATA[:400].split()

    prefix = " ".join(words[:min(7, len(words))])
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

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=USE_BF16):
        for i in range(len(input_tensor) - 1):
            x = input_tensor[i].view(1, 1)
            _, hidden, prev_sym, _, _, eth_loss, _ = model(x, hidden, prev_sym)
            anomalies.append(eth_loss.item())

    if len(anomalies) >= 2:
        prefix_ids = tokenizer.encode(prefix)
        split_idx = max(1, min(len(anomalies) - 1, len(prefix_ids) - 1))
        prefix_mean = float(np.mean(anomalies[:split_idx]))
        tail_vals = anomalies[split_idx:]
        tail_mean = float(np.mean(tail_vals))
        tail_max = float(np.max(tail_vals))
        ratio = tail_max / max(1e-6, prefix_mean)
        print(
            f"Anomaly Stats: ValidPrefixMean={prefix_mean:.2f} | "
            f"TailMean={tail_mean:.2f} TailMax={tail_max:.2f} (Contrast={ratio:.2f}x)"
        )

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

    trained_model = train(save_path=FILENAME)

    print(f"\n--- Saving Final Model to {FILENAME} ---")
    save_checkpoint(trained_model, FILENAME)
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