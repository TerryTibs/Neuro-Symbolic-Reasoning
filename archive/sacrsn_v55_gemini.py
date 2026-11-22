# ============================================================
# SACRSN v55 (Unified PonderNet + Clean Stack + Masked GWT + Decoupled Architecture)
# Upgrades from v54:
#   1. Fixed Phase-2 live_eth_weight (uses p2_prog instead of readiness)
#   2. Unified Train/Val/Inference Halting + True PonderNet per-step CE + Mixture logits
#   3. Fixed DifferentiableStack (pre-push read, clean pop erasure, relaxed NoOp cap)
#   4. Masked Ablated Streams in GlobalWorkspaceRouter (exact ablation dLoss)
#   5. Unit-RMS Codebook Projection + AdamW State Reset on K-Means & Revival
#   6. Decoupled CONFIG from Modules + load_checkpoint() + Import-Safe Execution
# ============================================================

import os
import sys
import copy
import atexit
import re
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import networkx as nx
from collections import defaultdict, Counter

# ==========================================
# 0. Hardware Constants & Logger Utilities
# ==========================================
class _TeeStream:
    def __init__(self, orig_stream, log_file):
        self.orig_stream = orig_stream
        self.log_file = log_file

    def write(self, message):
        self.orig_stream.write(message)
        if self.log_file is not None and not self.log_file.closed:
            self.log_file.write(message)
            if "\n" in message:
                self.log_file.flush()

    def flush(self):
        self.orig_stream.flush()
        if self.log_file is not None and not self.log_file.closed:
            self.log_file.flush()

    def isatty(self):
        return getattr(self.orig_stream, "isatty", lambda: False)()


class TerminalLogger:
    """Automatically mirrors all stdout/stderr terminal printouts to a text file."""
    def __init__(self, filepath="terminal outputs.txt"):
        self.filepath = filepath
        self.orig_stdout = sys.stdout
        self.orig_stderr = sys.stderr
        self.log_file = open(filepath, "w", encoding="utf-8", errors="replace")
        self._closed = False
        sys.stdout = _TeeStream(self.orig_stdout, self.log_file)
        sys.stderr = _TeeStream(self.orig_stderr, self.log_file)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout = self.orig_stdout
        sys.stderr = self.orig_stderr
        try:
            if not self.log_file.closed:
                self.log_file.flush()
                self.log_file.close()
        except Exception:
            pass


def setup_terminal_logger(filepath="terminal outputs.txt"):
    logger = TerminalLogger(filepath=filepath)
    atexit.register(logger.close)
    return logger


def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


SEED = 42
TERMINAL_LOG_FILE = "terminal outputs.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
VRAM_GB = (
    torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if DEVICE == "cuda"
    else 0.0
)
USE_BF16 = (DEVICE == "cuda") and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32
HIGH_VRAM = (DEVICE == "cuda") and (VRAM_GB >= 12.0)

# ==========================================
# 1. Default Configuration
# ==========================================
DEFAULT_CONFIG = {
    # Auto-Sizing Controls
    "auto_size": True,
    "min_symbols": 16,
    "max_symbols": 1024 if HIGH_VRAM else (768 if DEVICE == "cuda" else 512),
    "max_bpe_merges": 2048 if HIGH_VRAM else (1536 if DEVICE == "cuda" else 768),
    "max_dim": 288 if HIGH_VRAM else (192 if DEVICE == "cuda" else 128),
    "epoch_token_cap": 32768 if HIGH_VRAM else 4096,
    "diag_max_tokens": 2000,
    "val_split_min_chars": 12000,

    # Sequence & Model Capacity (auto-adjusted when auto_size=True)
    "batch_size": 4,
    "stream_len": None,
    "seq_len": 16,
    "bpe_merges": 32,
    "embedding_dim": 64,
    "n_symbols": 120,
    "stack_size": 16,
    "max_recursion_depth": 4,
    "decoder_dropout": 0.10,

    # Reasoning (PonderNet ACT) & Memory Controls
    "act_threshold": 0.85,
    "ponder_prior_halt": 0.45,
    "ponder_penalty": 0.030,
    "use_stack": True,
    "stack_temp": 0.75,
    "stack_balance_weight": 0.25,
    "holo_decay": 0.99,
    "holo_write_scale": 0.10,

    # Topology, Consistency & Stability
    "commitment_cost": 0.25,
    "vq_weight": 1.0,
    "graph_bias_scale": 0.1,
    "symbol_consistency_weight": 0.02,
    "ethical_weight": 0.12,
    "diversity_weight": 1.2,
    "conscience_p2": 0.12,
    "conscience_p3": 0.08,
    "adj_lr_mult": 8.0,
    "vq_temp": 0.10,
    "target_ppx_low": 25.0,
    "target_ppx_high": 75.0,
    "corpus_tok_ppx": 35.0,

    # 3-Phase Curriculum Fractions
    "phase1_frac": 0.10,
    "phase2_frac": 0.25,

    # Training Schedule (auto-adjusted when auto_size=True)
    "epochs": 260,
    "warmup_epochs": 20,
    "log_every": 10,
    "eval_every": 5,
    "sample_every": 30,
    "revive_every": 3,
    "lr_patience": 4,
    "early_stop_patience": 10,
    "learning_rate": 1.5e-3,
    "weight_decay": 3e-4,
    "grad_clip": 1.0,
    "grad_clip_max": 3.0,
    "eps": 1e-6,
    "comp_ratio": 1.75,

    # Optional Blocksworld/PDDL Symbolic Transition Benchmark
    "run_pddl_probe": False,
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


def load_text_corpus(data_file=DATA_FILE, verbose=True):
    text_data = DEFAULT_TEXT_DATA
    if data_file and os.path.isfile(data_file):
        try:
            with open(data_file, "r", encoding="utf-8-sig", errors="replace") as f:
                loaded_text = f.read()
            if len(loaded_text.strip()) >= 8:
                text_data = loaded_text
                if verbose:
                    print(f"Loaded corpus from '{data_file}' ({len(text_data):,} chars)")
            elif verbose:
                print(f"'{data_file}' is empty/too short — using built-in default text.")
        except OSError as e:
            if verbose:
                print(f"Could not read '{data_file}' ({e}) — using built-in default text.")
    elif verbose:
        print(f"'{data_file}' not found — using built-in default text.")
    return text_data


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

    @classmethod
    def from_saved(cls, vocab, merges):
        obj = cls.__new__(cls)
        obj.pattern = re.compile(r' ?\w+|\n|[^\w\s]|[ \t\r]+')
        obj.unk_token = "<UNK>"
        obj.unk_id = 0
        obj._encode_cache = {}
        obj.vocab = list(vocab)
        obj.merges = [tuple(m) for m in merges]
        obj.token_to_id = {tok: i for i, tok in enumerate(obj.vocab)}
        obj.id_to_token = {i: tok for i, tok in enumerate(obj.vocab)}
        obj.bpe_ranks = {pair: rank for rank, pair in enumerate(obj.merges)}
        obj.vocab_size = len(obj.vocab)
        obj.valid_chars = {tok for tok in obj.vocab if len(tok) == 1 and tok != obj.unk_token}
        return obj

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


def stratified_train_val_split(text, min_chars=12000):
    """Extracts evenly spaced validation slices across the full corpus to prevent tail-domain shift."""
    char_count = len(text)
    if char_count < min_chars:
        return text, None

    val_chars_total = min(32000, int(0.08 * char_count))
    n_slices = min(8, max(1, val_chars_total // 2000))
    if n_slices <= 1:
        split_at = char_count - val_chars_total
        ws_pos = text.find(" ", split_at)
        if ws_pos != -1 and (ws_pos - split_at) < 100:
            split_at = ws_pos + 1
        return text[:split_at], text[split_at:]

    region_len = char_count // n_slices
    slice_len = val_chars_total // n_slices
    train_parts = []
    val_parts = []
    prev_end = 0

    for i in range(n_slices):
        r_end = (i + 1) * region_len if i < n_slices - 1 else char_count
        raw_s = max(prev_end, r_end - slice_len - 100)
        raw_e = min(r_end, raw_s + slice_len)

        ws_s = text.find(" ", raw_s)
        s = (ws_s + 1) if (ws_s != -1 and (ws_s - raw_s) < 100 and ws_s + 1 < raw_e) else raw_s

        ws_e = text.find(" ", raw_e)
        e = (ws_e + 1) if (ws_e != -1 and (ws_e - raw_e) < 100 and ws_e + 1 <= r_end) else raw_e

        if e > s:
            if s > prev_end:
                train_parts.append(text[prev_end:s])
            val_parts.append(text[s:e])
            prev_end = e

    if prev_end < char_count:
        train_parts.append(text[prev_end:])

    train_text = "\n".join(train_parts)
    val_text = "\n".join(val_parts)
    if len(val_text) < 256:
        return text, None
    return train_text, val_text


def auto_configure_from_text(text, base_config=None):
    config = copy.deepcopy(DEFAULT_CONFIG if base_config is None else base_config)
    char_count = len(text)
    unique_chars = len(set(text))

    train_text, val_text = stratified_train_val_split(text, config["val_split_min_chars"])

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
    tok_counts = Counter(encoded_train)
    used_vocab = len(tok_counts)
    v_size = tok.vocab_size

    if num_tokens < 2:
        raise ValueError("TEXT_DATA is too short—provide at least a few words.")

    tok_ent = -sum((c / num_tokens) * math.log(c / num_tokens) for c in tok_counts.values())
    corpus_tok_ppx = float(math.exp(tok_ent))

    encoded_val = tok.encode(val_text) if val_text is not None else None
    if encoded_val is not None and len(encoded_val) < 16:
        encoded_val = None
        val_text = None

    comp_ratio = train_chars / max(1, num_tokens)

    if config.get("auto_size", True):
        n_steps = num_tokens - 1

        if used_vocab <= 128:
            target_by_vocab = int(round(1.5 * used_vocab))
        else:
            target_by_vocab = int(round(0.55 * used_vocab))
        target_by_tokens = max(config["min_symbols"], int(round(0.90 * num_tokens)))
        raw_symbols = min(target_by_vocab, target_by_tokens)
        snapped_symbols = int(round(raw_symbols / 8.0)) * 8
        n_symbols = max(config["min_symbols"], min(config["max_symbols"], snapped_symbols))

        if n_steps <= 4096:
            epoch_cap = n_steps
        else:
            max_cap = 32768 if HIGH_VRAM else (16384 if DEVICE == "cuda" else 4096)
            epoch_cap = max(4096, min(max_cap, n_steps // 8))

        if n_steps < 32:
            batch_size = 1
            stream_len = max(4, n_steps)
            adj_seq_len = min(8, stream_len)
        elif n_steps < 512:
            batch_size = 16 if DEVICE == "cuda" else 4
            stream_len = min(32, max(12, n_steps // 4))
            adj_seq_len = min(16, max(4, stream_len // 2))
        elif n_steps < 4096:
            batch_size = 32 if DEVICE == "cuda" else 8
            stream_len = max(16, min(n_steps, epoch_cap) // batch_size)
            adj_seq_len = max(8, min(24, max(8, stream_len // 2)))
        elif n_steps < 16384:
            batch_size = 64 if DEVICE == "cuda" else 16
            stream_len = max(32, min(n_steps, epoch_cap) // batch_size)
            adj_seq_len = max(16, min(32, stream_len // 2))
        else:
            batch_size = 256 if HIGH_VRAM else (128 if DEVICE == "cuda" else 16)
            stream_len = max(32, min(n_steps, epoch_cap) // batch_size)
            adj_seq_len = max(16, min(32, stream_len // 4))

        dim_mult = 9.3 if (HIGH_VRAM and n_steps >= 16384) else 8.0
        raw_dim = int(round(dim_mult * math.sqrt(max(16, v_size)) / 16.0)) * 16
        emb_dim = max(32, min(config["max_dim"], raw_dim))

        if n_steps < 80:
            stack_size = 8
        elif n_steps <= 2000:
            stack_size = 16
        else:
            stack_size = 24

        if n_steps < 512:
            rec_depth = 5
        elif n_steps <= 20000:
            rec_depth = 6
        else:
            rec_depth = 4

        adj_lr = float(1.5e-3 * math.sqrt(64.0 / emb_dim))
        adj_temp = float(max(0.05, min(0.18, 0.14 * math.sqrt(64.0 / emb_dim))))
        adj_wd = 1e-5 if n_steps < 4000 else 3e-4
        adj_drop = 0.0 if n_steps < 4000 else 0.10
        adj_grad_clip = float(1.0 * math.sqrt(emb_dim / 64.0))
        adj_ponder = 0.030
        adj_eth_weight = float(0.12 * math.sqrt(math.log(max(2, n_symbols)) / math.log(64.0)))
        adj_diag_tokens = min(n_steps, max(2000, 15 * n_symbols))

        ref_ppx = min(corpus_tok_ppx, float(n_symbols))
        ppx_low = min(0.45 * n_symbols, max(8.0, 0.70 * ref_ppx))
        ppx_high = min(0.75 * n_symbols, max(ppx_low * 1.4, 1.45 * ref_ppx))

        tokens_per_epoch = batch_size * stream_len
        chunks_per_epoch = max(1, math.ceil(stream_len / adj_seq_len))
        if n_steps < 512:
            epochs = max(200, min(320, 1000 // chunks_per_epoch))
        elif n_steps < 4096:
            target_opt_steps = min(16000, int(round(3000 * (max(1, n_steps) / 150.0) ** 0.35)))
            epochs = max(240, min(1000, target_opt_steps // chunks_per_epoch))
        else:
            max_ep_cap = 600 if epoch_cap >= 16384 else 1000
            target_opt_steps = min(40000, int(round(4000 * (max(1, n_steps) / 150.0) ** 0.35)))
            epochs_by_steps = target_opt_steps // chunks_per_epoch
            target_passes = 14.0 if epoch_cap >= 16384 else 4.0
            min_pass_epochs = int(math.ceil(target_passes * n_steps / max(1, tokens_per_epoch))) if n_steps > min(n_steps, epoch_cap) else 150
            epochs = max(min_pass_epochs, min(max_ep_cap, epochs_by_steps))
            epochs = max(150, min(max_ep_cap, epochs))

        warmup_epochs = max(5, int(round(0.08 * epochs)))
        log_every = max(1, epochs // 25 if epochs <= 400 else epochs // 40)
        eval_every = max(1, min(5, max(1, log_every // 3)))

        config["bpe_merges"] = len(tok.merges)
        config["n_symbols"] = n_symbols
        config["epoch_token_cap"] = epoch_cap
        config["batch_size"] = batch_size
        config["stream_len"] = stream_len
        config["seq_len"] = adj_seq_len
        config["embedding_dim"] = emb_dim
        config["stack_size"] = stack_size
        config["max_recursion_depth"] = rec_depth
        config["learning_rate"] = adj_lr
        config["weight_decay"] = adj_wd
        config["decoder_dropout"] = adj_drop
        config["grad_clip"] = adj_grad_clip
        config["ponder_penalty"] = adj_ponder
        config["ethical_weight"] = adj_eth_weight
        config["vq_temp"] = adj_temp
        config["diag_max_tokens"] = adj_diag_tokens
        config["epochs"] = epochs
        config["warmup_epochs"] = warmup_epochs
        config["log_every"] = log_every
        config["eval_every"] = eval_every
        config["sample_every"] = max(1, epochs // 8 if epochs <= 400 else epochs // 10)
        config["revive_every"] = max(1, min(3, epochs // 30))
        config["comp_ratio"] = comp_ratio
        config["corpus_tok_ppx"] = corpus_tok_ppx
        config["target_ppx_low"] = ppx_low
        config["target_ppx_high"] = ppx_high

    train_tensor = torch.tensor(encoded_train, dtype=torch.long)
    val_tensor = torch.tensor(encoded_val, dtype=torch.long) if encoded_val is not None else None

    n_steps = len(encoded_train) - 1
    bs = config["batch_size"]
    s_len = config["stream_len"] if config["stream_len"] is not None else max(1, min(n_steps, config["epoch_token_cap"]) // bs)
    est_passes = (config["epochs"] * bs * s_len) / max(1, n_steps)
    p1_end = max(5, int(config["phase1_frac"] * config["epochs"]))
    p2_end = max(p1_end + 5, int(config["phase2_frac"] * config["epochs"]))

    print("=== Auto-Configuration Summary (SACRSN v55 Unified) ===")
    print(f"Corpus       : {char_count:,} chars ({unique_chars} base chars, min_char_freq>={min_char_freq}) | Train={len(encoded_train):,} toks"
          + (f" | Val={len(encoded_val):,} toks (stratified)" if encoded_val is not None else " | Val=None (small text)"))
    print(
        f"Tokenizer    : Subword BPE | Vocab={v_size} (Used={used_vocab}, ZipfPPX={corpus_tok_ppx:.1f}) | "
        f"Merges={len(tok.merges)}/{target_merges} | Comp={comp_ratio:.2f}x"
    )
    print(
        f"Architecture : Dim={config['embedding_dim']} (2D={config['embedding_dim'] * 2}) | "
        f"Codebook={config['n_symbols']} (~{config['n_symbols'] / max(1, used_vocab):.2f}x used vocab, "
        f"TargetPPX=[{config['target_ppx_low']:.0f}..{config['target_ppx_high']:.0f}]) | Stack={config['stack_size']} | MaxDepth={config['max_recursion_depth']}"
    )
    print(
        f"Schedule     : Streams={bs}x{s_len} (Cap={config['epoch_token_cap']:,}, TBPTT={config['seq_len']}) | "
        f"Epochs={config['epochs']} (Phases: 0..{p1_end}->KMeans->{p2_end}->{config['epochs']}, ~{est_passes:.1f}x corpus passes)"
    )
    print(
        f"InitHyper    : LR={config['learning_rate']:.5f} (Plateau every {config['eval_every']}ep, AdjMult={config['adj_lr_mult']:.1f}x) | "
        f"WD={config['weight_decay']:.1e} | Drop={config['decoder_dropout']:.2f} | ClipMax={config['grad_clip_max']:.1f} | "
        f"EthW={config['ethical_weight']:.3f} | StkBalW={config['stack_balance_weight']:.2f} | "
        f"VQTempCap=[0.05..0.18] | Consc=({config['conscience_p2']:.2f}/{config['conscience_p3']:.2f}) | ACT_Thresh={config['act_threshold']:.2f} (PonderNet)"
    )
    print(f"Sample tokens: {[tok.format_token(tok.id_to_token[i]) for i in encoded_train[:12]]}")
    print("=========================================================")

    return tok, v_size, train_tensor, val_tensor, val_text, config


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
        norm = torch.sqrt(z_r.float() ** 2 + z_i.float() ** 2 + 1e-2).to(z_r.dtype)
        scale = torch.clamp(F.relu(norm + self.bias) / norm, max=4.0)
        return z_r * scale, z_i * scale


class ComplexLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc_real = nn.Linear(dim, dim, bias=False)
        self.fc_imag = nn.Linear(dim, dim, bias=False)
        # 1/sqrt(2) preserves variance across complex multiplication (W_r + i W_i)(z_r + i z_i)
        gain = 1.0 / math.sqrt(2.0)
        nn.init.orthogonal_(self.fc_real.weight, gain=gain)
        nn.init.orthogonal_(self.fc_imag.weight, gain=gain)

    def forward(self, z_r, z_i):
        return (
            self.fc_real(z_r) - self.fc_imag(z_i),
            self.fc_real(z_i) + self.fc_imag(z_r),
        )


class ComplexFFN(nn.Module):
    """Residual 2x complex MLP expansion block for stronger representational capacity."""
    def __init__(self, dim, mult=2):
        super().__init__()
        hidden = dim * mult
        self.norm = ComplexLayerNorm(dim)
        self.fc1_r = nn.Linear(dim, hidden, bias=False)
        self.fc1_i = nn.Linear(dim, hidden, bias=False)
        self.act = ModReLU(hidden)
        self.fc2_r = nn.Linear(hidden, dim, bias=False)
        self.fc2_i = nn.Linear(hidden, dim, bias=False)
        g1 = 1.0 / math.sqrt(2.0)
        g2 = 0.5 / math.sqrt(2.0)
        nn.init.orthogonal_(self.fc1_r.weight, gain=g1)
        nn.init.orthogonal_(self.fc1_i.weight, gain=g1)
        nn.init.orthogonal_(self.fc2_r.weight, gain=g2)
        nn.init.orthogonal_(self.fc2_i.weight, gain=g2)

    def forward(self, z_r, z_i):
        n_r, n_i = self.norm(z_r, z_i)
        h_r = self.fc1_r(n_r) - self.fc1_i(n_i)
        h_i = self.fc1_r(n_i) + self.fc1_i(n_r)
        a_r, a_i = self.act(h_r, h_i)
        o_r = self.fc2_r(a_r) - self.fc2_i(a_i)
        o_i = self.fc2_r(a_i) + self.fc2_i(a_r)
        return z_r + 0.5 * o_r, z_i + 0.5 * o_i


# ==========================================
# 4. Memory Modules (Stack, Holographic HRR, VQ Graph)
# ==========================================
class DifferentiableStack(nn.Module):
    def __init__(self, dim, size):
        super().__init__()
        self.dim = dim
        self.size = size

    def forward(self, z_r, z_i, memory, ptr, control):
        ctrl_f = control.float()
        push = ctrl_f[:, 0:1]
        pop = ctrl_f[:, 1:2]
        noop = ctrl_f[:, 2:3]

        ptr_up = torch.zeros_like(ptr)
        ptr_up[:, 1:] = ptr[:, :-1]
        ptr_up[:, -1] = ptr_up[:, -1] + ptr[:, -1]

        ptr_down = torch.zeros_like(ptr)
        ptr_down[:, :-1] = ptr[:, 1:]
        ptr_down[:, 0] = ptr_down[:, 0] + ptr[:, 0]

        new_ptr = (push * ptr_up) + (pop * ptr_down) + (noop * ptr)
        new_ptr = new_ptr / (new_ptr.sum(dim=1, keepdim=True) + 1e-6)

        z_flat = torch.cat([z_r.float(), z_i.float()], dim=-1)
        write_mask = push * ptr_up
        clear_mask = pop * ptr
        write_val = write_mask.unsqueeze(2) * z_flat.unsqueeze(1)
        # Cleanly erase popped slots (no ghost residue) while gently decaying stale entries
        retain_mask = (
            (1.0 - write_mask.unsqueeze(2)).clamp(min=0.0)
            * (1.0 - clear_mask.unsqueeze(2)).clamp(min=0.0)
            * 0.99
        )
        new_memory = write_val + (memory * retain_mask)

        # Push peeks pre-push top (ptr) instead of echoing self; Pop reads newly exposed top (ptr_down)
        read_ptr = (push * ptr) + (pop * ptr_down) + (noop * ptr)
        read_ptr = read_ptr / (read_ptr.sum(dim=1, keepdim=True) + 1e-6)
        read_flat = torch.sum(new_memory * read_ptr.unsqueeze(2), dim=1)
        read_r = read_flat[:, :self.dim].to(z_r.dtype)
        read_i = read_flat[:, self.dim:].to(z_i.dtype)

        return read_r, read_i, new_memory, new_ptr


class HolographicMemory(nn.Module):
    """Complex Holographic Reduced Representation (HRR) Associative Trace."""
    def __init__(self, dim, decay=0.99, write_scale=0.10):
        super().__init__()
        self.dim = dim
        self.decay = decay
        self.write_scale = write_scale
        self.key_proj = ComplexLinear(dim)
        self.val_proj = ComplexLinear(dim)
        self.query_proj = ComplexLinear(dim)
        self.out_norm = ComplexLayerNorm(dim)

    def retrieve(self, z_r, z_i, holo_r, holo_i):
        q_r, q_i = self.query_proj(z_r, z_i)
        q_mag = torch.sqrt(q_r.float() ** 2 + q_i.float() ** 2 + 1e-6).to(q_r.dtype)
        uq_r = q_r / q_mag
        uq_i = q_i / q_mag

        hr = holo_r.to(z_r.dtype)
        hi = holo_i.to(z_i.dtype)
        ret_r = hr * uq_r + hi * uq_i
        ret_i = hi * uq_r - hr * uq_i
        return self.out_norm(ret_r, ret_i)

    def write(self, z_r, z_i, holo_r, holo_i):
        k_r, k_i = self.key_proj(z_r, z_i)
        v_r, v_i = self.val_proj(z_r, z_i)

        kr_f, ki_f = k_r.float(), k_i.float()
        vr_f, vi_f = v_r.float(), v_i.float()
        k_mag = torch.sqrt(kr_f ** 2 + ki_f ** 2 + 1e-6)
        uk_r, uk_i = kr_f / k_mag, ki_f / k_mag

        bind_r = uk_r * vr_f - uk_i * vi_f
        bind_i = uk_r * vi_f + uk_i * vr_f

        new_hr = self.decay * holo_r + self.write_scale * bind_r
        new_hi = self.decay * holo_i + self.write_scale * bind_i

        h_rms = torch.sqrt(torch.mean(new_hr ** 2 + new_hi ** 2, dim=-1, keepdim=True) + 1e-6)
        cap_scale = torch.clamp(1.5 / h_rms, max=1.0)
        return new_hr * cap_scale, new_hi * cap_scale


class GraphMemoryVQ(nn.Module):
    def __init__(self, latent_dim, n_symbols, commitment_cost=0.25, vq_temp=0.10):
        super().__init__()
        self.n_symbols = n_symbols
        self.commitment_cost = float(commitment_cost)

        self.codebook = nn.Parameter(torch.empty(n_symbols, latent_dim * 2))
        nn.init.normal_(self.codebook, mean=0.0, std=1.0)
        self.normalize_codebook_()

        self.adjacency = nn.Parameter(torch.zeros(n_symbols, n_symbols))
        self.register_buffer("usage_count", torch.zeros(n_symbols))
        self.register_buffer("ema_usage", torch.full((n_symbols,), 1.0 / n_symbols))
        self.register_buffer("ema_d_std", torch.tensor(float(vq_temp) / 0.40))

    @property
    def live_temp(self):
        return float(torch.clamp(self.ema_d_std * 0.40, min=0.05, max=0.18).item())

    @torch.no_grad()
    def normalize_codebook_(self):
        """Projects codebook rows to unit-RMS so cb_f and cb_norm remain identical."""
        cb_f = self.codebook.data.float()
        rms = torch.sqrt(torch.mean(cb_f ** 2, dim=-1, keepdim=True) + 1e-4)
        self.codebook.data.copy_((cb_f / rms).to(self.codebook.dtype))

    def normalized_codebook(self):
        cb_f = self.codebook.float()
        cb_rms = torch.sqrt(torch.mean(cb_f ** 2, dim=-1, keepdim=True) + 1e-4)
        return cb_f / cb_rms

    def forward(
        self,
        z_r,
        z_i,
        prev_symbol_idx=None,
        bias_scale=0.0,
        commit_cost=None,
        active_weights=None,
        cb_norm=None,
        conscience_scale=0.0,
    ):
        if commit_cost is None:
            commit_cost = self.commitment_cost

        dim = z_r.shape[-1]
        z_raw = torch.cat([z_r.float(), z_i.float()], dim=-1)
        cb_f = self.codebook.float()

        z_flat = z_raw / torch.sqrt(torch.mean(z_raw ** 2, dim=-1, keepdim=True) + 1e-4)
        if cb_norm is None:
            cb_norm = self.normalized_codebook()

        d_raw = (
            torch.sum(z_flat ** 2, dim=-1, keepdim=True)
            + torch.sum(cb_norm ** 2, dim=-1)
            - 2.0 * torch.matmul(z_flat, cb_norm.t())
        ) / z_flat.shape[-1]

        if self.training and d_raw.numel() > 1:
            with torch.no_grad():
                batch_d_std = d_raw.detach().std(dim=-1).mean().clamp(min=0.02, max=2.0)
                self.ema_d_std.mul_(0.95).add_(0.05 * batch_d_std)

        temp_tensor = torch.clamp(self.ema_d_std * 0.40, min=0.05, max=0.18)
        soft_probs = F.softmax(-d_raw / temp_tensor, dim=-1)

        d_select = d_raw
        if prev_symbol_idx is not None and bias_scale > 0.0:
            graph_prior = self.adjacency[prev_symbol_idx].float()
            bias = bias_scale * torch.sigmoid(graph_prior)
            d_select = d_select - bias

        if self.training and conscience_scale > 0.0:
            rel_usage = self.ema_usage * float(self.n_symbols)
            hub_penalty = conscience_scale * F.relu(torch.log1p(rel_usage) - math.log(4.0))
            d_select = d_select + hub_penalty.unsqueeze(0)

        min_indices = torch.argmin(d_select, dim=-1)

        if self.training:
            with torch.no_grad():
                if active_weights is None:
                    weights = torch.ones_like(min_indices.view(-1), dtype=self.usage_count.dtype)
                else:
                    weights = active_weights.view(-1).to(self.usage_count.dtype)
                self.usage_count.index_add_(0, min_indices.view(-1), weights)

                w_sum = weights.sum()
                if w_sum > 0:
                    batch_hist = torch.zeros_like(self.ema_usage)
                    batch_hist.index_add_(0, min_indices.view(-1), weights / w_sum)
                    self.ema_usage.mul_(0.995).add_(0.005 * batch_hist)

        z_q = F.embedding(min_indices, cb_f)

        loss_vq_vec = torch.mean((z_q - z_flat.detach()) ** 2, dim=-1)
        loss_commit_vec = torch.mean((z_q.detach() - z_flat) ** 2, dim=-1)
        vq_loss_vec = loss_vq_vec + commit_cost * loss_commit_vec
        vq_cos_vec = F.cosine_similarity(z_flat.detach(), z_q.detach(), dim=-1)

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
            vq_cos_vec.detach(),
        )


# ==========================================
# 5. Core Processor (Attention, GWT Router, Ethics, Cell)
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
        q_r, q_i = self.q_proj(z_r, z_i)

        if stack_mem is not None:
            mem_r = stack_mem[..., :self.dim].to(z_r.dtype)
            mem_i = stack_mem[..., self.dim:].to(z_i.dtype)
            all_r = torch.cat([z_r.unsqueeze(1), mem_r], dim=1)
            all_i = torch.cat([z_i.unsqueeze(1), mem_i], dim=1)
            mem_valid = stack_mem.detach().abs().amax(dim=-1) > 1e-4
            self_valid = torch.ones(B, 1, device=z_r.device, dtype=torch.bool)
            attn_mask = torch.cat([self_valid, mem_valid], dim=1)
        else:
            all_r = z_r.unsqueeze(1)
            all_i = z_i.unsqueeze(1)
            attn_mask = None

        M = all_r.size(1)
        k_r, k_i = self.k_proj(all_r, all_i)
        v_r, v_i = self.v_proj(all_r, all_i)

        H, hd = self.n_heads, self.head_dim
        q_h = torch.cat([q_r.view(B, H, 1, hd), q_i.view(B, H, 1, hd)], dim=-1)
        k_h = torch.cat(
            [k_r.view(B, M, H, hd).transpose(1, 2), k_i.view(B, M, H, hd).transpose(1, 2)],
            dim=-1,
        )

        attn_scores = torch.matmul(q_h, k_h.transpose(-2, -1)).float() * self.scale
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(~attn_mask.view(B, 1, 1, M), -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1).to(v_r.dtype)

        v_r_h = v_r.view(B, M, H, hd).transpose(1, 2)
        v_i_h = v_i.view(B, M, H, hd).transpose(1, 2)

        out_r = torch.matmul(attn_weights, v_r_h).squeeze(2).reshape(B, self.dim)
        out_i = torch.matmul(attn_weights, v_i_h).squeeze(2).reshape(B, self.dim)

        return z_r + 0.25 * out_r, z_i + 0.25 * out_i


class GlobalWorkspaceRouter(nn.Module):
    """Competitive broadcast router across Perception, Stack, Holographic, and Graph streams."""
    def __init__(self, dim):
        super().__init__()
        self.scorer = nn.Linear(dim * 2, 1)
        self.stream_bias = nn.Parameter(torch.zeros(4))

    def forward(self, p_r, p_i, stk_r, stk_i, holo_r, holo_i, g_r, g_i, active_mask=None):
        streams_r = torch.stack([p_r, stk_r, holo_r, g_r], dim=1)
        streams_i = torch.stack([p_i, stk_i, holo_i, g_i], dim=1)
        streams_flat = torch.cat([streams_r, streams_i], dim=-1)

        logits = self.scorer(streams_flat).squeeze(-1).float() + self.stream_bias
        if active_mask is not None:
            if active_mask.dim() == 1:
                logits = logits.masked_fill(~active_mask.view(1, 4), -1e9)
            else:
                logits = logits.masked_fill(~active_mask, -1e9)

        gates = F.softmax(logits, dim=-1)
        g_cast = gates.to(p_r.dtype)

        p_weight = 0.5 + 0.5 * g_cast[:, 0:1]
        ws_r = p_weight * p_r + g_cast[:, 1:2] * stk_r + g_cast[:, 2:3] * holo_r + g_cast[:, 3:4] * g_r
        ws_i = p_weight * p_i + g_cast[:, 1:2] * stk_i + g_cast[:, 2:3] * holo_i + g_cast[:, 3:4] * g_i
        return ws_r, ws_i, gates


class EthicalConstraint(nn.Module):
    def __init__(self, n_symbols):
        super().__init__()
        self.n_symbols = int(n_symbols)

    def forward(self, prev_sym, curr_sym, adjacency, curr_conf=None):
        if prev_sym is None:
            return torch.zeros((), device=adjacency.device, dtype=torch.float32)
        row_logits = adjacency[prev_sym].float().view(-1, self.n_symbols)
        smoothing = 0.05 if self.training else 0.0
        ce = F.cross_entropy(row_logits, curr_sym.view(-1), reduction="none", label_smoothing=smoothing)
        if curr_conf is not None:
            w = curr_conf.detach().view(-1).clamp(min=0.1, max=1.0)
            return (ce * w).sum() / w.sum().clamp(min=1.0)
        return ce.mean()


class AdaptiveRecursiveCell(nn.Module):
    def __init__(self, dim, stack_temp=0.75, ponder_prior_halt=0.45):
        super().__init__()
        self.stack_temp = float(stack_temp)
        self.linear = ComplexLinear(dim)
        self.norm = ComplexLayerNorm(dim)
        self.act = ModReLU(dim)
        self.attention = ComplexAttention(dim)
        self.ffn = ComplexFFN(dim, mult=2)
        self.halt_linear = nn.Linear(dim * 2, 1)
        self.stack_ctrl = nn.Linear(dim * 2, 3)

        p0 = min(0.60, max(0.20, float(ponder_prior_halt)))
        halt_bias = float(math.log(p0 / (1.0 - p0)))
        nn.init.constant_(self.halt_linear.bias, halt_bias)

    def forward(self, z_r, z_i, stack_mem=None):
        l_r, l_i = self.linear(z_r, z_i)
        n_r, n_i = self.norm(l_r, l_i)
        a_r, a_i = self.act(n_r, n_i)
        att_r, att_i = self.attention(a_r, a_i, stack_mem)
        p_r, p_i = self.ffn(att_r, att_i)
        z_flat = torch.cat([p_r, p_i], dim=-1)
        halt_prob = torch.sigmoid(self.halt_linear(z_flat).float())
        stack_probs = F.softmax(self.stack_ctrl(z_flat).float() / self.stack_temp, dim=-1)
        return p_r, p_i, halt_prob, stack_probs


# ==========================================
# 6. Master Model (Decoupled Hyperparameters + Unified PonderNet)
# ==========================================
class UberCRSN(nn.Module):
    def __init__(self, vocab_size, dim, config=None):
        super().__init__()
        cfg = copy.deepcopy(DEFAULT_CONFIG if config is None else config)
        self.cfg = cfg
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.n_symbols = int(cfg["n_symbols"])
        self.stack_size = int(cfg["stack_size"])
        self.max_recursion_depth = int(cfg["max_recursion_depth"])
        self.use_stack = bool(cfg["use_stack"])
        self.act_threshold = float(cfg["act_threshold"])
        self.ponder_prior_halt = float(cfg["ponder_prior_halt"])
        self.graph_bias_scale = float(cfg["graph_bias_scale"])
        self.stack_temp = float(cfg["stack_temp"])

        self.emb_mag = nn.Embedding(vocab_size, dim)
        self.emb_phase = nn.Parameter(torch.randn(vocab_size, dim))
        self.carry_gate = nn.Linear(dim * 4, dim)
        nn.init.constant_(self.carry_gate.bias, 0.0)

        self.cell = AdaptiveRecursiveCell(
            dim,
            stack_temp=self.stack_temp,
            ponder_prior_halt=self.ponder_prior_halt,
        )
        self.vq_layer = GraphMemoryVQ(
            dim,
            self.n_symbols,
            commitment_cost=float(cfg["commitment_cost"]),
            vq_temp=float(cfg["vq_temp"]),
        )
        self.holo_mem = HolographicMemory(
            dim,
            decay=float(cfg["holo_decay"]),
            write_scale=float(cfg["holo_write_scale"]),
        )
        self.gwt_router = GlobalWorkspaceRouter(dim)
        self.dec_norm = nn.LayerNorm(dim * 2)
        self.dec_drop = nn.Dropout(float(cfg.get("decoder_dropout", 0.10)))
        self.decoder = nn.Linear(dim * 2, vocab_size)

        if self.use_stack:
            self.stack = DifferentiableStack(dim, self.stack_size)

        self.ethics = EthicalConstraint(self.n_symbols)
        self.register_buffer("prev_sym_soft", torch.full((self.n_symbols,), 1.0 / self.n_symbols))
        self.register_buffer("sym_alpha", torch.tensor(0.0))
        self.register_buffer("graph_gate", torch.tensor(0.0))

    def embed(self, idx):
        r = self.emb_mag(idx)
        t = self.emb_phase[idx]
        return r * torch.cos(t), r * torch.sin(t)

    def forward(
        self,
        input_ids,
        hidden=None,
        prev_sym=None,
        bias_scale=None,
        commit_cost=None,
        return_stats=False,
        use_graph_echo=True,
        cached_P_adj=None,
        cached_cb_norm=None,
        conscience_scale=0.05,
        ablate_stack=False,
        ablate_stack_read=False,
        ablate_stack_attn=False,
        ablate_holo=False,
        ablate_graph=False,
        ablate_vq=False,
        ablate_carry=False,
        ablate_act_1step=False,
    ):
        if bias_scale is None:
            bias_scale = self.graph_bias_scale

        use_echo_eff = use_graph_echo and (not ablate_graph)
        bias_eff = 0.0 if ablate_graph else bias_scale
        use_stack_engine = self.use_stack and (not ablate_stack)

        batch_size = input_ids.size(0)
        z_r, z_i = self.embed(input_ids)
        z_r = z_r.squeeze(1)
        z_i = z_i.squeeze(1)

        if hidden is None:
            stack_mem = torch.zeros(batch_size, self.stack_size, self.dim * 2, device=z_r.device, dtype=torch.float32)
            stack_ptr = torch.zeros(batch_size, self.stack_size, device=z_r.device, dtype=torch.float32)
            stack_ptr[:, 0] = 1.0
            holo_r = torch.zeros(batch_size, self.dim, device=z_r.device, dtype=torch.float32)
            holo_i = torch.zeros(batch_size, self.dim, device=z_r.device, dtype=torch.float32)
            carry_mean_val = torch.zeros((), device=z_r.device, dtype=torch.float32)
            carry_std_val = torch.zeros((), device=z_r.device, dtype=torch.float32)
        else:
            prev_r, prev_i, stack_mem, stack_ptr, holo_r, holo_i = hidden
            if ablate_carry:
                carry_mean_val = torch.zeros((), device=z_r.device, dtype=torch.float32)
                carry_std_val = torch.zeros((), device=z_r.device, dtype=torch.float32)
            else:
                pr = prev_r.to(z_r.dtype)
                pi = prev_i.to(z_i.dtype)
                gate_in = torch.cat([z_r, z_i, pr, pi], dim=-1)
                carry_g = torch.sigmoid(self.carry_gate(gate_in))
                z_r = (1.0 - carry_g) * z_r + carry_g * pr
                z_i = (1.0 - carry_g) * z_i + carry_g * pi
                carry_f = carry_g.detach().float()
                carry_mean_val = carry_f.mean()
                carry_std_val = carry_f.std(unbiased=False) if carry_f.numel() > 1 else torch.zeros((), device=z_r.device, dtype=torch.float32)

        cb_norm = cached_cb_norm if cached_cb_norm is not None else self.vq_layer.normalized_codebook()
        if use_echo_eff:
            P_adj = cached_P_adj if cached_P_adj is not None else F.softmax(self.vq_layer.adjacency.float(), dim=-1)
        else:
            P_adj = None

        g_gate = self.graph_gate.to(z_r.dtype)
        alpha = torch.zeros((), device=z_r.device, dtype=z_r.dtype) if ablate_vq else self.sym_alpha.to(z_r.dtype)
        log_K = float(math.log(max(2, self.n_symbols)))

        if use_echo_eff and prev_sym is not None:
            sym_belief = P_adj[prev_sym]
        else:
            sym_belief = None

        # Mask ablated streams out of GlobalWorkspaceRouter softmax cleanly
        gwt_active_mask = torch.tensor(
            [
                True,
                bool(use_stack_engine and (not ablate_stack_read)),
                bool(not ablate_holo),
                bool(not ablate_graph),
            ],
            device=z_r.device,
            dtype=torch.bool,
        )

        # Unified PonderNet halting state (identical across train, val, and batch_size=1 eval)
        unhalted = torch.ones(batch_size, 1, device=z_r.device, dtype=torch.float32)
        cum_halt = torch.zeros(batch_size, 1, device=z_r.device, dtype=torch.float32)
        selected_done = torch.zeros(batch_size, dtype=torch.bool, device=z_r.device)
        steps_taken = torch.zeros(batch_size, 1, device=z_r.device, dtype=torch.float32)
        step_p_list = []
        step_logits_list = []

        stack_depth_sum = torch.zeros((), device=z_r.device, dtype=torch.float32)
        stack_ctrl_sum = torch.zeros(3, device=z_r.device, dtype=torch.float32)
        gwt_gates_sum = torch.zeros(4, device=z_r.device, dtype=torch.float32)
        sig_rms_sum = torch.zeros(4, device=z_r.device, dtype=torch.float32)

        step_soft_probs = []
        step_active_weights = []
        t0_z_flats = None

        z_w_r = torch.zeros(batch_size, self.dim, device=z_r.device, dtype=z_r.dtype)
        z_w_i = torch.zeros(batch_size, self.dim, device=z_i.device, dtype=z_i.dtype)
        cont_w_r = torch.zeros(batch_size, self.dim, device=z_r.device, dtype=z_r.dtype)
        cont_w_i = torch.zeros(batch_size, self.dim, device=z_i.device, dtype=z_i.dtype)

        current_sym = None
        prev_step_soft = None
        curr_step_conf = None
        vq_loss_sum = torch.zeros((), device=z_r.device, dtype=torch.float32)
        vq_cos_sum = torch.zeros((), device=z_r.device, dtype=torch.float32)
        cons_loss_sum = torch.zeros((), device=z_r.device, dtype=torch.float32)
        cons_weight_total = torch.zeros((), device=z_r.device, dtype=torch.float32)
        p_weight_total = torch.zeros((), device=z_r.device, dtype=torch.float32)

        if use_stack_engine:
            slot_indices = torch.arange(self.stack_size, device=z_r.device, dtype=torch.float32)

        M = self.max_recursion_depth
        act_thresh = self.act_threshold

        for t in range(M):
            mem_arg = stack_mem if (use_stack_engine and not ablate_stack_attn) else None
            p_r, p_i, lambda_t, stack_ctrl = self.cell(z_r, z_i, mem_arg)

            if t == M - 1 or ablate_act_1step:
                p_t = unhalted
                unhalted_next = torch.zeros_like(unhalted)
            else:
                p_t = unhalted * lambda_t
                unhalted_next = unhalted * (1.0 - lambda_t)

            step_p_list.append(p_t)
            p_w = p_t.squeeze(-1)
            p_w_det = p_w.detach()
            unhalted_det = unhalted.squeeze(-1).detach()

            still_discrete = (~selected_done).float().unsqueeze(-1)
            steps_taken = steps_taken + still_discrete

            stack_ctrl_sum = stack_ctrl_sum + (stack_ctrl * p_w_det.unsqueeze(-1)).sum(dim=0)

            if use_stack_engine:
                read_r, read_i, new_mem, new_ptr = self.stack(p_r, p_i, stack_mem, stack_ptr, stack_ctrl)
                run_w = unhalted.detach()
                run_3d = run_w.unsqueeze(-1)
                stack_mem = new_mem * run_3d + stack_mem * (1.0 - run_3d)
                stack_ptr = new_ptr * run_w + stack_ptr * (1.0 - run_w)
                depth_vec = torch.sum(stack_ptr * slot_indices, dim=1)
                stack_depth_sum = stack_depth_sum + (depth_vec * p_w_det).sum()
                if ablate_stack_read:
                    read_r = torch.zeros_like(p_r)
                    read_i = torch.zeros_like(p_i)
            else:
                read_r = torch.zeros_like(p_r)
                read_i = torch.zeros_like(p_i)

            if ablate_holo:
                holo_read_r = torch.zeros_like(p_r)
                holo_read_i = torch.zeros_like(p_i)
            else:
                holo_read_r, holo_read_i = self.holo_mem.retrieve(p_r, p_i, holo_r, holo_i)

            if use_echo_eff and sym_belief is not None:
                g_echo = torch.matmul(sym_belief, cb_norm).to(p_r.dtype) * g_gate
                g_echo_r = g_echo[:, :self.dim]
                g_echo_i = g_echo[:, self.dim:]
            else:
                g_echo_r = torch.zeros_like(p_r)
                g_echo_i = torch.zeros_like(p_i)

            comb_r, comb_i, gwt_gates = self.gwt_router(
                p_r, p_i, read_r, read_i, holo_read_r, holo_read_i, g_echo_r, g_echo_i,
                active_mask=gwt_active_mask,
            )
            gwt_gates_sum = gwt_gates_sum + (gwt_gates.detach() * p_w_det.unsqueeze(-1)).sum(dim=0)

            if return_stats:
                g_det = gwt_gates.detach().to(p_r.dtype)
                p_eff = 0.5 + 0.5 * g_det[:, 0:1]
                s_rms = torch.stack([
                    torch.sqrt(torch.mean((p_eff * p_r).detach().float() ** 2 + (p_eff * p_i).detach().float() ** 2, dim=-1) + 1e-8).mean(),
                    torch.sqrt(torch.mean((g_det[:, 1:2] * read_r).detach().float() ** 2 + (g_det[:, 1:2] * read_i).detach().float() ** 2, dim=-1) + 1e-8).mean(),
                    torch.sqrt(torch.mean((g_det[:, 2:3] * holo_read_r).detach().float() ** 2 + (g_det[:, 2:3] * holo_read_i).detach().float() ** 2, dim=-1) + 1e-8).mean(),
                    torch.sqrt(torch.mean((g_det[:, 3:4] * g_echo_r).detach().float() ** 2 + (g_det[:, 3:4] * g_echo_i).detach().float() ** 2, dim=-1) + 1e-8).mean(),
                ])
                sig_rms_sum = sig_rms_sum + s_rms * p_w_det.sum()

            prior_sym = prev_sym if t == 0 else None
            step_bias = bias_eff if t == 0 else 0.0

            sym_r, sym_i, norm_r, norm_i, vq_loss_vec, sym_idx, soft_probs, z_flat_det, vq_cos_vec = self.vq_layer(
                comb_r,
                comb_i,
                prior_sym,
                bias_scale=step_bias,
                commit_cost=commit_cost,
                active_weights=p_w_det,
                cb_norm=cb_norm,
                conscience_scale=conscience_scale,
            )
            if t == 0:
                t0_z_flats = z_flat_det

            step_ent = -(soft_probs * torch.log(soft_probs + 1e-8)).sum(dim=-1)
            step_conf = (1.0 - step_ent / log_K).clamp(min=0.1, max=1.0)

            if prev_step_soft is not None:
                kl_step = F.kl_div(
                    torch.log(soft_probs + 1e-8),
                    prev_step_soft.detach(),
                    reduction="none",
                ).sum(dim=-1).clamp(min=0.0)
                cons_loss_sum = cons_loss_sum + (kl_step * unhalted_det).sum()
                cons_weight_total = cons_weight_total + unhalted_det.sum()
            prev_step_soft = soft_probs

            if use_echo_eff:
                if sym_belief is None:
                    sym_belief = torch.matmul(soft_probs, P_adj)
                else:
                    sym_belief = 0.5 * soft_probs + 0.5 * torch.matmul(sym_belief, P_adj)

            new_cum_halt = cum_halt + p_t
            is_halt_step = (~selected_done) & ((new_cum_halt.squeeze(-1) >= act_thresh) | (t == M - 1) | ablate_act_1step)
            if current_sym is None:
                current_sym = sym_idx
            else:
                current_sym = torch.where(is_halt_step, sym_idx, current_sym)

            if curr_step_conf is None:
                curr_step_conf = step_conf
            else:
                curr_step_conf = torch.where(is_halt_step, step_conf, curr_step_conf)

            selected_done = selected_done | is_halt_step

            step_soft_probs.append(soft_probs)
            step_active_weights.append(p_w_det)

            z_r = (1.0 - alpha) * norm_r + alpha * sym_r
            z_i = (1.0 - alpha) * norm_i + alpha * sym_i

            # Per-step decoding enforces standalone validity at every thinking step
            step_feat = self.dec_drop(self.dec_norm(torch.cat([z_r, z_i], dim=-1)))
            step_logits_list.append(self.decoder(step_feat))

            p_cast = p_t.to(z_r.dtype)
            z_w_r = z_w_r + (p_cast * z_r)
            z_w_i = z_w_i + (p_cast * z_i)
            cont_w_r = cont_w_r + (p_cast * norm_r)
            cont_w_i = cont_w_i + (p_cast * norm_i)

            vq_loss_sum = vq_loss_sum + (vq_loss_vec * p_w_det).sum()
            vq_cos_sum = vq_cos_sum + (vq_cos_vec * p_w_det).sum()
            p_weight_total = p_weight_total + p_w_det.sum()

            cum_halt = new_cum_halt
            unhalted = unhalted_next

            if ablate_act_1step:
                break

        denom = p_weight_total.clamp(min=1.0)
        vq_loss_total = vq_loss_sum / denom
        vq_cos_mean = vq_cos_sum / denom
        cons_loss_total = cons_loss_sum / cons_weight_total.clamp(min=1.0)
        avg_stack = stack_depth_sum / denom
        avg_stack_ctrl = stack_ctrl_sum / denom
        avg_gwt_gates = gwt_gates_sum / denom

        ethical_loss = self.ethics(prev_sym, current_sym, self.vq_layer.adjacency, curr_conf=curr_step_conf)

        # PonderNet KL(p || truncated geometric prior) + gentle expected-step bias
        p_stack = torch.cat(step_p_list, dim=1)
        n_run = p_stack.size(1)
        lam_p = self.ponder_prior_halt
        prior_probs = [lam_p * ((1.0 - lam_p) ** i) for i in range(n_run - 1)]
        prior_probs.append((1.0 - lam_p) ** (n_run - 1))
        p_prior = torch.tensor(prior_probs, device=z_r.device, dtype=torch.float32).clamp(min=1e-6)
        p_prior = p_prior / p_prior.sum()
        kl_ponder = (p_stack * (torch.log(p_stack.clamp(min=1e-6)) - torch.log(p_prior.view(1, -1)))).sum(dim=-1)
        step_indices = torch.arange(1, n_run + 1, device=z_r.device, dtype=torch.float32)
        exp_steps = (p_stack * step_indices.view(1, -1)).sum(dim=-1)
        ponder_loss = (kl_ponder + 0.10 * (exp_steps - 1.0)).mean()
        act_steps_mean = steps_taken.mean().detach()

        # Mixture predictive log-probabilities: log(sum_t p_t * softmax(step_logits_t))
        step_logits_stack = torch.stack(step_logits_list, dim=1)
        step_probs_stack = F.softmax(step_logits_stack.float(), dim=-1)
        mix_probs = (p_stack.unsqueeze(-1) * step_probs_stack).sum(dim=1)
        mix_probs = mix_probs / mix_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        logits = torch.log(mix_probs.clamp(min=1e-8))

        next_r = 0.5 * cont_w_r + 0.5 * z_w_r
        next_i = 0.5 * cont_w_i + 0.5 * z_w_i
        if ablate_holo:
            next_holo_r, next_holo_i = holo_r, holo_i
        else:
            next_holo_r, next_holo_i = self.holo_mem.write(next_r, next_i, holo_r, holo_i)
        next_hidden = (next_r, next_i, stack_mem, stack_ptr, next_holo_r, next_holo_i)

        if return_stats:
            ptr_entropy = -(stack_ptr * torch.log(stack_ptr + 1e-8)).sum(dim=-1).mean()
            mem_occ_pct = (stack_mem.detach().abs().amax(dim=-1) > 1e-4).float().mean() * 100.0
            extra_stats = {
                "sig_rms": (sig_rms_sum / denom).detach(),
                "carry_mean": carry_mean_val.detach(),
                "carry_std": carry_std_val.detach(),
                "step_logits": step_logits_stack,
                "step_p": p_stack,
            }
            return (
                logits,
                next_hidden,
                current_sym,
                ponder_loss,
                vq_loss_total,
                ethical_loss,
                avg_stack,
                torch.cat(step_soft_probs, dim=0),
                torch.cat(step_active_weights, dim=0),
                t0_z_flats,
                avg_stack_ctrl,
                steps_taken.squeeze(-1).detach(),
                ptr_entropy.detach(),
                mem_occ_pct.detach(),
                vq_cos_mean.detach(),
                cons_loss_total,
                avg_gwt_gates.detach(),
                extra_stats,
            )

        return logits, next_hidden, current_sym, act_steps_mean, vq_loss_total, ethical_loss, avg_stack


# ==========================================
# 7. Training, K-Means Crystallization, Checkpoint & 3-Phase Engine
# ==========================================
def save_checkpoint(model, tokenizer, config, path):
    if path is None:
        return
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': copy.deepcopy(config),
        'vocab': tokenizer.vocab,
        'merges': tokenizer.merges,
    }, path)


def load_checkpoint(path, device=DEVICE):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = copy.deepcopy(ckpt["config"])
    tok = SimpleBPETokenizer.from_saved(ckpt["vocab"], ckpt["merges"])
    model = UberCRSN(tok.vocab_size, cfg["embedding_dim"], config=cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, tok, cfg


def reset_param_adam_state(opt, param, row_mask=None):
    """Clears stale AdamW moments in-place after codebook K-means or dead-code revival."""
    state = opt.state.get(param)
    if not state:
        return
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        buf = state.get(key)
        if isinstance(buf, torch.Tensor):
            if row_mask is None:
                buf.zero_()
            else:
                buf[row_mask] = 0.0


@torch.no_grad()
def crystallize_codebook_kmeans(model, z_pool_list, n_symbols, iters=15):
    if not z_pool_list:
        return
    pool = torch.cat(z_pool_list, dim=0).float()
    if pool.size(0) < 2:
        return

    if pool.size(0) >= n_symbols:
        perm = torch.randperm(pool.size(0), device=pool.device)[:n_symbols]
        centroids = pool[perm].clone()
        for _ in range(iters):
            dists = (
                (pool ** 2).sum(dim=-1, keepdim=True)
                + (centroids ** 2).sum(dim=-1).unsqueeze(0)
                - 2.0 * torch.matmul(pool, centroids.t())
            )
            assign = torch.argmin(dists, dim=-1)
            new_c = torch.zeros_like(centroids)
            counts = torch.zeros(n_symbols, 1, device=pool.device, dtype=pool.dtype)
            new_c.index_add_(0, assign, pool)
            counts.index_add_(0, assign, torch.ones(pool.size(0), 1, device=pool.device, dtype=pool.dtype))
            empty = counts.squeeze(-1) == 0
            valid = ~empty
            if valid.any():
                centroids[valid] = new_c[valid] / counts[valid]
            if empty.any():
                n_emp = int(empty.sum().item())
                r_idx = torch.randint(0, pool.size(0), (n_emp,), device=pool.device)
                centroids[empty] = pool[r_idx] + 0.03 * torch.randn(n_emp, pool.size(1), device=pool.device)
            rms = torch.sqrt(torch.mean(centroids ** 2, dim=-1, keepdim=True) + 1e-4)
            centroids = centroids / rms
    else:
        r_idx = torch.randint(0, pool.size(0), (n_symbols,), device=pool.device)
        centroids = pool[r_idx] + 0.05 * torch.randn(n_symbols, pool.size(1), device=pool.device)
        rms = torch.sqrt(torch.mean(centroids ** 2, dim=-1, keepdim=True) + 1e-4)
        centroids = centroids / rms

    model.vq_layer.codebook.data.copy_(centroids.to(model.vq_layer.codebook.dtype))
    model.vq_layer.normalize_codebook_()
    model.vq_layer.usage_count.zero_()
    model.vq_layer.ema_usage.fill_(1.0 / n_symbols)


def build_epoch_streams(cpu_tensor, batch_size, epoch_cap, epoch, stream_len_override=None, device=DEVICE):
    n_steps = len(cpu_tensor) - 1
    if stream_len_override is not None:
        stream_len = max(2, min(n_steps, stream_len_override))
    else:
        epoch_window = min(n_steps, epoch_cap)
        stream_len = max(1, epoch_window // batch_size)
        if batch_size * stream_len > n_steps:
            stream_len = max(1, n_steps // batch_size)

    max_start = max(0, n_steps - stream_len)
    if batch_size > 1 and max_start > 0:
        stride = max_start / float(batch_size)
        offsets = []
        for b in range(batch_size):
            lo = int(round(b * stride))
            hi = min(max_start, max(lo, int(round((b + 1) * stride))))
            offsets.append(random.randint(lo, hi))
        random.shuffle(offsets)
    else:
        base = random.randint(0, max_start) if max_start > 0 else 0
        offsets = [base for _ in range(batch_size)]

    streams = torch.stack([cpu_tensor[off : off + stream_len + 1] for off in offsets], dim=0)
    return streams.to(device, non_blocking=True)


def build_val_batches(val_cpu_tensor, device=DEVICE):
    if val_cpu_tensor is None or len(val_cpu_tensor) < 4:
        return []
    n_val = len(val_cpu_tensor) - 1
    s_len = min(64, n_val)
    n_chunks = max(1, n_val // s_len)
    chunk_batch = min(256, n_chunks)
    batches = []
    for start_c in range(0, n_chunks, chunk_batch):
        end_c = min(n_chunks, start_c + chunk_batch)
        streams = torch.stack(
            [val_cpu_tensor[c * s_len : c * s_len + s_len + 1] for c in range(start_c, end_c)],
            dim=0,
        ).to(device)
        batches.append(streams)
    return batches


def _use_amp_for_model(model, batch_size_hint=64):
    dev = next(model.parameters()).device
    return USE_BF16 and (dev.type == "cuda") and (model.dim >= 128 or batch_size_hint >= 64)


@torch.no_grad()
def evaluate_val(model, val_batches, use_amp=None, **fwd_kwargs):
    if not val_batches:
        return None, None
    model.eval()
    dev = next(model.parameters()).device
    if use_amp is None:
        use_amp = _use_amp_for_model(model, val_batches[0].size(0))

    total_loss_t = torch.zeros((), device=dev)
    total_correct_t = torch.zeros((), dtype=torch.long, device=dev)
    total_tokens = 0

    cached_cb = model.vq_layer.normalized_codebook()
    cached_padj = F.softmax(model.vq_layer.adjacency.float(), dim=-1)

    with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=use_amp):
        for streams in val_batches:
            b_curr = streams.size(0)
            s_len = streams.size(1) - 1
            hidden, prev_sym = None, None
            for t in range(s_len):
                x = streams[:, t].unsqueeze(1)
                y = streams[:, t + 1]
                logits, hidden, prev_sym, _, _, _, _ = model(
                    x, hidden, prev_sym, cached_P_adj=cached_padj, cached_cb_norm=cached_cb, **fwd_kwargs
                )
                total_loss_t += F.cross_entropy(logits.float(), y, reduction="sum")
                total_correct_t += (logits.argmax(dim=-1) == y).sum()
                total_tokens += b_curr

    model.train()
    return total_loss_t.item() / max(1, total_tokens), 100.0 * total_correct_t.item() / max(1, total_tokens)


def sample_next_token(logits, recent_ids=None, temperature=0.75, top_k=40, top_p=0.90, rep_penalty=1.15):
    """Top-K / Nucleus (Top-P) sampler with light repetition penalty."""
    scores = logits.squeeze(0).float().clone()
    if scores.dim() > 1:
        scores = scores[0]
    if recent_ids and rep_penalty > 1.0:
        for tid in set(recent_ids[-24:]):
            if scores[tid] > 0:
                scores[tid] /= rep_penalty
            else:
                scores[tid] *= rep_penalty

    if temperature <= 0.0:
        return int(torch.argmax(scores).item())

    scores = scores / temperature
    v_sz = scores.size(-1)
    if top_k is not None and 0 < top_k < v_sz:
        kth_vals = torch.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(scores < kth_vals, -float("inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, -float("inf"))

    probs = F.softmax(scores, dim=-1)
    if torch.isnan(probs).any() or probs.sum().item() <= 0.0:
        return int(torch.argmax(logits.squeeze(0)).item())
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate_sample_preview(model, tokenizer, data_tensor, config, use_amp=None):
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    if use_amp is None:
        use_amp = _use_amp_for_model(model, config["batch_size"])

    with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=use_amp):
        seed_id = data_tensor[0].item()
        gx = torch.tensor([[seed_id]], device=dev)
        gh, gp = None, None
        sample_ids = [seed_id]
        preview_toks = max(30, min(60, int(round(80 / max(1.0, config["comp_ratio"])))))
        for _ in range(preview_toks):
            g_logits, gh, gp, _, _, _, _ = model(gx, gh, gp)
            next_id = sample_next_token(
                g_logits, sample_ids, temperature=0.75, top_k=40, top_p=0.90, rep_penalty=1.15
            )
            gx = torch.tensor([[next_id]], device=dev)
            sample_ids.append(next_id)
    if was_training:
        model.train()
    return tokenizer.decode(sample_ids)


def train(vocab_size, tokenizer, data_tensor, val_tensor, config, save_path="crsn_omni_model.pth", use_amp=None):
    cfg = copy.deepcopy(config)
    model = UberCRSN(vocab_size, cfg["embedding_dim"], config=cfg).to(DEVICE)
    if use_amp is None:
        use_amp = _use_amp_for_model(model, cfg["batch_size"])

    val_batches = build_val_batches(val_tensor, device=DEVICE)

    sym_scale = math.sqrt(max(1.0, cfg["n_symbols"] / 64.0))
    cb_lr = cfg["learning_rate"] * min(1.5, sym_scale)
    adj_lr = cfg["learning_rate"] * float(cfg.get("adj_lr_mult", 8.0)) * min(2.0, sym_scale)

    codebook_params = [model.vq_layer.codebook]
    adj_params = [model.vq_layer.adjacency]
    other_params = [
        p for n, p in model.named_parameters()
        if n not in ("vq_layer.codebook", "vq_layer.adjacency")
    ]
    param_groups = [
        {"params": other_params, "weight_decay": cfg["weight_decay"], "lr": cfg["learning_rate"]},
        {"params": codebook_params, "weight_decay": 0.0, "lr": cb_lr},
        {"params": adj_params, "weight_decay": 0.0, "lr": adj_lr},
    ]
    try:
        opt = torch.optim.AdamW(param_groups, fused=(DEVICE == "cuda"))
        fused_used = (DEVICE == "cuda")
    except Exception:
        opt = torch.optim.AdamW(param_groups)
        fused_used = False

    mod_param_groups = {"Cell": [], "Att": [], "FFN": [], "VQ": [], "Adj": [], "Holo": [], "Dec": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "cell.attention" in name:
            mod_param_groups["Att"].append(p)
        elif "cell.ffn" in name:
            mod_param_groups["FFN"].append(p)
        elif name.startswith("cell."):
            mod_param_groups["Cell"].append(p)
        elif "codebook" in name:
            mod_param_groups["VQ"].append(p)
        elif "adjacency" in name:
            mod_param_groups["Adj"].append(p)
        elif "holo_mem" in name:
            mod_param_groups["Holo"].append(p)
        else:
            mod_param_groups["Dec"].append(p)

    mod_gnorm_sums = {k: torch.zeros((), device=DEVICE) for k in mod_param_groups}
    mod_gnorm_count = 0

    base_lrs = [pg["lr"] for pg in opt.param_groups]
    min_lrs = [max(2e-5, lr * 0.03) for lr in base_lrs]
    lr_warmup_epochs = max(3, min(15, cfg["epochs"] // 30))
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=cfg["lr_patience"],
        min_lr=min_lrs,
    )

    log_K = float(math.log(cfg["n_symbols"]))
    ppx_low = cfg["target_ppx_low"]
    ppx_high = cfg["target_ppx_high"]

    phase1_end = max(5, int(cfg["phase1_frac"] * cfg["epochs"]))
    phase2_end = max(phase1_end + 5, int(cfg["phase2_frac"] * cfg["epochs"]))
    eval_every = cfg.get("eval_every", 5)
    clip_max = float(cfg.get("grad_clip_max", 3.0))

    history = {
        "epoch": [],
        "train_pred": [],
        "val_loss": [],
        "gen_gap": [],
        "train_acc": [],
        "val_acc": [],
        "act_mean": [],
        "act_std": [],
        "act_dist": [],
        "sym_ppx": [],
        "top1_pct": [],
        "vq_cos": [],
        "adj_ppx": [],
        "active_syms": [],
        "stk_depth": [],
        "stk_push": [],
        "stk_pop": [],
        "stk_noop": [],
        "gwt_p": [],
        "gwt_s": [],
        "gwt_h": [],
        "gwt_g": [],
        "grad_avg": [],
        "clip_pct": [],
        "lr": [],
        "gm_cell": [],
        "gm_att": [],
        "gm_ffn": [],
        "gm_vq": [],
        "gm_adj": [],
        "gm_holo": [],
        "gm_dec": [],
        "phase1_end": phase1_end,
        "phase2_end": phase2_end,
        "best_epoch": -1,
        "lr_drop_epochs": [],
    }
    prev_logged_lr = None

    t_start = time.time()
    tr_time_since_log = 0.0
    va_time_since_log = 0.0
    io_time_since_log = 0.0
    revived_total = 0
    skipped_spikes_since_log = 0
    last_ep_revived = 0
    min_active_converge = max(4, min(16, cfg["n_symbols"] // 4))
    best_metric = float("inf")
    best_epoch = -1
    best_state = None
    no_improve_logs = 0
    improved_since_log = False
    improved_since_sched = False

    live_div_weight = cfg["diversity_weight"]
    live_commit_cost = cfg["commitment_cost"]
    live_ponder_penalty = cfg["ponder_penalty"]
    ema_gnorm = cfg["grad_clip"]
    readiness = 0.0
    ema_pred = None

    act_low = 1.40
    act_high = max(2.60, 0.80 * cfg["max_recursion_depth"])

    revive_accum_counts = torch.zeros(cfg["n_symbols"], device=DEVICE)
    revive_lookups_accum = 0
    min_revive_lookups = max(512, 8 * cfg["n_symbols"])
    epoch_z_pool = []
    phase1_z_pool = []
    tokens_since_log = 0
    clipped_chunks_since_log = 0
    total_chunks_since_log = 0

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"--- Training SACRSN v55 (BF16_AMP={use_amp} | FusedAdamW={fused_used} | TBPTT={cfg['seq_len']}) ---")

    try:
        for epoch in range(cfg["epochs"]):
            t_ep_train_0 = time.time()
            model.train()

            if epoch == phase1_end:
                crystallize_codebook_kmeans(model, phase1_z_pool, cfg["n_symbols"])
                reset_param_adam_state(opt, model.vq_layer.codebook)
                phase1_z_pool.clear()
                revive_accum_counts.zero_()
                revive_lookups_accum = 0
                print(f"[Phase 2 Crystallization @ Ep {epoch:04d}] Initialized VQ codebook via GPU K-Means (AdamW state reset).")
            elif epoch == phase2_end:
                print(f"[Phase 3 Reasoning @ Ep {epoch:04d}] Unlocked Multi-Hop Graph Echo & Full Ethical Rules.")

            if epoch < phase1_end:
                phase_id = 1
                model.sym_alpha.fill_(0.0)
                model.graph_gate.fill_(0.0)
                vq_scale = 0.25
                div_scale_phase = 0.25
                cons_weight_live = 0.0
                live_eth_weight = 0.0
                live_bias_scale = 0.0
                live_conscience = 0.0
            elif epoch < phase2_end:
                phase_id = 2
                p2_prog = (epoch - phase1_end + 1) / float(max(1, phase2_end - phase1_end))
                model.sym_alpha.fill_(0.30 * p2_prog)
                model.graph_gate.fill_(0.0)
                vq_scale = cfg["vq_weight"]
                div_scale_phase = 1.0
                cons_weight_live = cfg["symbol_consistency_weight"] * p2_prog
                # Fixed: use p2_prog so adjacency warms up during Phase 2 instead of sitting at 0.0
                live_eth_weight = 0.25 * cfg["ethical_weight"] * p2_prog
                live_bias_scale = 0.0
                live_conscience = float(cfg["conscience_p2"])
            else:
                phase_id = 3
                p3_ramp = min(1.0, (epoch - phase2_end + 1) / 25.0)
                model.graph_gate.fill_(p3_ramp)
                vq_scale = cfg["vq_weight"]
                div_scale_phase = 1.0
                cons_weight_live = cfg["symbol_consistency_weight"]
                live_eth_weight = cfg["ethical_weight"] * readiness
                live_bias_scale = float(torch.clamp(model.vq_layer.ema_d_std * 0.30, min=0.02, max=0.20).item()) * readiness
                live_conscience = float(cfg["conscience_p3"])

            if epoch < lr_warmup_epochs:
                lr_scale = (epoch + 1) / float(lr_warmup_epochs)
                for pg, blr in zip(opt.param_groups, base_lrs):
                    pg["lr"] = blr * lr_scale

            streams = build_epoch_streams(
                data_tensor,
                cfg["batch_size"],
                cfg["epoch_token_cap"],
                epoch,
                stream_len_override=cfg.get("stream_len"),
                device=DEVICE,
            )
            bs, span_len = streams.shape
            stream_len = span_len - 1

            hidden = None
            prev_sym = None

            total_loss_t = torch.zeros((), device=DEVICE)
            total_pred_t = torch.zeros((), device=DEVICE)
            total_vq_t = torch.zeros((), device=DEVICE)
            total_vq_cos_t = torch.zeros((), device=DEVICE)
            total_eth_t = torch.zeros((), device=DEVICE)
            total_cons_t = torch.zeros((), device=DEVICE)
            total_div_t = torch.zeros((), device=DEVICE)
            total_ponder_t = torch.zeros((), device=DEVICE)
            total_stack_t = torch.zeros((), device=DEVICE)
            total_ptr_ent_t = torch.zeros((), device=DEVICE)
            total_mem_occ_t = torch.zeros((), device=DEVICE)

            total_w_pred_t = torch.zeros((), device=DEVICE)
            total_w_vq_t = torch.zeros((), device=DEVICE)
            total_w_eth_t = torch.zeros((), device=DEVICE)
            total_w_cons_t = torch.zeros((), device=DEVICE)
            total_w_stk_t = torch.zeros((), device=DEVICE)
            total_w_pnd_t = torch.zeros((), device=DEVICE)

            stack_ops_epoch = torch.zeros(3, device=DEVICE)
            gwt_gates_epoch = torch.zeros(4, device=DEVICE)
            sig_rms_epoch = torch.zeros(4, device=DEVICE)
            carry_mean_epoch = torch.zeros((), device=DEVICE)
            carry_std_epoch = torch.zeros((), device=DEVICE)
            carry_steps = 0

            total_correct_t = torch.zeros((), dtype=torch.long, device=DEVICE)
            self_loops_t = torch.zeros((), dtype=torch.long, device=DEVICE)
            act_step_hist = torch.zeros(cfg["max_recursion_depth"], dtype=torch.long, device=DEVICE)
            act_steps_sum_t = torch.zeros((), device=DEVICE, dtype=torch.float32)
            act_steps_sq_sum_t = torch.zeros((), device=DEVICE, dtype=torch.float32)
            trans_toks = 0
            total_toks = 0
            grad_norms = []

            progress = epoch / max(1, cfg["epochs"] - 1)
            entropy_weight = 0.01 * (1.0 - progress)
            in_revive_phase = (phase_id >= 2) and (epoch < int(0.60 * cfg["epochs"]))
            use_echo = (phase_id == 3)

            chunk_loss = 0.0
            chunk_probs = []
            chunk_weights = []
            steps_in_chunk = 0
            chunk_cb_norm = None
            chunk_P_adj = None

            for t in range(stream_len):
                if steps_in_chunk == 0:
                    chunk_cb_norm = model.vq_layer.normalized_codebook()
                    chunk_P_adj = F.softmax(model.vq_layer.adjacency.float(), dim=-1) if use_echo else None

                x = streams[:, t].unsqueeze(1)
                y = streams[:, t + 1]

                with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=use_amp):
                    (
                        logits,
                        hidden,
                        sym_idx,
                        ponder_loss,
                        vq_loss,
                        eth_loss,
                        avg_stack,
                        soft_probs,
                        active_w,
                        t0_z_flats,
                        avg_stk_ctrl,
                        steps_vec,
                        ptr_ent,
                        mem_occ_pct,
                        vq_cos,
                        cons_loss,
                        avg_gwt_gates,
                        extra_stats,
                    ) = model(
                        x,
                        hidden,
                        prev_sym,
                        bias_scale=live_bias_scale,
                        commit_cost=live_commit_cost,
                        return_stats=True,
                        use_graph_echo=use_echo,
                        cached_P_adj=chunk_P_adj,
                        cached_cb_norm=chunk_cb_norm,
                        conscience_scale=live_conscience,
                    )

                    # True PonderNet expected cross-entropy across thinking steps: sum_t p_t * CE(logits_t, y)
                    step_logits_t = extra_stats["step_logits"]
                    step_p_t = extra_stats["step_p"]
                    n_run_steps = step_logits_t.size(1)
                    step_ce = F.cross_entropy(
                        step_logits_t.float().reshape(-1, vocab_size),
                        y.unsqueeze(1).expand(-1, n_run_steps).reshape(-1),
                        reduction="none",
                    ).view(bs, n_run_steps)
                    loss_pred = (step_p_t * step_ce).sum(dim=-1).mean()
                    mix_nll_det = F.cross_entropy(logits.detach().float(), y)

                    loss_ponder = live_ponder_penalty * ponder_loss

                    probs = F.softmax(logits.float(), dim=-1)
                    loss_entropy = -entropy_weight * (-(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean())

                    loss_vq = vq_scale * vq_loss
                    loss_ethics = live_eth_weight * eth_loss
                    loss_cons = cons_weight_live * cons_loss

                    # Relaxed NoOp cap (<= 85%) so stack doesn't thrash on flat tokens while staying active (>= 10%)
                    loss_stk_bal = cfg["stack_balance_weight"] * (
                        (avg_stk_ctrl[0] - avg_stk_ctrl[1]) ** 2
                        + 0.5 * (F.relu(avg_stk_ctrl[2] - 0.85) ** 2)
                        + 0.25 * (F.relu(0.10 - (avg_stk_ctrl[0] + avg_stk_ctrl[1])) ** 2)
                    )

                    step_loss = (
                        loss_pred
                        + loss_ponder
                        + loss_vq
                        + loss_entropy
                        + loss_ethics
                        + loss_cons
                        + loss_stk_bal
                    )

                if prev_sym is not None:
                    self_loops_t += (prev_sym == sym_idx).sum()
                    trans_toks += bs

                prev_sym = sym_idx.detach()

                chunk_loss = chunk_loss + step_loss
                chunk_probs.append(soft_probs)
                chunk_weights.append(active_w)

                if phase_id == 1 and t % 4 == 0:
                    phase1_z_pool.append(t0_z_flats[:128])
                    if len(phase1_z_pool) > 32:
                        phase1_z_pool.pop(0)
                elif in_revive_phase and len(epoch_z_pool) < 64:
                    epoch_z_pool.append(t0_z_flats)

                steps_in_chunk += 1

                # Log mixture NLL apples-to-apples with evaluate_val
                total_pred_t += mix_nll_det
                total_vq_t += vq_loss.detach()
                total_vq_cos_t += vq_cos
                total_eth_t += eth_loss.detach()
                total_cons_t += cons_loss.detach()
                total_ponder_t += ponder_loss.detach()
                total_stack_t += avg_stack.detach()
                total_ptr_ent_t += ptr_ent
                total_mem_occ_t += mem_occ_pct

                total_w_pred_t += loss_pred.detach()
                total_w_vq_t += loss_vq.detach()
                total_w_eth_t += loss_ethics.detach()
                total_w_cons_t += loss_cons.detach()
                total_w_stk_t += loss_stk_bal.detach()
                total_w_pnd_t += loss_ponder.detach()

                stack_ops_epoch += avg_stk_ctrl.detach()
                gwt_gates_epoch += avg_gwt_gates
                sig_rms_epoch += extra_stats["sig_rms"]
                if t > 0:
                    carry_mean_epoch += extra_stats["carry_mean"]
                    carry_std_epoch += extra_stats["carry_std"]
                    carry_steps += 1

                total_correct_t += (logits.detach().argmax(dim=-1) == y).sum()

                steps_idx = (steps_vec.long() - 1).clamp(min=0, max=cfg["max_recursion_depth"] - 1)
                act_step_hist.index_add_(0, steps_idx, torch.ones_like(steps_idx))
                act_steps_sum_t += steps_vec.sum()
                act_steps_sq_sum_t += (steps_vec ** 2).sum()

                total_toks += bs
                tokens_since_log += bs

                if steps_in_chunk == cfg["seq_len"] or t == stream_len - 1:
                    probs_cat = torch.cat(chunk_probs, dim=0)
                    w_cat = torch.cat(chunk_weights, dim=0)
                    w_sum = w_cat.sum().clamp(min=1.0)
                    w_exp = w_cat.unsqueeze(-1)

                    avg_probs = (probs_cat * w_exp).sum(dim=0) / w_sum
                    marginal_entropy = -(avg_probs * torch.log(avg_probs + 1e-8)).sum()
                    norm_kl = (log_K - marginal_entropy) / max(1e-8, log_K)
                    loss_diversity = (div_scale_phase * live_div_weight) * norm_kl

                    total_chunk_loss = (chunk_loss / steps_in_chunk) + loss_diversity

                    clip_thresh = max(0.5, min(clip_max, 1.8 * ema_gnorm))
                    opt.zero_grad(set_to_none=True)
                    total_chunk_loss.backward()
                    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_thresh).item()

                    if math.isfinite(gnorm) and gnorm < 100.0:
                        ema_gnorm = 0.95 * ema_gnorm + 0.05 * min(gnorm, 10.0)
                        if gnorm > clip_thresh:
                            clipped_chunks_since_log += 1
                        with torch.no_grad():
                            for k_mod, plist in mod_param_groups.items():
                                sq_s = torch.zeros((), device=DEVICE)
                                for p_m in plist:
                                    if p_m.grad is not None:
                                        g_m = p_m.grad.detach().float()
                                        sq_s = sq_s + (g_m * g_m).sum()
                                mod_gnorm_sums[k_mod] = mod_gnorm_sums[k_mod] + torch.sqrt(sq_s)
                            mod_gnorm_count += 1
                        opt.step()
                        # Keep codebook strictly unit-RMS so cb_f and cb_norm never drift apart
                        model.vq_layer.normalize_codebook_()
                        grad_norms.append(gnorm)
                    else:
                        opt.zero_grad(set_to_none=True)
                        skipped_spikes_since_log += 1

                    total_chunks_since_log += 1

                    h_r, h_i, h_mem, h_ptr, h_hr, h_hi = hidden
                    hidden = (
                        h_r.detach(),
                        h_i.detach(),
                        h_mem.detach(),
                        h_ptr.detach(),
                        h_hr.detach(),
                        h_hi.detach(),
                    )

                    total_loss_t += total_chunk_loss.detach() * steps_in_chunk
                    total_div_t += loss_diversity.detach() * steps_in_chunk
                    chunk_loss = 0.0
                    chunk_probs = []
                    chunk_weights = []
                    steps_in_chunk = 0

            tr_time_since_log += time.time() - t_ep_train_0

            avg_pred = total_pred_t.item() / stream_len
            avg_vq = total_vq_t.item() / stream_len
            avg_act_steps = act_steps_sum_t.item() / max(1, total_toks)
            act_var = max(0.0, (act_steps_sq_sum_t.item() / max(1, total_toks)) - (avg_act_steps ** 2))
            act_std = math.sqrt(act_var)
            ema_pred = avg_pred if ema_pred is None else (0.80 * ema_pred + 0.20 * avg_pred)

            with torch.no_grad():
                counts = model.vq_layer.usage_count.clone()
                active_syms = int((counts > 0).sum().item())
                dist = counts / (counts.sum() + 1e-10)
                sym_drift = float((dist - model.prev_sym_soft).abs().sum().item() * 0.5)
                model.prev_sym_soft.mul_(0.85).add_(0.15 * dist)
                entropy_val = -(dist * torch.log(dist + 1e-10)).sum()
                true_ppx = torch.exp(entropy_val).item()
                top1_pct = float((counts.max() / (counts.sum() + 1e-10) * 100.0).item())

                revive_accum_counts.add_(counts)
                revive_lookups_accum += int(counts.sum().item())

                ep_revived = 0
                if (
                    in_revive_phase
                    and epoch % cfg["revive_every"] == 0
                    and revive_lookups_accum >= min_revive_lookups
                    and len(epoch_z_pool) > 0
                ):
                    dead_mask = revive_accum_counts <= 1.0
                    n_dead = int(dead_mask.sum().item())
                    if n_dead > 0 and (true_ppx < ppx_low or n_dead > int(0.05 * cfg["n_symbols"])):
                        pool = torch.cat(epoch_z_pool, dim=0)
                        rand_idx = torch.randint(0, pool.size(0), (n_dead,), device=DEVICE)
                        new_vecs = pool[rand_idx] + 0.05 * torch.randn(n_dead, pool.size(1), device=DEVICE)
                        new_vecs = new_vecs / torch.sqrt(torch.mean(new_vecs ** 2, dim=-1, keepdim=True) + 1e-4)
                        model.vq_layer.codebook.data[dead_mask] = new_vecs.to(model.vq_layer.codebook.dtype)
                        model.vq_layer.normalize_codebook_()
                        model.vq_layer.ema_usage.data[dead_mask] = 1.0 / float(cfg["n_symbols"])
                        reset_param_adam_state(opt, model.vq_layer.codebook, row_mask=dead_mask)
                        ep_revived = n_dead
                        revived_total += n_dead
                    revive_accum_counts.zero_()
                    revive_lookups_accum = 0
                    epoch_z_pool.clear()
                elif not in_revive_phase and len(epoch_z_pool) > 0:
                    epoch_z_pool.clear()

                last_ep_revived = ep_revived
                model.vq_layer.usage_count.zero_()

                if phase_id >= 2:
                    if true_ppx < ppx_low or top1_pct > 15.0:
                        live_div_weight = min(2.5, live_div_weight * 1.06)
                    elif true_ppx > ppx_high and top1_pct < 8.0:
                        live_div_weight = max(0.05, live_div_weight * 0.90)
                    else:
                        live_div_weight = max(0.15, live_div_weight * 0.96)

                    if avg_vq > 0.80:
                        live_commit_cost = min(0.40, live_commit_cost * 1.04)
                    elif avg_vq < 0.35 and avg_pred > 1.5:
                        live_commit_cost = max(0.10, live_commit_cost * 0.96)

                    if act_std < 0.25 and avg_act_steps <= 2.4:
                        live_ponder_penalty = max(0.002, live_ponder_penalty * 0.94)
                    elif avg_act_steps > act_high:
                        live_ponder_penalty = min(0.15, live_ponder_penalty * 1.05)
                    elif avg_act_steps < act_low:
                        live_ponder_penalty = max(0.002, live_ponder_penalty * 0.94)

                if phase_id == 3:
                    target_alpha = min(0.40, max(0.20, 0.40 - 0.25 * max(0.0, avg_vq - 0.25)))
                    model.sym_alpha.mul_(0.85).add_(0.15 * target_alpha)

                    u_score = min(1.0, true_ppx / max(1.0, ppx_low))
                    vq_score = min(1.0, max(0.0, (1.2 - avg_vq) / 0.5))
                    rev_score = max(0.25, 1.0 - last_ep_revived / max(1.0, 0.25 * cfg["n_symbols"]))
                    p3_ramp = min(1.0, (epoch - phase2_end + 1) / 15.0)
                    readiness = 0.85 * readiness + 0.15 * (u_score * vq_score * rev_score * p3_ramp)

            is_last_epoch = (epoch == cfg["epochs"] - 1)
            is_log_epoch = (epoch % cfg["log_every"] == 0) or is_last_epoch
            is_eval_epoch = (epoch % eval_every == 0) or is_log_epoch
            is_phase1_end = (epoch == phase1_end - 1)
            is_phase2_end = (epoch == phase2_end - 1)
            is_phase3_end = is_last_epoch
            printed_sample_this_epoch = False
            is_new_best = False
            is_ready = (
                (phase_id == 3)
                and (readiness >= 0.50 or epoch >= phase2_end + cfg["warmup_epochs"])
                and active_syms >= min_active_converge
            )
            acc_pct = 100.0 * total_correct_t.item() / max(1, total_toks)

            val_loss, val_acc = None, None
            monitor_loss = ema_pred

            if is_ready:
                t_va_0 = time.time()
                val_loss, val_acc = evaluate_val(model, val_batches, use_amp=use_amp)
                va_time_since_log += time.time() - t_va_0
                monitor_loss = val_loss if val_loss is not None else ema_pred
                if monitor_loss < best_metric - 1e-3:
                    best_metric = monitor_loss
                    best_epoch = epoch
                    history["best_epoch"] = best_epoch
                    t_io_0 = time.time()
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    save_checkpoint(model, tokenizer, cfg, save_path)
                    io_time_since_log += time.time() - t_io_0
                    is_new_best = True
                    improved_since_log = True
                    improved_since_sched = True

                if epoch >= lr_warmup_epochs and is_eval_epoch:
                    sched_metric = best_metric if improved_since_sched else monitor_loss
                    plateau_scheduler.step(sched_metric)
                    improved_since_sched = False

            if is_log_epoch:
                if not is_ready:
                    t_va_0 = time.time()
                    val_loss, val_acc = evaluate_val(model, val_batches, use_amp=use_amp)
                    va_time_since_log += time.time() - t_va_0
                    monitor_loss = val_loss if val_loss is not None else ema_pred

                if epoch >= lr_warmup_epochs and is_ready:
                    if improved_since_log:
                        no_improve_logs = 0
                    elif progress >= 0.4:
                        no_improve_logs += 1
                    improved_since_log = False

                avg_loss = total_loss_t.item() / stream_len
                avg_eth = total_eth_t.item() / stream_len
                avg_cons = total_cons_t.item() / stream_len
                avg_div = total_div_t.item() / stream_len
                avg_stk = total_stack_t.item() / stream_len
                avg_vq_cos = total_vq_cos_t.item() / stream_len
                ptr_ppx = math.exp(min(10.0, total_ptr_ent_t.item() / stream_len))
                mem_occ = total_mem_occ_t.item() / stream_len
                adj_ppx = math.exp(min(12.0, avg_eth))
                stk_ops = (stack_ops_epoch / stream_len * 100.0).tolist()
                gwt_pcts = (gwt_gates_epoch / stream_len * 100.0).tolist()
                sig_rms_vals = (sig_rms_epoch / stream_len).tolist()
                carry_m = (carry_mean_epoch.item() / max(1, carry_steps)) if carry_steps > 0 else 0.0
                carry_s = (carry_std_epoch.item() / max(1, carry_steps)) if carry_steps > 0 else 0.0

                w_pred_v = total_w_pred_t.item() / stream_len
                w_vq_v = total_w_vq_t.item() / stream_len
                w_eth_v = total_w_eth_t.item() / stream_len
                w_cons_v = total_w_cons_t.item() / stream_len
                w_stk_v = total_w_stk_t.item() / stream_len
                w_pnd_v = total_w_pnd_t.item() / stream_len
                w_div_v = avg_div
                w_tot = max(1e-6, w_pred_v + w_vq_v + w_eth_v + w_cons_v + w_stk_v + w_pnd_v + w_div_v)

                mod_avgs = {k_m: (mod_gnorm_sums[k_m].item() / max(1, mod_gnorm_count)) for k_m in mod_param_groups}
                for k_m in mod_param_groups:
                    mod_gnorm_sums[k_m].zero_()
                mod_gnorm_count = 0

                tok_ppx = float(np.exp(min(avg_pred, 20.0)))
                loop_pct = 100.0 * self_loops_t.item() / max(1, trans_toks)
                strong_edges = int((F.softmax(model.vq_layer.adjacency.float(), dim=-1) > 0.08).sum().item())
                avg_gnorm = sum(grad_norms) / max(1, len(grad_norms))
                max_gnorm = max(grad_norms) if grad_norms else 0.0
                clip_pct = 100.0 * clipped_chunks_since_log / max(1, total_chunks_since_log)
                clip_thresh_now = max(0.5, min(clip_max, 1.8 * ema_gnorm))

                act_dist = (act_step_hist.float() / max(1, total_toks) * 100.0).tolist()
                act_dist_str = "/".join(f"{p:2.0f}" for p in act_dist)

                dt = max(1e-4, time.time() - t_start)
                tok_per_sec = tokens_since_log / dt
                t_start = time.time()
                tokens_since_log = 0
                clipped_chunks_since_log = 0
                total_chunks_since_log = 0
                lr = opt.param_groups[0]["lr"]

                if prev_logged_lr is not None and lr < prev_logged_lr * 0.75 and epoch >= lr_warmup_epochs:
                    history["lr_drop_epochs"].append(epoch)
                prev_logged_lr = lr

                history["epoch"].append(epoch)
                history["train_pred"].append(avg_pred)
                history["val_loss"].append(val_loss if val_loss is not None else float("nan"))
                history["gen_gap"].append((val_loss - avg_pred) if val_loss is not None else float("nan"))
                history["train_acc"].append(acc_pct)
                history["val_acc"].append(val_acc if val_acc is not None else float("nan"))
                history["act_mean"].append(avg_act_steps)
                history["act_std"].append(act_std)
                history["act_dist"].append(act_dist)
                history["sym_ppx"].append(true_ppx)
                history["top1_pct"].append(top1_pct)
                history["vq_cos"].append(avg_vq_cos)
                history["adj_ppx"].append(adj_ppx)
                history["active_syms"].append(active_syms)
                history["stk_depth"].append(avg_stk)
                history["stk_push"].append(stk_ops[0])
                history["stk_pop"].append(stk_ops[1])
                history["stk_noop"].append(stk_ops[2])
                history["gwt_p"].append(gwt_pcts[0])
                history["gwt_s"].append(gwt_pcts[1])
                history["gwt_h"].append(gwt_pcts[2])
                history["gwt_g"].append(gwt_pcts[3])
                history["grad_avg"].append(avg_gnorm)
                history["clip_pct"].append(clip_pct)
                history["lr"].append(lr)
                history["gm_cell"].append(mod_avgs["Cell"])
                history["gm_att"].append(mod_avgs["Att"])
                history["gm_ffn"].append(mod_avgs["FFN"])
                history["gm_vq"].append(mod_avgs["VQ"])
                history["gm_adj"].append(mod_avgs["Adj"])
                history["gm_holo"].append(mod_avgs["Holo"])
                history["gm_dec"].append(mod_avgs["Dec"])

                if val_loss is not None:
                    val_ppx = float(np.exp(min(val_loss, 20.0)))
                    gen_gap = val_loss - avg_pred
                    best_tag = "*BEST*" if is_new_best else (f"Best:{best_metric:.3f}@{best_epoch}" if best_epoch >= 0 else "")
                    val_str = f" | Val:{val_loss:.3f} ({val_acc:4.1f}% PPX:{val_ppx:5.1f} Gap:{gen_gap:+.2f}) {best_tag}"
                else:
                    best_tag = "*BEST*" if is_new_best else (f"Best:{best_metric:.3f}@{best_epoch}" if best_epoch >= 0 else "")
                    val_str = f" | {best_tag}" if best_tag else ""

                vram_str = (
                    f" VRAM:{torch.cuda.max_memory_allocated() / (1024 ** 3):.1f}G"
                    if DEVICE == "cuda"
                    else ""
                )

                print(
                    f"Ep {epoch:04d} [P{phase_id}] | Loss:{avg_loss:.3f} (Pred:{avg_pred:.3f} VQ:{avg_vq:.3f} "
                    f"Div:{avg_div:.3f} Eth:{avg_eth:.3f} Cons:{avg_cons:.3f}) | Acc:{acc_pct:5.1f}% TokPPX:{tok_ppx:5.2f}{val_str} | "
                    f"{tok_per_sec:,.0f} tok/s (Tr:{tr_time_since_log:.1f}s Va:{va_time_since_log:.1f}s Io:{io_time_since_log:.1f}s){vram_str}"
                )
                print(
                    f"        | ACT:{avg_act_steps:.2f}±{act_std:.2f}[{act_dist_str}%] "
                    f"Stk:{avg_stk:4.1f}({stk_ops[0]:2.0f}/{stk_ops[1]:2.0f}/{stk_ops[2]:2.0f} PtrPPX:{ptr_ppx:4.1f} Occ:{mem_occ:2.0f}%) "
                    f"GWT[P:{gwt_pcts[0]:2.0f} S:{gwt_pcts[1]:2.0f} H:{gwt_pcts[2]:2.0f} G:{gwt_pcts[3]:2.0f}] "
                    f"SigRMS[{sig_rms_vals[0]:.2f}/{sig_rms_vals[1]:.2f}/{sig_rms_vals[2]:.2f}/{sig_rms_vals[3]:.2f}] "
                    f"CarryG:{carry_m:.2f}±{carry_s:.2f} | Grad:{avg_gnorm:.2f}/{max_gnorm:.1f} Skips:{skipped_spikes_since_log} LR:{lr:.5f}"
                )
                print(
                    f"        | GradMod[Cell:{mod_avgs['Cell']:.2f} Att:{mod_avgs['Att']:.2f} FFN:{mod_avgs['FFN']:.2f} "
                    f"VQ:{mod_avgs['VQ']:.3f} Adj:{mod_avgs['Adj']:.3f} Holo:{mod_avgs['Holo']:.2f} Dec:{mod_avgs['Dec']:.2f}] | "
                    f"LossPct[Pr:{100*w_pred_v/w_tot:2.0f}% VQ:{100*w_vq_v/w_tot:2.0f}% Et:{100*w_eth_v/w_tot:2.0f}% "
                    f"Cn:{100*w_cons_v/w_tot:2.0f}% St:{100*w_stk_v/w_tot:2.0f}% Pn:{100*w_pnd_v/w_tot:2.0f}% Dv:{100*w_div_v/w_tot:2.0f}%]"
                )
                print(
                    f"        | Syms:{active_syms:3d}/{cfg['n_symbols']} SymPPX:{true_ppx:5.1f} VQCos:{avg_vq_cos:.2f} Drift:{sym_drift:.3f} "
                    f"Top1:{top1_pct:4.1f}% Loop:{loop_pct:4.1f}% Rev:{revived_total:3d} Edges:{strong_edges:4d}(AdjPPX:{adj_ppx:5.1f}) | "
                    f"Tune[Ready:{readiness*100:3.0f}% DivW:{live_div_weight:.2f} Temp:{model.vq_layer.live_temp:.2f} Cmt:{live_commit_cost:.2f} "
                    f"Alpha:{model.sym_alpha.item():.2f} Bias:{live_bias_scale:.3f} GGate:{model.graph_gate.item():.2f} PndP:{live_ponder_penalty:.4f} "
                    f"Consc:{live_conscience:.2f} Clip:{clip_thresh_now:.2f}({clip_pct:2.0f}%) Pat:{no_improve_logs}/{cfg['early_stop_patience']}]"
                )
                revived_total = 0
                skipped_spikes_since_log = 0
                tr_time_since_log = 0.0
                va_time_since_log = 0.0
                io_time_since_log = 0.0

            if is_new_best:
                t_io_0 = time.time()
                if val_loss is not None:
                    val_ppx_b = float(np.exp(min(val_loss, 20.0)))
                    print(
                        f"   >>> *BEST* EPOCH {epoch:04d} | Val Loss: {best_metric:.4f} "
                        f"(Acc: {val_acc:4.1f}% PPX: {val_ppx_b:5.1f}) | Saved checkpoint -> {save_path}"
                    )
                else:
                    print(
                        f"   >>> *BEST* EPOCH {epoch:04d} | Pred Loss (EMA): {best_metric:.4f} "
                        f"(Pred: {avg_pred:.4f} Acc: {acc_pct:4.1f}%) | Saved checkpoint -> {save_path}"
                    )
                print(f"   Sample (*BEST* Ep {epoch:04d}): {repr(generate_sample_preview(model, tokenizer, data_tensor, cfg, use_amp=use_amp))}")
                io_time_since_log += time.time() - t_io_0
                printed_sample_this_epoch = True

            if is_log_epoch and (epoch % cfg["sample_every"] == 0) and not printed_sample_this_epoch and not (is_phase1_end or is_phase2_end or is_phase3_end):
                print(f"   Sample: {repr(generate_sample_preview(model, tokenizer, data_tensor, cfg, use_amp=use_amp))}")
                printed_sample_this_epoch = True

            if is_phase1_end:
                print(f"   [End of Phase 1 Sample @ Ep {epoch:04d}]: {repr(generate_sample_preview(model, tokenizer, data_tensor, cfg, use_amp=use_amp))}")
            elif is_phase2_end:
                print(f"   [End of Phase 2 Sample @ Ep {epoch:04d}]: {repr(generate_sample_preview(model, tokenizer, data_tensor, cfg, use_amp=use_amp))}")
            elif is_phase3_end:
                print(f"   [End of Phase 3 Sample @ Ep {epoch:04d}]: {repr(generate_sample_preview(model, tokenizer, data_tensor, cfg, use_amp=use_amp))}")

            if is_ready:
                small_text_converged = (
                    val_tensor is None
                    and epoch >= phase2_end + 30
                    and avg_pred < 0.025
                    and acc_pct >= 98.5
                )
                if avg_pred < 0.01 or small_text_converged:
                    print("\n--- CONVERGENCE REACHED ---")
                    if not is_new_best:
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        if val_tensor is None:
                            best_metric = avg_pred
                            best_epoch = epoch
                            history["best_epoch"] = best_epoch
                        save_checkpoint(model, tokenizer, cfg, save_path)
                        print(f"   >>> *BEST* EPOCH {best_epoch:04d} | Metric: {best_metric:.4f} (Acc: {acc_pct:.1f}%) | Saved checkpoint -> {save_path}")
                    if not is_phase3_end:
                        print(f"   [End of Phase 3 Sample @ Ep {epoch:04d}]: {repr(generate_sample_preview(model, tokenizer, data_tensor, cfg, use_amp=use_amp))}")
                    break

            if is_log_epoch and progress >= 0.4 and lr <= min_lrs[0] * 2.1 and no_improve_logs >= cfg["early_stop_patience"]:
                print(f"\n--- EARLY STOPPING (LR converged & plateaued for {no_improve_logs} checks) ---")
                if not is_phase3_end:
                    print(f"   [End of Phase 3 Sample @ Ep {epoch:04d}]: {repr(generate_sample_preview(model, tokenizer, data_tensor, cfg, use_amp=use_amp))}")
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    if best_state is not None:
        model.load_state_dict(best_state)
        metric_name = "Val Loss" if val_tensor is not None else "Pred Loss (EMA)"
        print(f"Restored best healthy checkpoint ({metric_name}: {best_metric:.4f} at Ep {best_epoch}).")
    elif save_path is not None:
        save_checkpoint(model, tokenizer, cfg, save_path)
        print(f"Saved final checkpoint -> {save_path}")

    return model, history


# ==========================================
# 8. Visualization Suite (0..6 Diagnostics)
# ==========================================
def plot_training_dynamics(history, config):
    epochs = history.get("epoch", [])
    if len(epochs) < 2:
        print("Not enough logged epochs for 0_training_dynamics.png.")
        return

    fig, axes = plt.subplots(3, 2, figsize=(16, 13.5))
    p1_end = history.get("phase1_end", -1)
    p2_end = history.get("phase2_end", -1)
    best_ep = history.get("best_epoch", -1)
    lr_drops = history.get("lr_drop_epochs", [])

    def _add_phase_markers(ax, show_legend_labels=False):
        if p1_end > 0:
            ax.axvline(p1_end, color="gray", linestyle="--", linewidth=1.1, alpha=0.7, label="P2 KMeans" if show_legend_labels else None)
        if p2_end > 0:
            ax.axvline(p2_end, color="teal", linestyle="--", linewidth=1.1, alpha=0.7, label="P3 Graph+Eth" if show_legend_labels else None)
        if best_ep >= 0:
            ax.axvline(best_ep, color="gold", linestyle="-.", linewidth=1.4, alpha=0.9, label=f"Best Ep {best_ep}" if show_legend_labels else None)
        for idx_d, d_ep in enumerate(lr_drops):
            ax.axvline(d_ep, color="navy", linestyle=":", linewidth=1.0, alpha=0.6, label="LR Drop" if (show_legend_labels and idx_d == 0) else None)

    ax1 = axes[0, 0]
    tr_pred = np.array(history["train_pred"], dtype=np.float64)
    va_loss = np.array(history["val_loss"], dtype=np.float64)
    ax1.plot(epochs, tr_pred, color="royalblue", linewidth=2.0, label="Train Pred Loss")
    if np.isfinite(va_loss).any():
        ax1.plot(epochs, va_loss, color="crimson", linewidth=2.0, label="Val Loss")
        valid_m = np.isfinite(va_loss)
        ep_arr = np.array(epochs)
        ax1.fill_between(ep_arr[valid_m], tr_pred[valid_m], va_loss[valid_m], color="crimson", alpha=0.10, label="Gen Gap")
    _add_phase_markers(ax1, show_legend_labels=True)
    ax1.set_title("Loss Trajectory & Generalization Gap")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right", fontsize=8, ncol=2)

    ax2 = axes[0, 1]
    act_mat = np.array(history["act_dist"], dtype=np.float64)
    step_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
    for d_i in range(act_mat.shape[1]):
        ax2.plot(epochs, act_mat[:, d_i], color=step_colors[d_i % len(step_colors)], linewidth=1.8, label=f"{d_i+1}-Step %")
    ax2_r = ax2.twinx()
    ax2_r.plot(epochs, history["act_mean"], color="black", linestyle="--", linewidth=1.5, label="Mean Steps")
    ax2_r.set_ylabel("Mean ACT Steps", color="black")
    ax2_r.set_ylim(0.5, config["max_recursion_depth"] + 0.3)
    _add_phase_markers(ax2)
    ax2.set_title("ACT Depth Distribution Over Training")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Token Share (%)")
    ax2.set_ylim(-2, 102)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", fontsize=8)
    ax2_r.legend(loc="upper right", fontsize=8)

    ax3 = axes[1, 0]
    ax3.plot(epochs, history["sym_ppx"], color="darkorchid", linewidth=2.0, label="SymPPX")
    ax3.axhspan(config["target_ppx_low"], config["target_ppx_high"], color="green", alpha=0.10, label="Target SymPPX Band")
    ax3.plot(epochs, np.minimum(history["adj_ppx"], float(config["n_symbols"])), color="darkorange", linewidth=1.6, linestyle="-.", label="AdjPPX (capped)")
    ax3_r = ax3.twinx()
    ax3_r.plot(epochs, history["top1_pct"], color="crimson", linewidth=1.4, linestyle=":", label="Top1 %")
    ax3_r.plot(epochs, np.array(history["vq_cos"]) * 100.0, color="seagreen", linewidth=1.4, linestyle="--", label="VQCos x100")
    ax3_r.set_ylabel("Top1% / VQCos(%)")
    _add_phase_markers(ax3)
    ax3.set_title("Codebook Perplexity, Top-1 Share & Graph Adjacency")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Perplexity")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper left", fontsize=8)
    ax3_r.legend(loc="upper right", fontsize=8)

    ax4 = axes[1, 1]
    ax4.plot(epochs, history["gwt_p"], color="steelblue", linewidth=1.6, label="GWT P%")
    ax4.plot(epochs, history["gwt_s"], color="purple", linewidth=1.6, label="GWT S%")
    ax4.plot(epochs, history["gwt_h"], color="teal", linewidth=1.6, label="GWT H%")
    ax4.plot(epochs, history["gwt_g"], color="darkorange", linewidth=1.6, label="GWT G%")
    ax4_r = ax4.twinx()
    ax4_r.plot(epochs, history["stk_depth"], color="black", linestyle="-.", linewidth=1.5, label="Stack Depth")
    ax4_r.set_ylabel("Stack Depth")
    _add_phase_markers(ax4)
    ax4.set_title("GWT Workspace Routing & Stack Depth")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("GWT Gate Share (%)")
    ax4.grid(True, alpha=0.25)
    ax4.legend(loc="upper left", fontsize=8, ncol=2)
    ax4_r.legend(loc="upper right", fontsize=8)

    ax5 = axes[2, 0]
    mod_specs = [
        ("gm_cell", "Cell", "royalblue"),
        ("gm_ffn", "FFN", "darkorange"),
        ("gm_dec", "Dec/Emb", "crimson"),
        ("gm_holo", "Holo", "teal"),
        ("gm_att", "Att", "purple"),
        ("gm_vq", "VQ", "seagreen"),
        ("gm_adj", "Adj", "saddlebrown"),
    ]
    for key_m, lbl_m, col_m in mod_specs:
        vals_m = np.maximum(np.array(history[key_m], dtype=np.float64), 1e-4)
        ax5.semilogy(epochs, vals_m, color=col_m, linewidth=1.6, label=lbl_m)
    _add_phase_markers(ax5)
    ax5.set_title("Per-Module Gradient Norms (Log Scale)")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("L2 Grad Norm (log)")
    ax5.grid(True, which="both", alpha=0.22)
    ax5.legend(loc="upper right", fontsize=8, ncol=2)

    ax6 = axes[2, 1]
    ax6.plot(epochs, history["train_acc"], color="royalblue", linewidth=2.0, label="Train Acc %")
    va_acc = np.array(history["val_acc"], dtype=np.float64)
    if np.isfinite(va_acc).any():
        ax6.plot(epochs, va_acc, color="crimson", linewidth=2.0, label="Val Acc %")
    ax6_r = ax6.twinx()
    ax6_r.semilogy(epochs, np.maximum(np.array(history["lr"], dtype=np.float64), 1e-7), color="black", linestyle=":", linewidth=1.5, label="LR (log)")
    ax6_r.set_ylabel("Learning Rate (log)")
    _add_phase_markers(ax6)
    ax6.set_title("Token Accuracy & Learning Rate Schedule")
    ax6.set_xlabel("Epoch")
    ax6.set_ylabel("Accuracy (%)")
    ax6.grid(True, alpha=0.25)
    ax6.legend(loc="upper left", fontsize=8)
    ax6_r.legend(loc="lower right", fontsize=8)

    fig.suptitle("SACRSN v55 — Training Dynamics Dashboard", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("0_training_dynamics.png", dpi=150)
    print("Saved 0_training_dynamics.png")
    plt.close()


def build_symbol_token_table(model, tokenizer, target_tensor, config, max_steps=None):
    if max_steps is None:
        max_steps = min(len(target_tensor) - 1, max(config["diag_max_tokens"], 20000))
    else:
        max_steps = min(len(target_tensor) - 1, max_steps)

    model.eval()
    dev = next(model.parameters()).device
    use_amp = _use_amp_for_model(model, config["batch_size"])
    symbol_to_tok = defaultdict(list)
    sym_token_counts = defaultdict(Counter)
    hidden, prev_sym = None, None

    cached_cb = model.vq_layer.normalized_codebook()
    cached_padj = F.softmax(model.vq_layer.adjacency.float(), dim=-1)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=use_amp):
        for i in range(max_steps):
            tok_id = target_tensor[i].item()
            x = torch.tensor([[tok_id]], device=dev)
            _, hidden, prev_sym, _, _, _, _ = model(
                x, hidden, prev_sym, cached_P_adj=cached_padj, cached_cb_norm=cached_cb
            )
            tok_str = tokenizer.format_token(tokenizer.id_to_token[tok_id])
            sym_id = prev_sym.item()
            symbol_to_tok[sym_id].append(tok_str)
            sym_token_counts[sym_id][tok_id] += 1

    return symbol_to_tok, sym_token_counts


def visualize_all(model, tokenizer, data_tensor, config):
    print("\n--- Generating Diagnostics & Images ---")
    model.eval()
    dev = next(model.parameters()).device
    use_amp = _use_amp_for_model(model, config["batch_size"])
    n_symbols = model.n_symbols

    diag_steps = min(len(data_tensor) - 1, config["diag_max_tokens"])
    symbol_to_tok, sym_token_counts = build_symbol_token_table(model, tokenizer, data_tensor, config, diag_steps)

    all_counts = np.zeros(n_symbols, dtype=np.int64)
    purity_map = {}
    total_hits = 0
    purity_hits = 0
    for s_idx in range(n_symbols):
        cntr = sym_token_counts.get(s_idx)
        if cntr:
            s_tot = sum(cntr.values())
            all_counts[s_idx] = s_tot
            total_hits += s_tot
            top_c = cntr.most_common(1)[0][1]
            purity_hits += top_c
            purity_map[s_idx] = 100.0 * top_c / max(1, s_tot)
        else:
            purity_map[s_idx] = 0.0

    sorted_counts = np.sort(all_counts)[::-1]
    nonzero_counts = sorted_counts[sorted_counts > 0]
    top10_k = max(1, n_symbols // 10)
    top10_share = 100.0 * float(sorted_counts[:top10_k].sum()) / max(1, total_hits)
    med_hits = float(np.median(nonzero_counts)) if len(nonzero_counts) > 0 else 0.0
    avg_purity = 100.0 * purity_hits / max(1, total_hits)
    print(
        f"Codebook Health : Active={len(nonzero_counts)}/{n_symbols} "
        f"| Top10%Share={top10_share:.1f}% | MedianHits={med_hits:.0f} | AvgPurity={avg_purity:.1f}%"
    )

    top_syms = sorted(sym_token_counts.items(), key=lambda kv: sum(kv[1].values()), reverse=True)[:5]
    for s_id, cntr in top_syms:
        s_tot = sum(cntr.values())
        top3_str = ", ".join(
            f"'{tokenizer.format_token(tokenizer.id_to_token[tid])}'({cnt})"
            for tid, cnt in cntr.most_common(3)
        )
        p_s = purity_map[s_id]
        print(f"   Hub S_{s_id:<4} (hits={s_tot:<4}, purity={p_s:4.1f}%): [{top3_str}]")

    node_labels = {}
    for sym_idx in range(n_symbols):
        tok_list = symbol_to_tok.get(sym_idx, [])
        if tok_list:
            most_common = max(set(tok_list), key=tok_list.count)
            node_labels[sym_idx] = f"{most_common}\n({len(tok_list)})"
        else:
            node_labels[sym_idx] = f"{sym_idx}"

    adj_probs = F.softmax(model.vq_layer.adjacency.detach().float(), dim=-1).cpu().numpy()
    G = nx.DiGraph()

    active_sorted = sorted(symbol_to_tok.keys(), key=lambda s: len(symbol_to_tok[s]), reverse=True)
    if n_symbols <= 64:
        nodes_to_plot = list(range(n_symbols))
    else:
        nodes_to_plot = active_sorted[:64] if active_sorted else list(range(min(64, n_symbols)))

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

    purity_cmap = LinearSegmentedColormap.from_list("purity_cmap", ["#fef0b8", "#8dd3c7", "#4393c3"])
    norm_pur = Normalize(vmin=0.0, vmax=100.0)
    fig, ax = plt.subplots(figsize=(14, 13))
    if len(G.nodes()) > 0:
        try:
            pos = nx.spring_layout(G, k=0.35, iterations=80, seed=42)
        except Exception:
            pos = nx.circular_layout(G)

        node_count = len(G.nodes())
        node_size = max(250, min(850, 30000 // max(1, node_count)))
        font_size = max(6, min(9, 450 // max(1, node_count)))
        node_colors = [
            purity_cmap(norm_pur(purity_map[i])) if i in symbol_to_tok else "#ffe5e5"
            for i in G.nodes()
        ]
        active_labels = {i: node_labels[i] for i in G.nodes() if i in node_set}

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, alpha=0.95, linewidths=0.8, edgecolors="#555555", ax=ax)
        nx.draw_networkx_labels(G, pos, labels=active_labels, font_size=font_size, font_weight="bold", ax=ax)

        for (u, v), w in zip(edges, weights):
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], width=max(0.6, w * 5.0), alpha=min(0.85, max(0.2, w * 2.0)),
                edge_color='gray', arrowstyle='->', arrowsize=12, ax=ax
            )

        sm = plt.cm.ScalarMappable(cmap=purity_cmap, norm=norm_pur)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.032, pad=0.02, label="Symbol Purity (% Top-Token Share)")

    ax.set_title(f"1_semantic_topology (Active in scan: {len(symbol_to_tok)}/{n_symbols}, Shown: {len(G.nodes())}, Colored by Purity)")
    plt.tight_layout()
    plt.savefig("1_semantic_topology.png", dpi=150)
    print("Saved 1_semantic_topology.png")
    plt.close()

    hidden, prev_sym = None, None
    seed_id = data_tensor[0].item()
    x = torch.tensor([[seed_id]], device=dev)
    stack_history, act_history = [], []
    push_history, pop_history, noop_history = [], [], []
    pred_ent_history = []
    latent_traj = []
    scan_tok_labels = []
    gen_ids = [seed_id]
    scan_len = max(60, min(120, int(round(180 / max(1.0, config["comp_ratio"])))))

    cached_cb = model.vq_layer.normalized_codebook()
    cached_padj = F.softmax(model.vq_layer.adjacency.float(), dim=-1)

    print("Running Inference Scan...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=use_amp):
        for _ in range(scan_len):
            in_tok_id = x.item()
            scan_tok_labels.append(tokenizer.format_token(tokenizer.id_to_token[in_tok_id]))

            (
                logits,
                hidden,
                prev_sym,
                _,
                _,
                _,
                avg_stack,
                _,
                _,
                _,
                avg_stk_ctrl,
                steps_vec,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = model(
                x, hidden, prev_sym, return_stats=True, cached_P_adj=cached_padj, cached_cb_norm=cached_cb
            )

            stack_history.append(avg_stack.item())
            act_history.append(steps_vec.item())
            ctrl_list = avg_stk_ctrl.detach().cpu().tolist()
            push_history.append(ctrl_list[0])
            pop_history.append(ctrl_list[1])
            noop_history.append(ctrl_list[2])

            probs = F.softmax(logits.squeeze(0).float(), dim=-1)
            pred_ent = -(probs * torch.log(probs + 1e-8)).sum().item()
            pred_ent_history.append(pred_ent)

            zr = hidden[0].detach().float().cpu().squeeze(0)
            zi = hidden[1].detach().float().cpu().squeeze(0)
            latent_traj.append(torch.cat([zr, zi], dim=-1))

            next_id = sample_next_token(
                logits, gen_ids, temperature=0.80, top_k=40, top_p=0.92, rep_penalty=1.10
            )
            gen_ids.append(next_id)
            x = torch.tensor([[next_id]], device=dev)

    act_counts = Counter(int(round(a)) for a in act_history)
    act_dist_str = ", ".join(
        f"{k} steps: {100.0 * v / len(act_history):.0f}%" for k, v in sorted(act_counts.items())
    )
    print(f"ACT Step Breakdown : {act_dist_str}")
    print(f"Stack Depth Range  : [{min(stack_history):.2f}, {max(stack_history):.2f}]")
    print(f"Generated          : {tokenizer.decode(gen_ids)}\n")

    x_idx = np.arange(len(stack_history))

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 6.8), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.0]})
    ax_top.plot(x_idx, stack_history, color='purple', linewidth=2.0, label='Stack Depth')
    ax_top.fill_between(x_idx, stack_history, color='purple', alpha=0.12)
    ax_top.set_title("2_stack_mri (Memory Depth & Per-Token Push/Pop/NoOp Control)")
    ax_top.set_ylabel("Expected Depth")
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(loc="upper left")

    ax_bot.plot(x_idx, push_history, color='seagreen', linewidth=1.7, label='Push')
    ax_bot.plot(x_idx, pop_history, color='crimson', linewidth=1.7, label='Pop')
    ax_bot.plot(x_idx, noop_history, color='slategray', linewidth=1.3, linestyle='--', alpha=0.85, label='NoOp')
    ax_bot.set_ylabel("Controller Prob")
    ax_bot.set_ylim(-0.05, 1.05)
    ax_bot.grid(True, alpha=0.25)
    ax_bot.legend(loc="upper right", ncol=3)
    ax_bot.set_xticks(x_idx)
    ax_bot.set_xticklabels(scan_tok_labels, rotation=55, ha='right', fontsize=7)
    plt.tight_layout()
    plt.savefig("2_stack_mri.png", dpi=150)
    print("Saved 2_stack_mri.png")
    plt.close()

    fig, ax_act = plt.subplots(figsize=(16, 4.8))
    ax_act.bar(x_idx, act_history, color='orange', alpha=0.85, label='ACT Steps')
    ax_act.set_ylim(0, model.max_recursion_depth + 0.5)
    ax_act.set_ylabel("Thinking Steps")
    ax_act.grid(True, axis='y', alpha=0.25)

    ax_ent = ax_act.twinx()
    ax_ent.plot(x_idx, pred_ent_history, color='navy', linewidth=1.6, marker='o', markersize=3, alpha=0.85, label='Predictive Entropy (nats)')
    ax_ent.set_ylabel("Predictive Entropy (nats)", color='navy')

    ax_act.set_xticks(x_idx)
    ax_act.set_xticklabels(scan_tok_labels, rotation=55, ha='right', fontsize=7)
    ax_act.set_title("3_act_profile (Thinking Steps vs. Predictive Uncertainty)")
    lines_1, labels_1 = ax_act.get_legend_handles_labels()
    lines_2, labels_2 = ax_ent.get_legend_handles_labels()
    ax_act.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig("3_act_profile.png", dpi=150)
    print("Saved 3_act_profile.png")
    plt.close()

    Z_traj = torch.stack(latent_traj, dim=0)
    T_pts = Z_traj.size(0)
    if T_pts >= 3 and Z_traj.size(1) >= 2:
        Z_c = Z_traj - Z_traj.mean(dim=0, keepdim=True)
        _, _, V_pca = torch.pca_lowrank(Z_c, q=2)
        pc = (Z_c @ V_pca[:, :2]).numpy()
    else:
        pc = Z_traj[:, :2].numpy()

    norms = Z_traj.norm(dim=-1).numpy()
    velocities = (Z_traj[1:] - Z_traj[:-1]).norm(dim=-1).numpy() if T_pts > 1 else np.zeros(1)

    fig, (ax_pca, ax_nv) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.2, 1.0]})
    ax_pca.plot(pc[:, 0], pc[:, 1], color='gray', linewidth=0.8, alpha=0.4, zorder=1)
    sc = ax_pca.scatter(pc[:, 0], pc[:, 1], c=np.arange(T_pts), cmap='plasma', s=45, alpha=0.88, zorder=2)
    ax_pca.scatter([pc[0, 0]], [pc[0, 1]], color='lime', edgecolor='black', s=95, zorder=4, label='Start')
    ax_pca.scatter([pc[-1, 0]], [pc[-1, 1]], color='cyan', edgecolor='black', s=95, zorder=4, label='End')
    for idx_t, lbl in enumerate(scan_tok_labels):
        if lbl in ("\\n", ":", ".", "!", "?") or idx_t % 10 == 0:
            ax_pca.annotate(lbl, (pc[idx_t, 0], pc[idx_t, 1]), fontsize=7, alpha=0.8, xytext=(3, 3), textcoords="offset points")
    fig.colorbar(sc, ax=ax_pca, label="Token Step")
    ax_pca.set_title(f"4_phase_plot (2D PCA of {Z_traj.size(1)}-D Complex Latent)")
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    ax_pca.grid(True, alpha=0.25)
    ax_pca.legend(loc="upper right", fontsize=8)

    ax_nv.plot(np.arange(T_pts), norms, color='teal', linewidth=1.8, label='State Norm ||z_t||')
    if T_pts > 1:
        ax_nv.plot(np.arange(1, T_pts), velocities, color='darkorange', linewidth=1.5, alpha=0.9, label='Step Velocity ||z_t - z_{t-1}||')
    ax_nv.set_title("Latent State Norm & Step-to-Step Velocity")
    ax_nv.set_xlabel("Token Step")
    ax_nv.set_ylabel("L2 Magnitude")
    ax_nv.grid(True, alpha=0.25)
    ax_nv.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig("4_phase_plot.png", dpi=150)
    print("Saved 4_phase_plot.png")
    plt.close()

    return sym_token_counts, sorted_counts


# ==========================================
# 9. Logic Rule Extraction (1-Hop + 2-Hop Abstractions)
# ==========================================
def extract_logic_rules(model, tokenizer, target_tensor, config, sym_token_counts=None):
    print("\n--- Extracting Explicit Logic Rules (1-Hop & 2-Hop Chains) ---")
    model.eval()
    dev = next(model.parameters()).device
    use_amp = _use_amp_for_model(model, config["batch_size"])

    rule_book_1 = defaultdict(list)
    rule_book_2 = defaultdict(list)
    hidden = None
    prev_sym = None
    prev_prev_sym = None
    diag_steps = min(len(target_tensor) - 1, config["diag_max_tokens"])
    min_rule_1 = max(2, diag_steps // 1000)
    min_rule_2 = max(2, diag_steps // 1500)

    local_counts = defaultdict(Counter) if sym_token_counts is None else sym_token_counts
    cached_cb = model.vq_layer.normalized_codebook()
    cached_padj = F.softmax(model.vq_layer.adjacency.float(), dim=-1)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=use_amp):
        for i in range(diag_steps):
            tok_id = target_tensor[i].item()
            x = torch.tensor([[tok_id]], device=dev)
            _, hidden, sym_idx, act_steps, _, _, _ = model(
                x, hidden, prev_sym, cached_P_adj=cached_padj, cached_cb_norm=cached_cb
            )
            if sym_token_counts is None:
                local_counts[sym_idx.item()][tok_id] += 1
            curr = sym_idx.item()
            steps_v = act_steps.item()
            if prev_sym is not None:
                src = prev_sym.item()
                rule_book_1[(src, curr)].append(steps_v)
                if prev_prev_sym is not None:
                    rule_book_2[(prev_prev_sym, src, curr)].append(steps_v)
                prev_prev_sym = src
            prev_sym = sym_idx

    def sym_tag(s):
        c = local_counts.get(s)
        if not c:
            return f"S_{s:<4}"
        top_tok = tokenizer.format_token(tokenizer.id_to_token[c.most_common(1)[0][0]])
        if len(top_tok) > 8:
            top_tok = top_tok[:7] + "…"
        return f"S_{s:<4}({top_tok})"

    print(f"\n[1-Hop Symbolic Rules] {'FROM':<16} | {'TO':<16} | {'COUNT':<6} | {'AVG STEPS':<10}")
    print("-" * 60)
    sorted_rules_1 = sorted(rule_book_1.items(), key=lambda x: len(x[1]), reverse=True)
    printed = 0
    for (src, dst), steps_list in sorted_rules_1:
        if len(steps_list) >= min_rule_1:
            print(f"{sym_tag(src):<16} -> {sym_tag(dst):<16} | {len(steps_list):<6} | {sum(steps_list)/len(steps_list):.2f}")
            printed += 1
            if printed >= 25:
                break

    print(f"\n[2-Hop Rule Chains]    {'S_A -> S_B -> S_C':<52} | {'COUNT':<6} | {'AVG STEPS':<10}")
    print("-" * 76)
    sorted_rules_2 = sorted(rule_book_2.items(), key=lambda x: len(x[1]), reverse=True)
    printed_2 = 0
    for (a, b, c), steps_list in sorted_rules_2:
        if len(steps_list) >= min_rule_2:
            chain_str = f"{sym_tag(a)} -> {sym_tag(b)} -> {sym_tag(c)}"
            print(f"{chain_str:<52} | {len(steps_list):<6} | {sum(steps_list)/len(steps_list):.2f}")
            printed_2 += 1
            if printed_2 >= 15:
                break


# ==========================================
# 10. Advanced Interaction, Probes, Split Ablation & Benchmarks
# ==========================================
def dream_mode(model, tokenizer, data_tensor, config, sym_token_counts=None):
    print("\n--- Dream Mode (Entropy-Targeted Symbolic Walk) ---")
    model.eval()
    dev = next(model.parameters()).device
    use_amp = _use_amp_for_model(model, config["batch_size"])

    if sym_token_counts is None:
        _, sym_token_counts = build_symbol_token_table(model, tokenizer, data_tensor, config)

    adj_logits = model.vq_layer.adjacency.detach().float()
    seed_id = data_tensor[0].item()
    x = torch.tensor([[seed_id]], device=dev)
    cached_cb = model.vq_layer.normalized_codebook()
    cached_padj = F.softmax(model.vq_layer.adjacency.float(), dim=-1)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=use_amp):
        logits, hidden, prev_sym, _, _, _, _ = model(
            x, None, None, cached_P_adj=cached_padj, cached_cb_norm=cached_cb
        )

    curr_sym = prev_sym.item()
    output_ids = [seed_id]

    target_branch_ppx = min(4.0, max(2.0, 0.04 * model.n_symbols))
    dream_temps = []
    dream_ppxs = []
    emitted_from_table = 0

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

        lm_probs = F.softmax(logits.squeeze(0).float(), dim=-1).cpu().numpy()
        tok_counter = sym_token_counts.get(next_sym)
        if tok_counter:
            cand_ids = list(tok_counter.keys())
            cand_counts = np.array([tok_counter[tid] for tid in cand_ids], dtype=np.float64)
            cand_emit = cand_counts / cand_counts.sum()
            cand_lm = np.maximum(lm_probs[cand_ids], 1e-9) ** 0.35
            combined = cand_emit * cand_lm
            combined = combined / combined.sum()
            tok_idx = int(np.random.choice(cand_ids, p=combined))
            emitted_from_table += 1
        else:
            tok_idx = int(np.argmax(lm_probs))

        output_ids.append(tok_idx)
        curr_sym = next_sym

        x_next = torch.tensor([[tok_idx]], device=dev)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=use_amp):
            logits, hidden, _, _, _, _, _ = model(
                x_next, hidden, torch.tensor([curr_sym], device=dev, dtype=torch.long),
                cached_P_adj=cached_padj, cached_cb_norm=cached_cb
            )

    print(
        f"Dream Stats  : AvgTemp={sum(dream_temps)/len(dream_temps):.2f} | "
        f"AvgBranchPPX={sum(dream_ppxs)/len(dream_ppxs):.2f} (Target={target_branch_ppx:.2f}) | "
        f"GroundedSyms={emitted_from_table}/60"
    )
    print(f"Dream Output : {tokenizer.decode(output_ids)}\n")


def anomaly_detector(model, tokenizer, val_text_str, text_data, config):
    print("\n--- Anomaly Detection Test (With Warmup Context) ---")
    model.eval()
    dev = next(model.parameters()).device
    use_amp = _use_amp_for_model(model, config["batch_size"])

    if val_text_str is not None and len(val_text_str) > 200:
        mid = len(val_text_str) // 4
        snippet = val_text_str[mid : mid + 500]
        words = snippet.split()
        if len(words) > 2:
            words = words[1:]
    else:
        words = text_data[:500].split()

    n_w = len(words)
    n_warm = min(4, max(0, n_w - 4)) if n_w >= 8 else min(2, max(0, n_w - 2))
    warmup_words = words[:n_warm]
    prefix_words = words[n_warm : min(n_w, n_warm + 7)]

    warmup_str = " ".join(warmup_words)
    eval_prefix_str = " ".join(prefix_words)
    full_prefix_str = " ".join(warmup_words + prefix_words)

    anomaly_word = "banana" if "banana" not in text_data.lower() else "xyzzy"
    corrupt_text = f"{full_prefix_str} {anomaly_word}" if full_prefix_str else anomaly_word
    print(f"Warmup: '{warmup_str}' | Test: '{eval_prefix_str} {anomaly_word}'")

    input_ids = tokenizer.encode(corrupt_text)
    if len(input_ids) < 3:
        print("Input too short for anomaly scan.")
        return

    w_len = min(len(input_ids) - 2, len(tokenizer.encode(warmup_str))) if warmup_str else 0
    eval_start = max(1, w_len)
    p_len = min(len(input_ids) - 1, max(eval_start + 1, len(tokenizer.encode(full_prefix_str))))

    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=dev)
    hidden, prev_sym, prev_logits = None, None, None
    lm_surprisals = []
    eth_violations = []
    act_trace = []

    cached_cb = model.vq_layer.normalized_codebook()
    cached_padj = F.softmax(model.vq_layer.adjacency.float(), dim=-1)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=use_amp):
        for i in range(len(input_tensor)):
            x = input_tensor[i].view(1, 1)
            if i > 0 and prev_logits is not None:
                lm_nll = F.cross_entropy(prev_logits.float(), input_tensor[i].view(1)).item()
            else:
                lm_nll = None

            logits, hidden, prev_sym, act_steps, _, eth_loss, _ = model(
                x, hidden, prev_sym, cached_P_adj=cached_padj, cached_cb_norm=cached_cb
            )
            prev_logits = logits

            if i >= eval_start and lm_nll is not None:
                eth_val = eth_loss.item()
                lm_surprisals.append(lm_nll)
                eth_violations.append(eth_val)
                act_trace.append(act_steps.item())

    eth_weight_vis = 2.0
    scaled_eth = [eth_weight_vis * v for v in eth_violations]
    combined_scores = [lm + se for lm, se in zip(lm_surprisals, scaled_eth)]

    split_idx = max(1, min(len(combined_scores) - 1, p_len - eval_start))
    if len(combined_scores) >= 2:
        prefix_mean = float(np.mean(combined_scores[:split_idx]))
        tail_vals = combined_scores[split_idx:]
        tail_mean = float(np.mean(tail_vals))
        tail_max = float(np.max(tail_vals))
        ratio = tail_max / max(1e-6, prefix_mean)

        lm_pre = float(np.mean(lm_surprisals[:split_idx]))
        lm_tail_max = float(np.max(lm_surprisals[split_idx:]))
        eth_pre = float(np.mean(eth_violations[:split_idx]))
        eth_tail_max = float(np.max(eth_violations[split_idx:]))
        act_pre = float(np.mean(act_trace[:split_idx]))
        act_tail = float(np.mean(act_trace[split_idx:]))

        print(
            f"Anomaly Stats: ValidPrefixMean={prefix_mean:.2f} | "
            f"TailMean={tail_mean:.2f} TailMax={tail_max:.2f} (Contrast={ratio:.2f}x) | "
            f"[LM: {lm_pre:.2f}->{lm_tail_max:.2f} ({lm_tail_max/max(1e-6,lm_pre):.2f}x) | "
            f"SymEth: {eth_pre:.2f}->{eth_tail_max:.2f} ({eth_tail_max/max(1e-6,eth_pre):.2f}x) | "
            f"ACT: {act_pre:.2f}->{act_tail:.2f}]"
        )

    x_labels = [tokenizer.format_token(tokenizer.id_to_token[idx]) for idx in input_ids[eval_start:]]
    x_idx = np.arange(len(combined_scores))

    fig, (ax_v, ax_a) = plt.subplots(2, 1, figsize=(13, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    ax_v.plot(x_idx, combined_scores, color='skyblue', linestyle='-', linewidth=2, label=f'Combined Violation (LM + {eth_weight_vis:.1f}*Eth)')
    ax_v.plot(x_idx, lm_surprisals, color='orange', linestyle='--', linewidth=1.3, alpha=0.85, label='LM Token Surprisal (NLL)')
    ax_v.plot(x_idx, scaled_eth, color='purple', linestyle='-.', linewidth=1.3, alpha=0.85, label=f'Symbolic Topology ({eth_weight_vis:.1f}*Eth)')
    ax_v.scatter(x_idx, combined_scores, color='crimson', s=50, zorder=5, label='Anomaly Score')

    ax_a.bar(x_idx, act_trace, color='orange', alpha=0.80, label='ACT Thinking Steps')
    ax_a.set_ylim(0, model.max_recursion_depth + 0.5)
    ax_a.set_ylabel("ACT Steps")
    ax_a.grid(True, alpha=0.25)

    if len(combined_scores) >= 2:
        ax_v.axvline(split_idx - 0.5, color='gray', linestyle=':', linewidth=1.5, label='Anomaly Boundary')
        ax_a.axvline(split_idx - 0.5, color='gray', linestyle=':', linewidth=1.5)

    ax_a.set_xticks(x_idx)
    ax_a.set_xticklabels(x_labels, rotation=35, ha='right', fontsize=8)
    ax_v.set_title("Topological, Predictive & ACT Violation Score (Warmed-Up Anomaly Detection)")
    ax_v.set_ylabel("Violation Magnitude")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend(loc="upper left", fontsize=8, ncol=2)
    ax_a.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig("5_anomaly_detection.png", dpi=150)
    print("Saved 5_anomaly_detection.png")
    plt.close()


def context_difficulty_probe(model, data_tensor, val_tensor, config):
    """Checks whether ACT allocates more recursion steps on high-difficulty predictions on Val."""
    target = val_tensor if val_tensor is not None else data_tensor
    if len(target) < 64:
        return
    print("\n--- Context-Difficulty ACT Probe ---")
    model.eval()
    dev = next(model.parameters()).device
    use_amp = _use_amp_for_model(model, config["batch_size"])

    scan_n = min(len(target) - 1, 1024)
    hidden, prev_sym, prev_logits = None, None, None

    cached_cb = model.vq_layer.normalized_codebook()
    cached_padj = F.softmax(model.vq_layer.adjacency.float(), dim=-1)

    next_nlls, pred_ents, in_nlls, steps_aligned, steps_in = [], [], [], [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=use_amp):
        for i in range(scan_n):
            x = target[i].view(1, 1).to(dev)
            y_next = target[i + 1].view(1).to(dev)
            in_nll = (
                F.cross_entropy(prev_logits.float(), target[i].view(1).to(dev)).item()
                if (i > 0 and prev_logits is not None)
                else None
            )

            logits, hidden, prev_sym, act_steps, _, _, _ = model(
                x, hidden, prev_sym, cached_P_adj=cached_padj, cached_cb_norm=cached_cb
            )
            st = act_steps.item()
            next_nll = F.cross_entropy(logits.float(), y_next).item()
            probs = F.softmax(logits.float(), dim=-1)
            ent = -(probs * torch.log(probs + 1e-8)).sum().item()

            next_nlls.append(next_nll)
            pred_ents.append(ent)
            steps_aligned.append(st)
            if in_nll is not None:
                in_nlls.append(in_nll)
                steps_in.append(st)

            prev_logits = logits

    if next_nlls:
        med_next = float(np.median(next_nlls))
        easy_next = [st for nll, st in zip(next_nlls, steps_aligned) if nll <= med_next]
        hard_next = [st for nll, st in zip(next_nlls, steps_aligned) if nll > med_next]
        print(
            f"Val ACT (Aligned Target NLL) : Easy(NLL<={med_next:.2f})={np.mean(easy_next):.3f} steps | "
            f"Hard(NLL>{med_next:.2f})={np.mean(hard_next):.3f} steps"
        )

        med_ent = float(np.median(pred_ents))
        low_ent = [st for h, st in zip(pred_ents, steps_aligned) if h <= med_ent]
        high_ent = [st for h, st in zip(pred_ents, steps_aligned) if h > med_ent]
        print(
            f"Val ACT (Model Entropy)      : LowEnt(H<={med_ent:.2f})={np.mean(low_ent):.3f} steps | "
            f"HighEnt(H>{med_ent:.2f})={np.mean(high_ent):.3f} steps"
        )

    if in_nlls:
        med_in = float(np.median(in_nlls))
        easy_in = [st for nll, st in zip(in_nlls, steps_in) if nll <= med_in]
        hard_in = [st for nll, st in zip(in_nlls, steps_in) if nll > med_in]
        print(
            f"Val ACT (Input Surprisal)    : EasyIn(NLL<={med_in:.2f})={np.mean(easy_in):.3f} steps | "
            f"HardIn(NLL>{med_in:.2f})={np.mean(hard_in):.3f} steps"
        )


def ablation_probe(model, data_tensor, val_tensor):
    """Quantifies each cognitive subsystem's contribution (with masked GWT streams)."""
    dev = next(model.parameters()).device
    eval_batches = build_val_batches(val_tensor, device=dev)
    split_tag = "Val Set"
    if not eval_batches:
        eval_batches = build_val_batches(data_tensor[:min(len(data_tensor), 4096)], device=dev)
        split_tag = "TrainSlice"
    if not eval_batches:
        return []

    print(f"\n--- Split Module Ablation Probe ({split_tag}) ---")
    configs = [
        ("Full Model", {}),
        ("- No Stack (Both)", {"ablate_stack": True}),
        ("- No Stack Attn", {"ablate_stack_attn": True}),
        ("- No Stack Read", {"ablate_stack_read": True}),
        ("- No HoloMem", {"ablate_holo": True}),
        ("- No GraphEcho", {"ablate_graph": True}),
        ("- No VQ Feedback", {"ablate_vq": True}),
        ("- No Rec Carry", {"ablate_carry": True}),
        ("- Fixed ACT (1x)", {"ablate_act_1step": True}),
    ]
    results = []
    base_loss = None
    print(f"{'Configuration':<20} | {'Loss':<7} | {'PPX':<7} | {'Acc':<6} | {'dLoss vs Full':<13}")
    print("-" * 64)

    for label, kwargs in configs:
        loss_v, acc_v = evaluate_val(model, eval_batches, **kwargs)
        if loss_v is None:
            continue
        ppx_v = float(math.exp(min(loss_v, 20.0)))
        if base_loss is None:
            base_loss = loss_v
        delta = loss_v - base_loss
        print(f"{label:<20} | {loss_v:<7.4f} | {ppx_v:<7.2f} | {acc_v:5.1f}% | {delta:+10.4f}")
        results.append((label, loss_v, ppx_v, acc_v, delta))

    return results


def plot_codebook_and_ablation(sorted_counts, ablation_results):
    fig, (ax_zipf, ax_abl) = plt.subplots(1, 2, figsize=(14.5, 5.8))

    ranks = np.arange(1, len(sorted_counts) + 1)
    safe_counts = np.maximum(sorted_counts, 0)
    nz_mask = safe_counts > 0
    if nz_mask.any():
        ax_zipf.semilogy(ranks[nz_mask], safe_counts[nz_mask], color='royalblue', linewidth=2.0, label='Active Symbol Hits')
    if (~nz_mask).any():
        first_zero = int(np.argmax(~nz_mask)) + 1
        ax_zipf.axvline(first_zero, color='crimson', linestyle='--', linewidth=1.2, label=f'Dead Tail Start (Rank {first_zero})')
    total_hits = int(safe_counts.sum())
    if total_hits > 0:
        ideal_uniform = total_hits / float(max(1, len(sorted_counts)))
        ax_zipf.axhline(ideal_uniform, color='seagreen', linestyle=':', linewidth=1.4, label=f'Uniform Mean ({ideal_uniform:.1f})')
    ax_zipf.set_title("Codebook Utilization (Zipf Rank-Frequency)")
    ax_zipf.set_xlabel("Codebook Symbol Rank")
    ax_zipf.set_ylabel("Hit Count (log scale)")
    ax_zipf.grid(True, alpha=0.25)
    ax_zipf.legend(loc="upper right", fontsize=8)

    if len(ablation_results) > 1:
        sub = ablation_results[1:]
        labels = [r[0] for r in sub]
        deltas = [r[4] for r in sub]
        colors = ['crimson' if d > 0.005 else ('seagreen' if d < -0.002 else 'slategray') for d in deltas]
        y_pos = np.arange(len(labels))
        bars = ax_abl.barh(y_pos, deltas, color=colors, alpha=0.85)
        ax_abl.axvline(0.0, color='black', linewidth=1.0)
        ax_abl.set_yticks(y_pos)
        ax_abl.set_yticklabels(labels, fontsize=9)
        ax_abl.invert_yaxis()
        max_abs = max(0.05, max(abs(d) for d in deltas))
        ax_abl.set_xlim(min(0.0, min(deltas) * 1.15), max_abs * 1.18)
        ax_abl.set_xlabel("Delta Val Loss vs. Full Model (+ = module helps)")
        ax_abl.set_title("Split Module Ablation Impact (Val Loss Penalty)")
        ax_abl.grid(True, axis='x', alpha=0.25)
        for bar, d in zip(bars, deltas):
            ax_abl.text(
                bar.get_width() + (0.004 * max_abs if d >= 0 else -0.004 * max_abs),
                bar.get_y() + bar.get_height() / 2.0,
                f"{d:+.4f}",
                va='center',
                ha='left' if d >= 0 else 'right',
                fontsize=8,
            )
    else:
        ax_abl.text(0.5, 0.5, "No ablation data", ha="center", va="center")
        ax_abl.set_title("Split Module Ablation Impact")

    plt.tight_layout()
    plt.savefig("6_codebook_and_ablation.png", dpi=150)
    print("Saved 6_codebook_and_ablation.png")
    plt.close()


def generate_pddl_blocksworld_episodes(n_episodes=60, steps_per_ep=6, seed=SEED):
    """Generates valid deterministic 3-block Blocksworld state-action transitions."""
    rng = random.Random(seed)
    lines = []
    for _ in range(n_episodes):
        loc = {"A": "table", "B": "table", "C": "table"}
        holding = "none"
        for _ in range(steps_per_ep):
            clear = {b: (loc[b] != "hand" and b not in loc.values()) for b in ("A", "B", "C")}
            valid_actions = []
            if holding == "none":
                for b in ("A", "B", "C"):
                    if clear[b]:
                        if loc[b] == "table":
                            valid_actions.append(("pick", b, "table"))
                        else:
                            valid_actions.append(("unstack", b, loc[b]))
            else:
                valid_actions.append(("putdown", holding, "table"))
                for dst in ("A", "B", "C"):
                    if dst != holding and clear[dst]:
                        valid_actions.append(("stack", holding, dst))

            act, obj, src_dst = rng.choice(valid_actions)
            s_before = f"S A:{loc['A']} B:{loc['B']} C:{loc['C']} H:{holding}"
            if act == "pick":
                loc[obj] = "hand"
                holding = obj
                a_str = f"ACT pick {obj}"
            elif act == "unstack":
                loc[obj] = "hand"
                holding = obj
                a_str = f"ACT unstack {obj} from {src_dst}"
            elif act == "putdown":
                loc[obj] = "table"
                holding = "none"
                a_str = f"ACT putdown {obj}"
            else:
                loc[obj] = src_dst
                holding = "none"
                a_str = f"ACT stack {obj} on {src_dst}"
            s_after = f"NEXT A:{loc['A']} B:{loc['B']} C:{loc['C']} H:{holding}"
            lines.append(f"{s_before} | {a_str} -> {s_after}")
    return "\n".join(lines)


def run_pddl_probe(epochs=90):
    """Standalone Blocksworld/PDDL rule-inference sanity benchmark (zero global config mutation)."""
    print("\n=== Running Optional Blocksworld/PDDL Symbolic Rule Sanity Benchmark ===")
    pddl_text = generate_pddl_blocksworld_episodes(n_episodes=80, steps_per_ep=6)
    tokens = pddl_text.replace("\n", " \n ").split(" ")
    tokens = [t for t in tokens if t != ""]
    vocab = sorted(set(tokens))
    tok_to_id = {t: i for i, t in enumerate(vocab)}
    ids = torch.tensor([tok_to_id[t] for t in tokens], dtype=torch.long)

    probe_cfg = copy.deepcopy(DEFAULT_CONFIG)
    probe_cfg["n_symbols"] = 64
    probe_cfg["stack_size"] = 12
    probe_cfg["max_recursion_depth"] = 4
    probe_cfg["embedding_dim"] = 64
    probe_cfg["act_threshold"] = 0.85

    probe_model = UberCRSN(len(vocab), 64, config=probe_cfg).to(DEVICE)
    probe_model.sym_alpha.fill_(0.30)
    probe_model.graph_gate.fill_(1.0)
    opt = torch.optim.AdamW(probe_model.parameters(), lr=2e-3, weight_decay=1e-5)

    split = int(0.85 * len(ids))
    tr_ids = ids[:split].to(DEVICE)
    va_ids = ids[split:].to(DEVICE)
    seq = 16
    bs = max(1, (len(tr_ids) - 1) // seq)
    streams = torch.stack([tr_ids[b * seq : b * seq + seq + 1] for b in range(bs)], dim=0)

    for _ in range(epochs):
        probe_model.train()
        hidden, prev_sym = None, None
        loss_acc = 0.0
        for t in range(seq):
            x = streams[:, t].unsqueeze(1)
            y = streams[:, t + 1]
            logits, hidden, prev_sym, ponder_l, vq_l, eth_l, _, _, _, _, _, _, _, _, _, cons_l, _, extra = probe_model(
                x, hidden, prev_sym, return_stats=True
            )
            step_logits_t = extra["step_logits"]
            step_p_t = extra["step_p"]
            n_run_steps = step_logits_t.size(1)
            step_ce = F.cross_entropy(
                step_logits_t.float().reshape(-1, len(vocab)),
                y.unsqueeze(1).expand(-1, n_run_steps).reshape(-1),
                reduction="none",
            ).view(bs, n_run_steps)
            pred_loss = (step_p_t * step_ce).sum(dim=-1).mean()
            loss = pred_loss + 0.03 * ponder_l + 0.5 * vq_l + 0.1 * eth_l + 0.02 * cons_l
            loss_acc = loss_acc + loss
        opt.zero_grad(set_to_none=True)
        (loss_acc / seq).backward()
        torch.nn.utils.clip_grad_norm_(probe_model.parameters(), 1.5)
        opt.step()
        probe_model.vq_layer.normalize_codebook_()

    probe_model.eval()
    hidden, prev_sym = None, None
    correct, total = 0, 0
    act_on_next, act_on_other = [], []
    with torch.no_grad():
        for i in range(len(va_ids) - 1):
            x = va_ids[i].view(1, 1)
            y = va_ids[i + 1].item()
            logits, hidden, prev_sym, act_steps, _, _, _ = probe_model(x, hidden, prev_sym)
            correct += int(logits.argmax(dim=-1).item() == y)
            total += 1
            if vocab[va_ids[i].item()] in ("->", "NEXT", "ACT"):
                act_on_next.append(act_steps.item())
            else:
                act_on_other.append(act_steps.item())

    acc = 100.0 * correct / max(1, total)
    print(
        f"PDDL Probe Val Accuracy: {acc:.1f}% ({correct}/{total}) | "
        f"ACT(->/ACT/NEXT)={np.mean(act_on_next) if act_on_next else 0:.2f} vs "
        f"ACT(StateTok)={np.mean(act_on_other) if act_on_other else 0:.2f}"
    )


# ==========================================
# 11. Main Execution (Guarded Against Import-Time Side Effects)
# ==========================================
def main():
    logger = setup_terminal_logger(TERMINAL_LOG_FILE)
    try:
        set_global_seed(SEED)
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
        if "onedrive" in os.path.abspath(".").lower():
            print("[Notice] Working directory is inside OneDrive; frequent checkpoint saves can cause file-lock/sync slowdowns.")

        text_data = load_text_corpus(DATA_FILE, verbose=True)
        tokenizer, vocab_size, data_tensor, val_tensor, val_text_str, config = auto_configure_from_text(
            text_data, DEFAULT_CONFIG
        )

        filename = "crsn_omni_model.pth"
        trained_model, train_history = train(
            vocab_size=vocab_size,
            tokenizer=tokenizer,
            data_tensor=data_tensor,
            val_tensor=val_tensor,
            config=config,
            save_path=filename,
        )
        print(f"\nCheckpoint ready at '{filename}'.")

        plot_training_dynamics(train_history, config)
        sym_token_counts, sym_zipf_counts = visualize_all(trained_model, tokenizer, data_tensor, config)
        extract_logic_rules(trained_model, tokenizer, data_tensor, config, sym_token_counts=sym_token_counts)

        dream_mode(trained_model, tokenizer, data_tensor, config, sym_token_counts=sym_token_counts)
        anomaly_detector(trained_model, tokenizer, val_text_str, text_data, config)
        context_difficulty_probe(trained_model, data_tensor, val_tensor, config)
        ablation_results = ablation_probe(trained_model, data_tensor, val_tensor)
        plot_codebook_and_ablation(sym_zipf_counts, ablation_results)

        if config.get("run_pddl_probe", False):
            run_pddl_probe()

        print(f"\nSaved complete terminal output log to '{TERMINAL_LOG_FILE}'.")
    finally:
        logger.close()

    try:
        from google.colab import files
        files.download("crsn_omni_model.pth")
        files.download(TERMINAL_LOG_FILE)
        files.download("0_training_dynamics.png")
        files.download("1_semantic_topology.png")
        files.download("2_stack_mri.png")
        files.download("3_act_profile.png")
        files.download("4_phase_plot.png")
        files.download("5_anomaly_detection.png")
        files.download("6_codebook_and_ablation.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
