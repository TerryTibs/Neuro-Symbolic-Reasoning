# ============================================================
# SACRSN v60 (Dual-Clock Polar + Tuned Macro-Packets + Regularized Complex ACT/Stack)
# Unified Architecture Upgrades:
#   1. Dual-Clock Polar Tokenization:
#        - Content Lexeme Stem + Polar Morphological Phase Rotation (case/prefix/suffix)
#        - Structural Boundary Edge Stream routed directly to RelOp & Stack (not VQ concepts)
#        - 100% lossless round-trip encode/decode + Exact Bits-Per-Character (BPC) evaluation
#   2. Self-Evolving Holographic Macro-Packets (toggleable via CONFIG["use_macro_packets"]):
#        - Phases 1-2 stabilize atomic L=1 Dual-Clock Polar tokens
#        - Phase 3 mines mastered idioms (low ACT + NoOp + high RelTrans certainty)
#        - Superposes L=1..4 word packets via complex phase keys in fixed (B, T, 4) tensors
#        - Unbinds conjugate slots causally via shared stem/morph/edge decoders (zero vocab resize)
#   3. v60 Upgrades: 5x Faster Eval Gating, Complex/Label-Smooth Regularization,
#      Deeper Macro-Packet Mining, Stronger Pop-Clear Stack Balance, Clean Edge Sampling
# ============================================================

import os
import sys
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
# 0. Hardware Setup & Automatic Terminal Logger
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
    """Mirrors all stdout/stderr terminal printouts to a text file."""
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


SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
VRAM_GB = (
    torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if DEVICE == "cuda"
    else 0.0
)
USE_BF16 = (DEVICE == "cuda") and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32
AMP_DEVICE = "cuda" if DEVICE == "cuda" else "cpu"
HIGH_VRAM = (DEVICE == "cuda") and (VRAM_GB >= 12.0)


def set_reproducibility(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# ==========================================
# 1. Configuration
# ==========================================
DATA_FILE = "data.txt"
TERMINAL_LOG_FILE = "terminal outputs.txt"

DEFAULT_TEXT_DATA = """True, without falsehood, certain and most true. 
That which is above is like to that which is below, 
and that which is below is like to that which is above.
The father of all perfection in the whole world is here.
Its force or power is entire if it be converted into earth."""

CONFIG = {
    # Tokenizer & Macro-Packet Toggles
    "use_dual_clock_polar": True,
    "use_macro_packets": True,      # Toggle False for clean A/B ablation of Dual-Clock Polar alone
    "max_pkt_len": 4,
    "max_macro_phrases": 256,
    "macro_scan_cap": 262144,
    "macro_min_freq": 6,
    "macro_max_act": 2.40,
    "macro_min_noop": 0.30,
    "macro_min_rel_prob": 0.08,
    "max_stems": 4096 if HIGH_VRAM else (2560 if DEVICE == "cuda" else 1024),
    "max_morphs": 96,
    "max_edges": 96,

    # Auto-Sizing Controls
    "auto_size": True,
    "min_symbols": 16,
    "max_symbols": 1024 if HIGH_VRAM else (768 if DEVICE == "cuda" else 512),
    "max_dim": 288 if HIGH_VRAM else (192 if DEVICE == "cuda" else 128),
    "epoch_token_cap": 32768 if HIGH_VRAM else 4096,
    "diag_max_tokens": 2000,
    "val_split_min_chars": 12000,

    # Sequence & Model Capacity (auto-adjusted when auto_size=True)
    "batch_size": 4,
    "stream_len": None,
    "seq_len": 16,
    "embedding_dim": 64,
    "n_symbols": 120,
    "n_ops": 16,
    "stack_size": 16,
    "max_recursion_depth": 4,
    "decoder_dropout": 0.15,
    "cell_dropout": 0.10,
    "emb_dropout": 0.08,
    "label_smoothing": 0.05,

    # Reasoning (PonderNet ACT) & Memory Controls
    "act_threshold": 0.85,
    "ponder_prior_halt": 0.40,
    "ponder_penalty": 0.030,
    "ponder_min_penalty": 0.008,
    "use_stack": True,
    "stack_temp": 0.75,
    "stack_clear_decay": 0.80,
    "stack_pop_balance_mult": 3.0,
    "stack_balance_weight": 0.08,
    "holo_decay": 0.99,
    "holo_write_scale": 0.10,

    # Topology, Compositional Symbols, Consistency & Stability
    "commitment_cost": 0.25,
    "vq_weight": 1.0,
    "op_vq_weight": 0.60,
    "sym_lm_weight": 0.22,
    "graph_bias_scale": 0.1,
    "symbol_consistency_weight": 0.02,
    "ethical_weight": 0.14,
    "diversity_weight": 1.2,
    "op_diversity_weight": 0.50,
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
    "eps": 1e-5,
    "comp_ratio": 2.5,

    # Optional Blocksworld/PDDL Symbolic Transition Benchmark
    "run_pddl_probe": False,
}

# Global runtime handles initialized explicitly in initialize_runtime()
TEXT_DATA = DEFAULT_TEXT_DATA
tokenizer = None
vocab_size = 0
n_morphs = 0
n_edges = 0
data_atomic = None          # Dict of atomic tensors (stems, morphs, edges, chars)
val_atomic_slices = []      # List of atomic dicts per contiguous validation slice
val_text_str = None
USE_AMP = False


# ==========================================
# 2. Dual-Clock Polar Tokenizer, Macro-Packet Engine & Auto-Configurator
# ==========================================
def load_corpus_text(data_file=DATA_FILE):
    text = DEFAULT_TEXT_DATA
    if data_file and os.path.isfile(data_file):
        try:
            with open(data_file, "r", encoding="utf-8-sig", errors="replace") as f:
                loaded_text = f.read()
            if len(loaded_text.strip()) >= 8:
                text = loaded_text
                print(f"Loaded corpus from '{data_file}' ({len(text):,} chars)")
            else:
                print(f"'{data_file}' is empty/too short — using built-in default text.")
        except OSError as e:
            print(f"Could not read '{data_file}' ({e}) — using built-in default text.")
    else:
        print(f"'{data_file}' not found — using built-in default text.")
    return text


class DualClockPolarTokenizer:
    """
    Lossless Dual-Clock + Polar-Factored Tokenizer:
      - Factors every word into (stem_id, morph_id) where morph_id = (case, prefix, suffix, drop_e)
      - Keeps structural boundaries & punctuation in edge_id (not competing with VQ concepts)
      - Uses <CONT> edges for rare OOV subword splits so decode(encode(text)) is lossless.
    """
    PREFIXES = ("", "un", "re", "dis", "mis", "over", "under", "out", "in", "im")
    SUFFIX_RULES = (
        ("", False),
        ("s", False),
        ("es", False),
        ("ed", False),
        ("d", False),
        ("ing", False),
        ("ing", True),
        ("ed", True),
        ("er", False),
        ("er", True),
        ("est", False),
        ("est", True),
        ("eth", False),
        ("eth", True),
        ("ly", False),
        ("ness", False),
        ("ful", False),
        ("less", False),
        ("ment", False),
        ("'s", False),
        ("'d", False),
        ("'ll", False),
        ("'st", False),
        ("'t", False),
        ("st", False),
    )
    COMMON_BIGRAMS = (
        "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd",
        "ti", "es", "or", "te", "of", "ed", "is", "it", "al", "ar",
        "st", "to", "nt", "ng", "se", "ha", "as", "ou", "io", "le",
        "ve", "co", "me", "de", "hi", "ri", "ro", "ic", "ne", "ea",
    )

    def __init__(self, text, max_stems=2048, max_morphs=96, max_edges=96, min_char_freq=1):
        # Catch-all "|." ensures '_' and non-ASCII word chars are preserved via edge stream rather than dropped.
        self.chunk_pat = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]|\s+|.")
        self.word_pat = re.compile(r"^(?:[A-Za-z]+(?:'[A-Za-z]+)?|\d+)$")
        self._word_cache = {}

        # Special Stem IDs
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.sep_token = "<SEP>"
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.sep_id = 3

        # Special Morph IDs: 0=<PAD>, 1=("lo", "", "", False)
        self.morph_pad = ("<PAD>", "", "", False)
        self.morph_plain = ("lo", "", "", False)

        # Special Edge IDs: 0=<PAD>, 1="<CONT>", 2="<EDGE_UNK>", 3=""
        self.edge_pad = "<PAD>"
        self.edge_cont = "<CONT>"
        self.edge_unk = "<EDGE_UNK>"
        self.edge_empty = ""
        self.edge_pad_id = 0

        pieces = self.chunk_pat.findall(text)
        raw_word_counts = Counter()
        raw_sep_counts = Counter()
        word_chars = Counter()
        sep_chars = Counter()

        i = 0
        while i < len(pieces):
            p = pieces[i]
            if self.word_pat.match(p):
                for sub_p in self._split_mixed_case(p):
                    raw_word_counts[sub_p.lower()] += 1
                    for ch in sub_p.lower():
                        word_chars[ch] += 1
                i += 1
            else:
                sep_buf = []
                while i < len(pieces) and not self.word_pat.match(pieces[i]):
                    sep_buf.append(pieces[i])
                    for ch in pieces[i]:
                        sep_chars[ch] += 1
                    i += 1
                sep_str = "".join(sep_buf)
                raw_sep_counts[sep_str] += 1
                for sub_part in sep_buf:
                    raw_sep_counts[sub_part] += 1

        # Build Edge Vocabulary (all valid single sep chars by freq priority + frequent multi-char seams)
        self.edge_vocab = [self.edge_pad, self.edge_cont, self.edge_unk, self.edge_empty]
        edge_set = set(self.edge_vocab)
        avail_sep = max(0, max_edges - len(self.edge_vocab))
        base_sep_chars = sorted(
            ch for ch, _ in [
                (c, f) for c, f in sep_chars.most_common() if f >= min_char_freq
            ][:avail_sep]
        )
        for ch in base_sep_chars:
            if ch not in edge_set:
                self.edge_vocab.append(ch)
                edge_set.add(ch)
        for sep_s, _ in raw_sep_counts.most_common():
            if len(self.edge_vocab) >= max_edges:
                break
            if sep_s not in edge_set and len(sep_s) <= 4:
                self.edge_vocab.append(sep_s)
                edge_set.add(sep_s)

        self.edge_to_id = {e: idx for idx, e in enumerate(self.edge_vocab)}
        self.id_to_edge = {idx: e for idx, e in enumerate(self.edge_vocab)}
        self.edge_pad_id = self.edge_to_id[self.edge_pad]
        self.edge_cont_id = self.edge_to_id[self.edge_cont]
        self.edge_unk_id = self.edge_to_id[self.edge_unk]
        self.edge_empty_id = self.edge_to_id[self.edge_empty]
        self.n_edges = len(self.edge_vocab)
        self.sentence_end_edges = {
            idx for idx, e in self.id_to_edge.items()
            if any(ch in e for ch in (".", "?", "!", "\n", ":"))
        }

        # Build Stem Vocabulary (specials + valid chars + frequent bigrams + top stems)
        self.stem_vocab = [self.pad_token, self.unk_token, self.bos_token, self.sep_token]
        stem_set = set(self.stem_vocab)
        avail_wchars = max(0, max_stems - len(self.stem_vocab))
        base_word_chars = sorted(
            ch for ch, _ in [
                (c, f) for c, f in word_chars.most_common() if f >= min_char_freq
            ][:avail_wchars]
        )
        if not base_word_chars:
            base_word_chars = sorted(list(word_chars.keys()))[:avail_wchars]
        self.valid_word_chars = set(base_word_chars)

        for ch in base_word_chars:
            if ch not in stem_set:
                self.stem_vocab.append(ch)
                stem_set.add(ch)
        for bg in self.COMMON_BIGRAMS:
            if len(self.stem_vocab) < max_stems and bg not in stem_set and all(c in self.valid_word_chars for c in bg):
                self.stem_vocab.append(bg)
                stem_set.add(bg)

        # Iterative morphological stem discovery over frequent words
        cand_counts = Counter(raw_word_counts)
        self.stem_freq_map = cand_counts
        min_stem_freq = 1 if len(text) < 20000 else max(2, len(text) // 500000)
        frequent_roots = {w for w, f in cand_counts.items() if f >= min_stem_freq and len(w) >= 2}

        stem_freq = Counter()
        morph_freq = Counter()
        for low_w, freq in raw_word_counts.items():
            best_split = self._best_morph_split(low_w, frequent_roots, cand_counts)
            if best_split is not None:
                pref, stem, suf, drop_e = best_split
                stem_freq[stem] += freq
                morph_freq[("lo", pref, suf, drop_e)] += freq
                morph_freq[("Ti", pref, suf, drop_e)] += freq
                morph_freq[("UP", pref, suf, drop_e)] += freq
            else:
                stem_freq[low_w] += freq

        subword_reserve = min(256, max(0, (max_stems - len(self.stem_vocab)) // 8))
        stem_cap = max_stems - subword_reserve
        for st, freq in stem_freq.most_common():
            if len(self.stem_vocab) >= stem_cap:
                break
            if st not in stem_set and freq >= min_stem_freq:
                self.stem_vocab.append(st)
                stem_set.add(st)

        # Keep true root stems separate from subword fallback syllables
        self.stem_set = set(self.stem_vocab[4:])

        # Learn data-driven rare-word syllable chunks in reserved slots
        rare_splits = {}
        for low_w, freq in raw_word_counts.items():
            if low_w in self.stem_set:
                continue
            if self._best_morph_split(low_w, self.stem_set, cand_counts) is not None:
                continue
            syms = [ch for ch in low_w if ch in self.valid_word_chars]
            if len(syms) >= 2:
                rare_splits[low_w] = (syms, freq)

        for _ in range(subword_reserve):
            if len(self.stem_vocab) >= max_stems or not rare_splits:
                break
            pair_freqs = Counter()
            for syms, freq in rare_splits.values():
                for idx_s in range(len(syms) - 1):
                    pair_freqs[(syms[idx_s], syms[idx_s + 1])] += freq
            if not pair_freqs:
                break
            (p0, p1), best_f = pair_freqs.most_common(1)[0]
            if best_f < 2:
                break
            merged = p0 + p1
            if merged not in stem_set:
                self.stem_vocab.append(merged)
                stem_set.add(merged)
            for rw, (syms, freq) in rare_splits.items():
                if len(syms) < 2 or p0 not in syms or p1 not in syms:
                    continue
                ns, idx_s = [], 0
                while idx_s < len(syms):
                    if idx_s < len(syms) - 1 and syms[idx_s] == p0 and syms[idx_s + 1] == p1:
                        ns.append(merged)
                        idx_s += 2
                    else:
                        ns.append(syms[idx_s])
                        idx_s += 1
                rare_splits[rw] = (ns, freq)

        self.stem_to_id = {s: idx for idx, s in enumerate(self.stem_vocab)}
        self.id_to_stem = {idx: s for idx, s in enumerate(self.stem_vocab)}
        self.vocab_size = len(self.stem_vocab)

        # Build Morph Vocabulary
        self.morph_vocab = [
            self.morph_pad,
            ("lo", "", "", False),
            ("Ti", "", "", False),
            ("UP", "", "", False),
        ]
        morph_set = set(self.morph_vocab)
        for m_tup, _ in morph_freq.most_common():
            if len(self.morph_vocab) >= max_morphs:
                break
            if m_tup not in morph_set:
                self.morph_vocab.append(m_tup)
                morph_set.add(m_tup)

        self.morph_to_id = {m: idx for idx, m in enumerate(self.morph_vocab)}
        self.id_to_morph = {idx: m for idx, m in enumerate(self.morph_vocab)}
        self.n_morphs = len(self.morph_vocab)
        self.morph_plain_id = self.morph_to_id[self.morph_plain]

        # Compatibility aliases
        self.vocab = self.stem_vocab
        self.token_to_id = self.stem_to_id
        self.id_to_token = self.id_to_stem

    @staticmethod
    def _split_mixed_case(word):
        """Splits mixed-case tokens (e.g. McDonald) cleanly into lo/Ti/UP segments."""
        if len(word) <= 1 or word.islower() or word.isupper():
            return [word]
        if word[0].isupper() and not any(c.isupper() for c in word[1:]):
            return [word]
        # Split at uppercase transitions while guaranteeing lossless character coverage
        parts = re.findall(r"[A-Z]+(?=[A-Z][a-z']|$)|[A-Z][a-z']*'?|[a-z']+|[0-9]+", word)
        return parts if (parts and "".join(parts) == word) else [word]

    def _best_morph_split(self, low_w, valid_roots, root_counts):
        if len(low_w) < 4:
            return None
        best = None
        best_score = -1
        for pref in self.PREFIXES:
            if pref and not low_w.startswith(pref):
                continue
            rem = low_w[len(pref):]
            if len(rem) < 3:
                continue
            for suf, drop_e in self.SUFFIX_RULES:
                if suf:
                    if not rem.endswith(suf) or len(rem) <= len(suf) + 2:
                        continue
                    base = rem[:-len(suf)]
                else:
                    base = rem
                stem = (base + "e") if drop_e else base
                if stem in valid_roots and len(stem) >= 2:
                    # Prefer unsplit only if exact word isn't overwhelmed by root frequency
                    score = root_counts[stem] * (2 if not pref and not suf else 1) - (1 if drop_e else 0)
                    if score > best_score:
                        best_score = score
                        best = (pref, stem, suf, drop_e)
        return best

    @staticmethod
    def _detect_case(piece):
        if len(piece) > 1 and piece.isupper():
            return "UP"
        if piece and piece[0].isupper() and not any(c.isupper() for c in piece[1:]):
            return "Ti"
        return "lo"

    def _encode_single_word(self, word_piece):
        cached = self._word_cache.get(word_piece)
        if cached is not None:
            return cached

        out_units = []
        sub_pieces = self._split_mixed_case(word_piece)
        for sp in sub_pieces:
            case_tag = self._detect_case(sp)
            low = sp.lower()
            split = self._best_morph_split(low, self.stem_set, self.stem_freq_map)
            if split is not None:
                pref, stem, suf, drop_e = split
                m_tup = (case_tag, pref, suf, drop_e)
                if stem in self.stem_to_id and m_tup in self.morph_to_id:
                    out_units.append((self.stem_to_id[stem], self.morph_to_id[m_tup], len(sp)))
                    continue

            if low in self.stem_to_id:
                m_tup = (case_tag, "", "", False)
                m_id = self.morph_to_id.get(m_tup, self.morph_plain_id)
                out_units.append((self.stem_to_id[low], m_id, len(sp)))
                continue

            # Greedy longest subword/char fallback with exact case preservation
            idx = 0
            L = len(sp)
            while idx < L:
                matched = None
                for end in range(min(L, idx + 12), idx, -1):
                    sub_raw = sp[idx:end]
                    sub_case = self._detect_case(sub_raw)
                    sub_low = sub_raw.lower()
                    m_id = self.morph_to_id.get((sub_case, "", "", False), self.morph_plain_id)
                    if sub_low in self.stem_to_id and (sub_case == "lo" or self.id_to_morph[m_id][0] == sub_case):
                        matched = (self.stem_to_id[sub_low], m_id, len(sub_raw))
                        idx = end
                        break
                if matched is None:
                    ch = sp[idx]
                    ch_case = self._detect_case(ch)
                    ch_low = ch.lower()
                    s_id = self.stem_to_id.get(ch_low, self.unk_id)
                    m_id = self.morph_to_id.get((ch_case, "", "", False), self.morph_plain_id)
                    matched = (s_id, m_id, 1)
                    idx += 1
                out_units.append(matched)

        if len(self._word_cache) < 100000:
            self._word_cache[word_piece] = out_units
        return out_units

    def _split_separator(self, sep_str):
        if not sep_str:
            return [self.edge_empty_id]
        if sep_str in self.edge_to_id:
            return [self.edge_to_id[sep_str]]
        out = []
        idx = 0
        L = len(sep_str)
        while idx < L:
            matched_id = None
            for end in range(min(L, idx + 4), idx, -1):
                sub = sep_str[idx:end]
                if sub in self.edge_to_id:
                    matched_id = self.edge_to_id[sub]
                    idx = end
                    break
            if matched_id is None:
                matched_id = self.edge_unk_id
                idx += 1
            out.append(matched_id)
        return out

    def encode_atomic(self, text):
        """Encodes text into aligned 1D tensors of (stems, morphs, edges, chars)."""
        pieces = self.chunk_pat.findall(text)
        stems, morphs, edges, char_lens = [], [], [], []

        i = 0
        n_p = len(pieces)
        while i < n_p:
            p = pieces[i]
            if self.word_pat.match(p):
                word_units = self._encode_single_word(p)
                i += 1
                sep_buf = []
                while i < n_p and not self.word_pat.match(pieces[i]):
                    sep_buf.append(pieces[i])
                    i += 1
                sep_str = "".join(sep_buf)
                edge_ids = self._split_separator(sep_str)

                # Attach <CONT> to internal word pieces, first sep chunk to final piece
                for u_idx, (s_id, m_id, w_clen) in enumerate(word_units):
                    stems.append(s_id)
                    morphs.append(m_id)
                    if u_idx < len(word_units) - 1:
                        edges.append(self.edge_cont_id)
                        char_lens.append(w_clen)
                    else:
                        e0 = edge_ids[0]
                        edges.append(e0)
                        e0_str = self.id_to_edge[e0]
                        e0_clen = 0 if e0 in (self.edge_cont_id, self.edge_pad_id) else (1 if e0 == self.edge_unk_id else len(e0_str))
                        char_lens.append(w_clen + e0_clen)

                for extra_e in edge_ids[1:]:
                    stems.append(self.sep_id)
                    morphs.append(self.morph_plain_id)
                    edges.append(extra_e)
                    e_str = self.id_to_edge[extra_e]
                    char_lens.append(1 if extra_e == self.edge_unk_id else len(e_str))
            else:
                sep_buf = []
                while i < n_p and not self.word_pat.match(pieces[i]):
                    sep_buf.append(pieces[i])
                    i += 1
                sep_str = "".join(sep_buf)
                edge_ids = self._split_separator(sep_str)
                for e_id in edge_ids:
                    stems.append(self.bos_id if not stems else self.sep_id)
                    morphs.append(self.morph_plain_id)
                    edges.append(e_id)
                    e_str = self.id_to_edge[e_id]
                    char_lens.append(1 if e_id == self.edge_unk_id else len(e_str))

        return {
            "stems": torch.tensor(stems, dtype=torch.long),
            "morphs": torch.tensor(morphs, dtype=torch.long),
            "edges": torch.tensor(edges, dtype=torch.long),
            "chars": torch.tensor(char_lens, dtype=torch.float32),
        }

    def decode_word(self, stem_id, morph_id):
        if stem_id in (self.pad_id, self.bos_id, self.sep_id):
            return ""
        stem_str = self.id_to_stem.get(int(stem_id), self.unk_token)
        if stem_str == self.unk_token:
            return self.unk_token
        case_tag, pref, suf, drop_e = self.id_to_morph.get(int(morph_id), self.morph_plain)
        if case_tag == "<PAD>":
            return stem_str
        base = stem_str[:-1] if (drop_e and len(stem_str) > 1 and stem_str.endswith("e")) else stem_str
        word = f"{pref}{base}{suf}"
        if case_tag == "UP":
            return word.upper()
        if case_tag == "Ti":
            return word[0].upper() + word[1:] if word else ""
        return word

    def decode_edge(self, edge_id):
        if edge_id in (self.edge_pad_id, self.edge_cont_id):
            return ""
        if edge_id == self.edge_unk_id:
            return "?"
        return self.id_to_edge.get(int(edge_id), "")

    def decode_atomic(self, stems, morphs, edges):
        out = []
        for s_id, m_id, e_id in zip(stems, morphs, edges):
            s_v = int(s_id)
            if s_v == self.pad_id:
                continue
            out.append(self.decode_word(s_v, int(m_id)))
            out.append(self.decode_edge(int(e_id)))
        return "".join(out)

    def decode_packets(self, packet_list):
        out = []
        for pkt in packet_list:
            for s_id, m_id, e_id in pkt:
                if int(s_id) == self.pad_id:
                    break
                out.append(self.decode_word(int(s_id), int(m_id)))
                out.append(self.decode_edge(int(e_id)))
        return "".join(out)

    def format_unit(self, stem_id, morph_id=1, edge_id=None):
        w = self.decode_word(stem_id, morph_id)
        if not w and stem_id == self.sep_id and edge_id is not None:
            w = self.id_to_edge.get(int(edge_id), "")
        elif not w and stem_id == self.bos_id:
            w = "<BOS>"
        elif edge_id is not None and edge_id not in (self.edge_pad_id, self.edge_cont_id, self.edge_empty_id):
            e_s = self.id_to_edge.get(int(edge_id), "")
            if e_s not in (" ",):
                w = w + e_s
        return w.replace("\n", "\\n").replace(" ", "·")

    @staticmethod
    def format_token(tok):
        return tok.replace("\n", "\\n").replace(" ", "·")


def pack_atomic_to_packets(atomic_dict, macro_trie=None, max_pkt_len=4):
    """
    Packs an atomic stream into fixed-width (N_pkts, 4) packet tensors.
    When macro_trie is None, each atomic unit becomes an L=1 packet in slot 0.
    """
    stems = atomic_dict["stems"].tolist()
    morphs = atomic_dict["morphs"].tolist()
    edges = atomic_dict["edges"].tolist()
    chars = atomic_dict["chars"].tolist()
    N = len(stems)

    pkt_s, pkt_m, pkt_e, pkt_l, pkt_c = [], [], [], [], []
    i = 0
    while i < N:
        matched_len = 1
        if macro_trie is not None and i + 1 < N:
            node = macro_trie
            best_l = 1
            for off in range(min(max_pkt_len, N - i)):
                key = (stems[i + off], morphs[i + off], edges[i + off])
                if key in node:
                    node = node[key]
                    if "__end__" in node and off >= 1:
                        best_l = off + 1
                else:
                    break
            matched_len = best_l

        row_s = [0] * max_pkt_len
        row_m = [0] * max_pkt_len
        row_e = [0] * max_pkt_len
        c_sum = 0.0
        for j in range(matched_len):
            row_s[j] = stems[i + j]
            row_m[j] = morphs[i + j]
            row_e[j] = edges[i + j]
            c_sum += chars[i + j]

        pkt_s.append(row_s)
        pkt_m.append(row_m)
        pkt_e.append(row_e)
        pkt_l.append(matched_len)
        pkt_c.append(c_sum)
        i += matched_len

    return {
        "stems": torch.tensor(pkt_s, dtype=torch.long),
        "morphs": torch.tensor(pkt_m, dtype=torch.long),
        "edges": torch.tensor(pkt_e, dtype=torch.long),
        "lens": torch.tensor(pkt_l, dtype=torch.long),
        "chars": torch.tensor(pkt_c, dtype=torch.float32),
    }


def stratified_train_val_split(text, min_chars=12000):
    """Extracts evenly spaced contiguous validation slices without gluing unrelated val seams."""
    char_count = len(text)
    if char_count < min_chars:
        return text, [], None

    val_chars_total = min(48000, int(0.08 * char_count))
    n_slices = min(8, max(1, val_chars_total // 2000))
    if n_slices <= 1:
        split_at = char_count - val_chars_total
        ws_pos = text.find(" ", split_at)
        if ws_pos != -1 and (ws_pos - split_at) < 100:
            split_at = ws_pos + 1
        vp = text[split_at:]
        return text[:split_at], [vp], vp

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
        return text, [], None
    return train_text, val_parts, val_text


def auto_configure_from_text(text, config):
    char_count = len(text)
    unique_chars = len(set(text))

    train_text, val_parts, val_text = stratified_train_val_split(text, config["val_split_min_chars"])
    train_chars = len(train_text)
    # Keep single-char bases at min_char_freq=1 so train encoding stays 100% lossless;
    # multi-char root stems still scale via min_stem_freq inside DualClockPolarTokenizer.
    min_char_freq = 1

    tok = DualClockPolarTokenizer(
        train_text,
        max_stems=config["max_stems"],
        max_morphs=config["max_morphs"],
        max_edges=config["max_edges"],
        min_char_freq=min_char_freq,
    )
    tr_atomic = tok.encode_atomic(train_text)
    num_units = len(tr_atomic["stems"])
    stem_counts = Counter(tr_atomic["stems"].tolist())
    used_stems = len(stem_counts)
    v_size = tok.vocab_size

    if num_units < 2:
        raise ValueError("TEXT_DATA is too short—provide at least a few words.")

    n_unks = int((tr_atomic["stems"] == tok.unk_id).sum().item()) + int((tr_atomic["edges"] == tok.edge_unk_id).sum().item())
    if train_chars <= 500000:
        lossless_ok = (tok.decode_atomic(tr_atomic["stems"], tr_atomic["morphs"], tr_atomic["edges"]) == train_text) and (n_unks == 0)
    else:
        probe_text = train_text[:50000]
        probe_atom = tok.encode_atomic(probe_text)
        lossless_ok = (tok.decode_atomic(probe_atom["stems"], probe_atom["morphs"], probe_atom["edges"]) == probe_text) and (n_unks == 0)

    stem_ent = -sum((c / num_units) * math.log(c / num_units) for c in stem_counts.values())
    corpus_tok_ppx = float(math.exp(stem_ent))

    val_atom_list = []
    total_val_units = 0
    for vp in val_parts:
        va = tok.encode_atomic(vp)
        if len(va["stems"]) >= 16:
            val_atom_list.append(va)
            total_val_units += len(va["stems"])

    if total_val_units < 16:
        val_atom_list = []
        val_text = None

    comp_ratio = train_chars / max(1, num_units)

    if config.get("auto_size", True):
        n_steps = num_units - 1

        if used_stems <= 128:
            target_by_vocab = int(round(1.5 * used_stems))
        else:
            target_by_vocab = int(round(0.55 * used_stems))
        target_by_tokens = max(config["min_symbols"], int(round(0.90 * num_units)))
        raw_symbols = min(target_by_vocab, target_by_tokens)
        snapped_symbols = int(round(raw_symbols / 8.0)) * 8
        n_symbols = max(config["min_symbols"], min(config["max_symbols"], snapped_symbols))

        if n_steps < 512:
            n_ops = 8
        elif n_steps < 16384:
            n_ops = 16
        else:
            n_ops = 24

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
            batch_size = 128 if DEVICE == "cuda" else 16
            stream_len = max(32, min(n_steps, epoch_cap) // batch_size)
            adj_seq_len = 16

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
        adj_wd = 1e-5 if n_steps < 4000 else 8e-4
        adj_drop = 0.0 if n_steps < 4000 else 0.15
        adj_cell_drop = 0.0 if n_steps < 4000 else 0.10
        adj_emb_drop = 0.0 if n_steps < 4000 else 0.08
        adj_lbl_smooth = 0.0 if n_steps < 4000 else 0.05
        adj_max_macro = max(256, min(2048, used_stems // 2))
        adj_grad_clip = float(1.0 * math.sqrt(emb_dim / 64.0))
        adj_ponder = 0.030
        adj_eth_weight = float(0.14 * math.sqrt(math.log(max(2, n_symbols)) / math.log(64.0)))
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

        config["n_symbols"] = n_symbols
        config["n_ops"] = n_ops
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
        config["cell_dropout"] = adj_cell_drop
        config["emb_dropout"] = adj_emb_drop
        config["label_smoothing"] = adj_lbl_smooth
        config["max_macro_phrases"] = adj_max_macro
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

    n_steps = num_units - 1
    bs = config["batch_size"]
    s_len = config["stream_len"] if config["stream_len"] is not None else max(1, min(n_steps, config["epoch_token_cap"]) // bs)
    est_passes = (config["epochs"] * bs * s_len) / max(1, n_steps)
    p1_end = max(5, int(config["phase1_frac"] * config["epochs"]))
    p2_end = max(p1_end + 5, int(config["phase2_frac"] * config["epochs"]))

    print("=== Auto-Configuration Summary (SACRSN v60 Dual-Clock Polar + Macro-Packets) ===")
    print(f"Corpus       : {char_count:,} chars ({unique_chars} base chars, min_char_freq>={min_char_freq}) | Train={num_units:,} units"
          + (f" | Val={total_val_units:,} units ({len(val_atom_list)} slices)" if val_atom_list else " | Val=None (small text)"))
    print(
        f"Tokenizer    : DualClock-Polar | Stems={v_size} (Used={used_stems}, ZipfPPX={corpus_tok_ppx:.1f}) | "
        f"Morphs={tok.n_morphs} | Edges={tok.n_edges} | Comp={comp_ratio:.2f} chars/unit | "
        f"Lossless={'OK' if lossless_ok else f'UNK={n_unks}'} | MacroPkts={config['use_macro_packets']}"
    )
    print(
        f"Architecture : Dim={config['embedding_dim']} (2D={config['embedding_dim'] * 2}) | "
        f"Codebook={config['n_symbols']} (~{config['n_symbols'] / max(1, used_stems):.2f}x used stems, "
        f"TargetPPX=[{config['target_ppx_low']:.0f}..{config['target_ppx_high']:.0f}]) | "
        f"RelOps={config['n_ops']} | Stack={config['stack_size']} | MaxDepth={config['max_recursion_depth']}"
    )
    print(
        f"Schedule     : Streams={bs}x{s_len} (Cap={config['epoch_token_cap']:,}, TBPTT={config['seq_len']}) | "
        f"Epochs={config['epochs']} (Phases: 0..{p1_end}->KMeans->{p2_end}->Macro->{config['epochs']}, ~{est_passes:.1f}x corpus passes)"
    )
    print(
        f"InitHyper    : LR={config['learning_rate']:.5f} (Plateau every {config['eval_every']}ep, AdjMult={config['adj_lr_mult']:.1f}x) | "
        f"WD={config['weight_decay']:.1e} | Drop={config['decoder_dropout']:.2f}/{config.get('cell_dropout', 0.0):.2f}/{config.get('emb_dropout', 0.0):.2f} | LS={config.get('label_smoothing', 0.0):.2f} | ClipMax={config['grad_clip_max']:.1f} | "
        f"EthW={config['ethical_weight']:.3f} | SymLMW={config['sym_lm_weight']:.2f} | StkBalW={config['stack_balance_weight']:.2f}"
    )
    sample_preview = [
        tok.format_unit(tr_atomic["stems"][i].item(), tr_atomic["morphs"][i].item(), tr_atomic["edges"][i].item())
        for i in range(min(12, num_units))
    ]
    print(f"Sample units : {sample_preview}")
    print("====================================================================================")

    return tok, v_size, tok.n_morphs, tok.n_edges, tr_atomic, val_atom_list, val_text


def initialize_runtime(data_file=DATA_FILE, log_file=TERMINAL_LOG_FILE):
    global TEXT_DATA, tokenizer, vocab_size, n_morphs, n_edges, data_atomic, val_atomic_slices, val_text_str, USE_AMP
    logger = setup_terminal_logger(log_file)
    set_reproducibility(SEED)
    print(
        f"Hardware: {DEVICE} ({GPU_NAME}, {VRAM_GB:.1f} GB VRAM) | "
        f"BF16={USE_BF16} | TF32={DEVICE == 'cuda'}"
    )
    if "onedrive" in os.path.abspath(".").lower():
        print("[Notice] Working directory is inside OneDrive; checkpoint disk writes are deferred until milestones/finish.")
    TEXT_DATA = load_corpus_text(data_file)
    tokenizer, vocab_size, n_morphs, n_edges, data_atomic, val_atomic_slices, val_text_str = auto_configure_from_text(TEXT_DATA, CONFIG)
    USE_AMP = USE_BF16 and (CONFIG["embedding_dim"] >= 128 or CONFIG["batch_size"] >= 64)
    return logger


# ==========================================
# 3. Complex Primitives (Variance-Preserving & Native BF16-Ready)
# ==========================================
def complex_dropout(z_r, z_i, p, training):
    """Phase-preserving complex dropout (shared mask across real & imag channels)."""
    if (not training) or p <= 0.0:
        return z_r, z_i
    keep = 1.0 - float(p)
    mask = (torch.rand(z_r.shape, device=z_r.device, dtype=torch.float32) < keep).to(z_r.dtype) / keep
    return z_r * mask, z_i * mask


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
    def __init__(self, dim, init_bias=-0.05):
        super().__init__()
        self.bias = nn.Parameter(torch.full((dim,), float(init_bias)))

    def forward(self, z_r, z_i):
        eps = float(CONFIG.get("eps", 1e-5))
        norm = torch.sqrt(z_r.float() ** 2 + z_i.float() ** 2 + eps).to(z_r.dtype)
        scale = torch.clamp(F.relu(norm + self.bias) / norm, max=4.0)
        return z_r * scale, z_i * scale


class ComplexLinear(nn.Module):
    """Complex linear transform with 1/sqrt(2) orthogonal init to preserve signal power."""
    def __init__(self, dim, gain=None):
        super().__init__()
        if gain is None:
            gain = 1.0 / math.sqrt(2.0)
        self.fc_real = nn.Linear(dim, dim, bias=False)
        self.fc_imag = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.fc_real.weight, gain=gain)
        nn.init.orthogonal_(self.fc_imag.weight, gain=gain)

    def forward(self, z_r, z_i):
        return (
            self.fc_real(z_r) - self.fc_imag(z_i),
            self.fc_real(z_i) + self.fc_imag(z_r),
        )


class ComplexFFN(nn.Module):
    """Residual 2x complex MLP expansion block with variance-preserving scaling."""
    def __init__(self, dim, mult=2, dropout=None):
        super().__init__()
        hidden = dim * mult
        self.dropout = float(CONFIG.get("cell_dropout", 0.0)) if dropout is None else float(dropout)
        g1 = 1.0 / math.sqrt(2.0)
        g2 = 0.5 / math.sqrt(2.0)
        self.norm = ComplexLayerNorm(dim)
        self.fc1_r = nn.Linear(dim, hidden, bias=False)
        self.fc1_i = nn.Linear(dim, hidden, bias=False)
        self.act = ModReLU(hidden)
        self.fc2_r = nn.Linear(hidden, dim, bias=False)
        self.fc2_i = nn.Linear(hidden, dim, bias=False)
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
        o_r, o_i = complex_dropout(o_r, o_i, self.dropout, self.training)
        return z_r + 0.5 * o_r, z_i + 0.5 * o_i


# ==========================================
# 4. Memory & Compositional Symbolic Modules (Stack, Holo, GraphVQ, RelOpVQ)
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
        clear_decay = float(CONFIG.get("stack_clear_decay", 0.80))
        write_val = write_mask.unsqueeze(2) * z_flat.unsqueeze(1)
        retain_mask = (1.0 - write_mask.unsqueeze(2)) * (1.0 - clear_decay * clear_mask.unsqueeze(2)) * 0.992
        new_memory = write_val + (memory * retain_mask)

        read_ptr = push * ptr + (1.0 - push) * new_ptr
        read_mask = read_ptr.unsqueeze(2)
        read_flat = torch.sum(new_memory * read_mask, dim=1)
        read_r = read_flat[:, :self.dim].to(z_r.dtype)
        read_i = read_flat[:, self.dim:].to(z_i.dtype)

        return read_r, read_i, new_memory, new_ptr


class HolographicMemory(nn.Module):
    """Complex Phase-Bound Associative Memory (FHRR / Associative Trace)."""
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


class RelationalOperatorVQ(nn.Module):
    """Quantizes state-transition + edge deltas into emergent complex relational operators (R_t)."""
    def __init__(self, latent_dim, n_ops):
        super().__init__()
        self.dim = latent_dim
        self.n_ops = n_ops
        self.op_proj = ComplexLinear(latent_dim)
        self.codebook = nn.Parameter(torch.empty(n_ops, latent_dim * 2))
        nn.init.normal_(self.codebook, mean=0.0, std=1.0)
        self.normalize_codebook_weights_()

        self.rel_logit_scale = nn.Parameter(torch.tensor(4.0))
        self.register_buffer("usage_count", torch.zeros(n_ops))
        self.register_buffer("ema_usage", torch.full((n_ops,), 1.0 / n_ops))
        self.register_buffer("ema_d_std", torch.tensor(0.12 / 0.40))

    @torch.no_grad()
    def normalize_codebook_weights_(self):
        rms = torch.sqrt(torch.mean(self.codebook ** 2, dim=-1, keepdim=True) + 1e-4)
        self.codebook.div_(rms)

    @property
    def live_temp(self):
        return float(torch.clamp(self.ema_d_std * 0.40, min=0.05, max=0.22).item())

    def normalized_codebook(self):
        cb_f = self.codebook.float()
        cb_rms = torch.sqrt(torch.mean(cb_f ** 2, dim=-1, keepdim=True) + 1e-4)
        return cb_f / cb_rms

    def forward(self, delta_r, delta_i, active_weights=None, conscience_scale=0.0, commit_cost=0.25):
        pr, pi = self.op_proj(delta_r, delta_i)
        op_raw = torch.cat([pr.float(), pi.float()], dim=-1)
        op_flat = op_raw / torch.sqrt(torch.mean(op_raw ** 2, dim=-1, keepdim=True) + 1e-4)
        cb_norm = self.normalized_codebook()

        d_raw = (
            torch.sum(op_flat ** 2, dim=-1, keepdim=True)
            + torch.sum(cb_norm ** 2, dim=-1)
            - 2.0 * torch.matmul(op_flat, cb_norm.t())
        ) / op_flat.shape[-1]

        if self.training and d_raw.numel() > 1:
            with torch.no_grad():
                batch_d_std = d_raw.detach().std(dim=-1).mean().clamp(min=0.02, max=2.0)
                self.ema_d_std.mul_(0.95).add_(0.05 * batch_d_std)

        temp_tensor = torch.clamp(self.ema_d_std * 0.40, min=0.05, max=0.22)
        soft_probs = F.softmax(-d_raw / temp_tensor, dim=-1)

        d_select = d_raw
        if self.training and conscience_scale > 0.0:
            rel_usage = self.ema_usage * float(self.n_ops)
            hub_penalty = conscience_scale * F.relu(torch.log1p(rel_usage) - math.log(4.0))
            d_select = d_select + hub_penalty.unsqueeze(0)

        op_indices = torch.argmin(d_select, dim=-1)

        if self.training:
            with torch.no_grad():
                if active_weights is None:
                    weights = torch.ones_like(op_indices.view(-1), dtype=self.usage_count.dtype)
                else:
                    weights = active_weights.view(-1).to(self.usage_count.dtype)
                self.usage_count.index_add_(0, op_indices.view(-1), weights)
                w_sum = weights.sum()
                if w_sum > 0:
                    batch_hist = torch.zeros_like(self.ema_usage)
                    batch_hist.index_add_(0, op_indices.view(-1), weights / w_sum)
                    self.ema_usage.mul_(0.995).add_(0.005 * batch_hist)

        omega_q = F.embedding(op_indices, cb_norm)
        loss_vq = torch.mean((omega_q - op_flat.detach()) ** 2, dim=-1)
        loss_commit = torch.mean((omega_q.detach() - op_flat) ** 2, dim=-1)
        op_vq_loss = loss_vq + commit_cost * loss_commit
        op_cos = F.cosine_similarity(op_flat.detach(), omega_q.detach(), dim=-1)

        omega_ste = op_flat + (omega_q - op_flat).detach()
        return omega_ste, op_indices, soft_probs, op_vq_loss, op_flat.detach(), op_cos.detach()

    def bind_transition(self, prev_sym, omega_ste, cb_norm):
        """ComplEx relational composition: e_{prev} * omega_{op} -> predicted next symbol vector & logits."""
        e_prev = F.embedding(prev_sym, cb_norm)
        e_r = e_prev[..., :self.dim]
        e_i = e_prev[..., self.dim:]
        w_r = omega_ste.float()[..., :self.dim]
        w_i = omega_ste.float()[..., self.dim:]

        b_r = e_r * w_r - e_i * w_i
        b_i = e_r * w_i + e_i * w_r
        b_raw = torch.cat([b_r, b_i], dim=-1)
        b_norm = b_raw / torch.sqrt(torch.mean(b_raw ** 2, dim=-1, keepdim=True) + 1e-4)

        cos_all = torch.matmul(b_norm, cb_norm.t()) / float(2 * self.dim)
        scale = torch.clamp(self.rel_logit_scale.float(), min=1.0, max=12.0)
        rel_logits = cos_all * scale
        return b_norm, rel_logits


class GraphMemoryVQ(nn.Module):
    def __init__(self, latent_dim, n_symbols):
        super().__init__()
        self.n_symbols = n_symbols

        self.codebook = nn.Parameter(torch.empty(n_symbols, latent_dim * 2))
        nn.init.normal_(self.codebook, mean=0.0, std=1.0)
        self.normalize_codebook_weights_()

        self.adjacency = nn.Parameter(torch.zeros(n_symbols, n_symbols))
        self.register_buffer("usage_count", torch.zeros(n_symbols))
        self.register_buffer("ema_usage", torch.full((n_symbols,), 1.0 / n_symbols))
        self.register_buffer("ema_d_std", torch.tensor(float(CONFIG["vq_temp"]) / 0.40))

    @torch.no_grad()
    def normalize_codebook_weights_(self):
        rms = torch.sqrt(torch.mean(self.codebook ** 2, dim=-1, keepdim=True) + 1e-4)
        self.codebook.div_(rms)

    @property
    def live_temp(self):
        return float(torch.clamp(self.ema_d_std * 0.40, min=0.05, max=0.18).item())

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
        rel_logits=None,
    ):
        if commit_cost is None:
            commit_cost = CONFIG["commitment_cost"]

        dim = z_r.shape[-1]
        z_raw = torch.cat([z_r.float(), z_i.float()], dim=-1)
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
            if rel_logits is not None:
                graph_prior = graph_prior + rel_logits.detach()
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

        z_q = F.embedding(min_indices, cb_norm)

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
# 5. Core Processor (Attention, GWT Router, Relational Transition Loss, Cell)
# ==========================================
class ComplexAttention(nn.Module):
    def __init__(self, dim, dropout=None):
        super().__init__()
        self.dim = dim
        self.dropout = float(CONFIG.get("cell_dropout", 0.0)) if dropout is None else float(dropout)
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
        out_r, out_i = complex_dropout(out_r, out_i, self.dropout, self.training)

        return z_r + 0.25 * out_r, z_i + 0.25 * out_i


class GlobalWorkspaceRouter(nn.Module):
    """Competitive broadcast router across Perception, Stack, Holographic, and Graph streams."""
    def __init__(self, dim):
        super().__init__()
        self.scorer = nn.Linear(dim * 2, 1)
        self.stream_bias = nn.Parameter(torch.zeros(4))

    def forward(self, p_r, p_i, stk_r, stk_i, holo_r, holo_i, g_r, g_i, stream_mask=None):
        streams_r = torch.stack([p_r, stk_r, holo_r, g_r], dim=1)
        streams_i = torch.stack([p_i, stk_i, holo_i, g_i], dim=1)
        streams_flat = torch.cat([streams_r, streams_i], dim=-1)

        logits = self.scorer(streams_flat).squeeze(-1).float() + self.stream_bias
        if stream_mask is not None:
            logits = logits.masked_fill(~stream_mask.view(1, 4), -1e9)
        gates = F.softmax(logits, dim=-1)
        g_cast = gates.to(p_r.dtype)

        p_weight = 0.5 + 0.5 * g_cast[:, 0:1]
        ws_r = p_weight * p_r + g_cast[:, 1:2] * stk_r + g_cast[:, 2:3] * holo_r + g_cast[:, 3:4] * g_r
        ws_i = p_weight * p_i + g_cast[:, 1:2] * stk_i + g_cast[:, 2:3] * holo_i + g_cast[:, 3:4] * g_i
        return ws_r, ws_i, gates


class EthicalConstraint(nn.Module):
    """Relational transition constraint scoring P(S_t | S_{t-1}, R_t) over graph + ComplEx bindings."""
    def __init__(self):
        super().__init__()

    def forward(self, prev_sym, curr_sym, adjacency, curr_conf=None, rel_logits=None):
        if prev_sym is None:
            return torch.zeros((), device=adjacency.device, dtype=torch.float32)
        n_syms = adjacency.size(-1)
        row_logits = adjacency[prev_sym].float().view(-1, n_syms)
        if rel_logits is not None:
            row_logits = row_logits + rel_logits.float().view(-1, n_syms)
        smoothing = 0.05 if self.training else 0.0
        ce = F.cross_entropy(row_logits, curr_sym.view(-1), reduction="none", label_smoothing=smoothing)
        if curr_conf is not None:
            w = curr_conf.detach().view(-1).clamp(min=0.1, max=1.0)
            loss = (ce * w).sum() / w.sum().clamp(min=1.0)
        else:
            loss = ce.mean()
        return loss


class AdaptiveRecursiveCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = ComplexLinear(dim)
        self.norm = ComplexLayerNorm(dim)
        self.act = ModReLU(dim)
        self.attention = ComplexAttention(dim)
        self.ffn = ComplexFFN(dim, mult=2)
        self.halt_linear = nn.Linear(dim * 2, 1)
        self.stack_ctrl = nn.Linear(dim * 2, 3)

        p0 = float(CONFIG.get("ponder_prior_halt", 0.45))
        p0 = min(0.60, max(0.20, p0))
        halt_bias = float(math.log(p0 / (1.0 - p0)))
        nn.init.constant_(self.halt_linear.bias, halt_bias)

    def forward(self, z_r, z_i, stack_mem=None, stack_bias=None):
        l_r, l_i = self.linear(z_r, z_i)
        n_r, n_i = self.norm(l_r, l_i)
        a_r, a_i = self.act(n_r, n_i)
        att_r, att_i = self.attention(a_r, a_i, stack_mem)
        p_r, p_i = self.ffn(att_r, att_i)
        z_flat = torch.cat([p_r, p_i], dim=-1)
        halt_prob = torch.sigmoid(self.halt_linear(z_flat).float())
        stk_logits = self.stack_ctrl(z_flat).float()
        if stack_bias is not None:
            stk_logits = stk_logits + stack_bias
        stack_probs = F.softmax(stk_logits / CONFIG["stack_temp"], dim=-1)
        return p_r, p_i, halt_prob, stack_probs


# ==========================================
# 6. Master Model (UberCRSN v56: Dual-Clock Polar + Holographic Macro-Packets)
# ==========================================
class UberCRSN(nn.Module):
    def __init__(self, n_stems, dim, n_morph_classes=4, n_edge_classes=4):
        super().__init__()
        self.dim = dim
        self.n_stems = n_stems
        self.n_morphs = n_morph_classes
        self.n_edges = n_edge_classes
        self.max_pkt_len = CONFIG.get("max_pkt_len", 4)

        # Polar-Factored Lexeme + Morphological Phase Embeddings
        self.emb_mag = nn.Embedding(n_stems, dim, padding_idx=0)
        self.emb_phase = nn.Parameter(torch.randn(n_stems, dim))
        self.morph_scale_delta = nn.Embedding(n_morph_classes, dim, padding_idx=0)
        self.morph_phase = nn.Embedding(n_morph_classes, dim, padding_idx=0)
        nn.init.zeros_(self.morph_scale_delta.weight)
        nn.init.normal_(self.morph_phase.weight, mean=0.0, std=0.15)
        with torch.no_grad():
            self.morph_phase.weight[0].zero_()
            if n_morph_classes > 1:
                self.morph_phase.weight[1].zero_()

        # Dual-Clock Structural Edge Embeddings (routed to Stack & RelOp, not VQ)
        self.edge_emb = nn.Embedding(n_edge_classes, dim * 2, padding_idx=0)
        self.edge_to_op = nn.Linear(dim * 2, dim * 2)
        self.edge_to_stack = nn.Linear(dim * 2, 3)

        # Positional Complex Phase Keys for Holographic Packet Superposition/Unbinding
        # Slot 0 has phase 0.0 -> L=1 packets incur zero positional distortion
        g_pos = torch.Generator().manual_seed(SEED + 101)
        pos_keys = torch.rand(self.max_pkt_len, dim, generator=g_pos) * (2.0 * math.pi)
        pos_keys[0].zero_()
        self.register_buffer("pos_phase_keys", pos_keys)
        self.register_buffer("pos_cos", torch.cos(pos_keys))
        self.register_buffer("pos_sin", torch.sin(pos_keys))

        self.carry_gate = nn.Linear(dim * 4, dim)
        nn.init.constant_(self.carry_gate.bias, 0.0)

        self.cell = AdaptiveRecursiveCell(dim)
        self.vq_layer = GraphMemoryVQ(dim, CONFIG["n_symbols"])
        self.op_vq = RelationalOperatorVQ(dim, CONFIG["n_ops"])
        self.holo_mem = HolographicMemory(
            dim,
            decay=CONFIG["holo_decay"],
            write_scale=CONFIG["holo_write_scale"],
        )
        self.gwt_router = GlobalWorkspaceRouter(dim)
        self.dec_norm = nn.LayerNorm(dim * 2)
        self.dec_drop = nn.Dropout(CONFIG.get("decoder_dropout", 0.10))

        # Causal Slot Unbinder + Shared Multi-Channel Decoders
        self.slot_ctx_proj = nn.Linear(dim * 4, dim * 2)
        self.slot_ctx_gate = nn.Parameter(torch.tensor(-0.5))
        self.len_head = nn.Linear(dim * 2, self.max_pkt_len)
        self.decoder = nn.Linear(dim * 2, n_stems)
        self.morph_decoder = nn.Linear(dim * 2, n_morph_classes)
        self.edge_decoder = nn.Linear(dim * 2, n_edge_classes)

        # Grounded pure-symbolic predictive decoder: [e_{S_t}, e_{S_{t-1}} o omega_{R_t}] -> next head stem
        self.sym_proj = nn.Linear(dim * 4, dim * 2)
        self.sym_dec_norm = nn.LayerNorm(dim * 2)
        self.sym_decoder = nn.Linear(dim * 2, n_stems)

        if CONFIG["use_stack"]:
            self.stack = DifferentiableStack(dim, CONFIG["stack_size"])

        self.ethics = EthicalConstraint()
        self.register_buffer("prev_sym_soft", torch.full((CONFIG["n_symbols"],), 1.0 / CONFIG["n_symbols"]))
        self.register_buffer("sym_alpha", torch.tensor(0.0))
        self.register_buffer("graph_gate", torch.tensor(0.0))

    def embed_polar_words(self, stem_ids, morph_ids=None, ablate_morph=False):
        """Polar complex word embedding: (r_stem * s_morph) * exp(i * (th_stem + dphi_morph))."""
        r = self.emb_mag(stem_ids)
        t = self.emb_phase[stem_ids]
        if morph_ids is not None and not ablate_morph:
            s_mod = 1.0 + 0.5 * torch.tanh(self.morph_scale_delta(morph_ids))
            d_phi = self.morph_phase(morph_ids)
            r = r * s_mod
            t = t + d_phi
        return r * torch.cos(t), r * torch.sin(t)

    def embed_packet(self, pkt_stems, pkt_morphs=None, pkt_lens=None, ablate_morph=False):
        """Superposes L=1..4 polar word vectors into 1 complex vector via positional phase keys."""
        if pkt_stems.dim() == 1:
            pkt_stems = pkt_stems.unsqueeze(1)
            if pkt_morphs is not None and pkt_morphs.dim() == 1:
                pkt_morphs = pkt_morphs.unsqueeze(1)

        S_curr = pkt_stems.size(1)
        w_r, w_i = self.embed_polar_words(pkt_stems, pkt_morphs, ablate_morph=ablate_morph)
        if S_curr == 1:
            return w_r.squeeze(1), w_i.squeeze(1)

        k_c = self.pos_cos[:S_curr].unsqueeze(0).to(w_r.dtype)
        k_s = self.pos_sin[:S_curr].unsqueeze(0).to(w_i.dtype)
        rot_r = w_r * k_c - w_i * k_s
        rot_i = w_r * k_s + w_i * k_c

        slot_mask = (pkt_stems > 0).unsqueeze(-1).to(w_r.dtype)
        if pkt_lens is None:
            sqrt_l = torch.sqrt(slot_mask.sum(dim=1).clamp(min=1.0))
        else:
            sqrt_l = torch.sqrt(pkt_lens.float().clamp(min=1.0)).unsqueeze(-1).to(w_r.dtype)

        z_r = (rot_r * slot_mask).sum(dim=1) / sqrt_l
        z_i = (rot_i * slot_mask).sum(dim=1) / sqrt_l
        return z_r, z_i

    def decode_slot_features(self, z_w_r, z_w_i, slot_idx, prev_slot_flat=None):
        """Conjugate phase unbinding for slot j with optional causal previous-slot conditioning."""
        if slot_idx == 0:
            u_r, u_i = z_w_r, z_w_i
        else:
            kc = self.pos_cos[slot_idx].unsqueeze(0).to(z_w_r.dtype)
            ks = self.pos_sin[slot_idx].unsqueeze(0).to(z_w_i.dtype)
            u_r = z_w_r * kc + z_w_i * ks
            u_i = z_w_i * kc - z_w_r * ks

        u_flat = torch.cat([u_r, u_i], dim=-1)
        if slot_idx > 0 and prev_slot_flat is not None:
            gate = torch.sigmoid(self.slot_ctx_gate).to(u_flat.dtype)
            ctx_delta = self.slot_ctx_proj(torch.cat([u_flat, prev_slot_flat], dim=-1))
            u_flat = u_flat + gate * ctx_delta
        return self.dec_drop(self.dec_norm(u_flat))

    def decode_next_packet(self, z_w_r, z_w_i, tgt_stems=None, tgt_morphs=None, max_slots=1, ablate_morph=False):
        """Decodes len_logits + (stem, morph, edge) logits across 1..max_slots unbound slots."""
        base_feat = self.dec_drop(self.dec_norm(torch.cat([z_w_r, z_w_i], dim=-1)))
        len_logits = self.len_head(base_feat)

        if max_slots <= 1:
            s_l = self.decoder(base_feat).unsqueeze(1)
            m_l = self.morph_decoder(base_feat).unsqueeze(1)
            e_l = self.edge_decoder(base_feat).unsqueeze(1)
            return s_l, m_l, e_l, len_logits

        slot_s, slot_m, slot_e = [], [], []
        prev_flat = None
        for j in range(max_slots):
            feat_j = self.decode_slot_features(z_w_r, z_w_i, j, prev_slot_flat=prev_flat)
            s_j = self.decoder(feat_j)
            m_j = self.morph_decoder(feat_j)
            e_j = self.edge_decoder(feat_j)
            slot_s.append(s_j)
            slot_m.append(m_j)
            slot_e.append(e_j)

            if j + 1 < max_slots:
                if tgt_stems is not None and j < tgt_stems.size(1):
                    p_stem = tgt_stems[:, j]
                    p_morph = tgt_morphs[:, j] if tgt_morphs is not None else None
                else:
                    p_stem = s_j.argmax(dim=-1)
                    p_morph = m_j.argmax(dim=-1)
                pw_r, pw_i = self.embed_polar_words(p_stem, p_morph, ablate_morph=ablate_morph)
                prev_flat = torch.cat([pw_r, pw_i], dim=-1)

        return (
            torch.stack(slot_s, dim=1),
            torch.stack(slot_m, dim=1),
            torch.stack(slot_e, dim=1),
            len_logits,
        )

    def forward(
        self,
        pkt_stems,
        pkt_morphs=None,
        pkt_lens=None,
        prev_edge=None,
        hidden=None,
        prev_sym=None,
        tgt_stems=None,
        tgt_morphs=None,
        decode_slots=1,
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
        ablate_rel_op=False,
        ablate_edge=False,
        ablate_morph=False,
    ):
        if bias_scale is None:
            bias_scale = CONFIG["graph_bias_scale"]

        use_echo_eff = use_graph_echo and (not ablate_graph)
        bias_eff = 0.0 if ablate_graph else bias_scale
        use_stack_engine = CONFIG["use_stack"] and (not ablate_stack)
        use_rel_op = not ablate_rel_op

        batch_size = pkt_stems.size(0)
        z_r, z_i = self.embed_packet(pkt_stems, pkt_morphs, pkt_lens, ablate_morph=ablate_morph)
        z_r, z_i = complex_dropout(z_r, z_i, float(CONFIG.get("emb_dropout", 0.0)), self.training)

        # Dual-Clock Structural Edge signal (routed to Stack & RelOp, never VQ concepts)
        if prev_edge is not None and not ablate_edge:
            e_vec = self.edge_emb(prev_edge.view(-1)).to(z_r.dtype)
            edge_stack_bias = self.edge_to_stack(e_vec).float()
            edge_op_delta = self.edge_to_op(e_vec)
            edge_op_r = edge_op_delta[:, :self.dim]
            edge_op_i = edge_op_delta[:, self.dim:]
        else:
            edge_stack_bias = None
            edge_op_r = None
            edge_op_i = None

        if hidden is None:
            stack_mem = torch.zeros(batch_size, CONFIG["stack_size"], self.dim * 2, device=z_r.device, dtype=torch.float32)
            stack_ptr = torch.zeros(batch_size, CONFIG["stack_size"], device=z_r.device, dtype=torch.float32)
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

        z0_r, z0_i = z_r, z_i

        cb_norm = cached_cb_norm if cached_cb_norm is not None else self.vq_layer.normalized_codebook()
        if use_echo_eff:
            P_adj = cached_P_adj if cached_P_adj is not None else F.softmax(self.vq_layer.adjacency.float(), dim=-1)
        else:
            P_adj = None

        g_gate = self.graph_gate.to(z_r.dtype)
        alpha = torch.zeros((), device=z_r.device, dtype=z_r.dtype) if ablate_vq else self.sym_alpha.to(z_r.dtype)
        log_K = float(math.log(max(2, CONFIG["n_symbols"])))

        base_mask = [
            True,
            bool(use_stack_engine and (not ablate_stack_read)),
            bool(not ablate_holo),
        ]
        stream_mask_echo_off = torch.tensor(base_mask + [False], device=z_r.device, dtype=torch.bool)
        stream_mask_echo_on = (
            torch.tensor(base_mask + [True], device=z_r.device, dtype=torch.bool)
            if use_echo_eff
            else stream_mask_echo_off
        )

        if use_echo_eff and prev_sym is not None:
            sym_belief = P_adj[prev_sym]
        else:
            sym_belief = None
        last_b_norm = None

        unhalted = torch.ones(batch_size, 1, device=z_r.device, dtype=torch.float32)
        cum_halt = torch.zeros(batch_size, 1, device=z_r.device, dtype=torch.float32)
        selected_done = torch.zeros(batch_size, dtype=torch.bool, device=z_r.device)
        steps_taken = torch.zeros(batch_size, 1, device=z_r.device, dtype=torch.float32)
        step_p_list = []

        stack_depth_sum = torch.zeros((), device=z_r.device, dtype=torch.float32)
        stack_ctrl_sum = torch.zeros(3, device=z_r.device, dtype=torch.float32)
        stack_noop_vec = torch.zeros(batch_size, device=z_r.device, dtype=torch.float32)
        gwt_gates_sum = torch.zeros(4, device=z_r.device, dtype=torch.float32)
        sig_rms_sum = torch.zeros(4, device=z_r.device, dtype=torch.float32)

        step_soft_probs = []
        step_active_weights = []
        op_step_soft_probs = []
        t0_z_flats = None
        t0_op_flats = None

        z_w_r = torch.zeros(batch_size, self.dim, device=z_r.device, dtype=z_r.dtype)
        z_w_i = torch.zeros(batch_size, self.dim, device=z_i.device, dtype=z_i.dtype)
        cont_w_r = torch.zeros(batch_size, self.dim, device=z_r.device, dtype=z_r.dtype)
        cont_w_i = torch.zeros(batch_size, self.dim, device=z_i.device, dtype=z_i.dtype)
        cell_w_r = torch.zeros(batch_size, self.dim, device=z_r.device, dtype=z_r.dtype)
        cell_w_i = torch.zeros(batch_size, self.dim, device=z_i.device, dtype=z_i.dtype)
        sym_w_r = torch.zeros(batch_size, self.dim, device=z_r.device, dtype=z_r.dtype)
        sym_w_i = torch.zeros(batch_size, self.dim, device=z_i.device, dtype=z_i.dtype)
        rel_w = torch.zeros(batch_size, self.dim * 2, device=z_r.device, dtype=torch.float32)

        current_sym = None
        current_op = None
        halt_rel_logits = None
        prev_step_soft = None
        curr_step_conf = None
        vq_loss_sum = torch.zeros((), device=z_r.device, dtype=torch.float32)
        vq_cos_sum = torch.zeros((), device=z_r.device, dtype=torch.float32)
        op_vq_loss_sum = torch.zeros((), device=z_r.device, dtype=torch.float32)
        op_cos_sum = torch.zeros((), device=z_r.device, dtype=torch.float32)
        cons_loss_sum = torch.zeros((), device=z_r.device, dtype=torch.float32)
        cons_weight_total = torch.zeros((), device=z_r.device, dtype=torch.float32)
        p_weight_total = torch.zeros((), device=z_r.device, dtype=torch.float32)

        if use_stack_engine:
            slot_indices = torch.arange(CONFIG["stack_size"], device=z_r.device, dtype=torch.float32)

        M = CONFIG["max_recursion_depth"]
        act_thresh = CONFIG["act_threshold"]
        commit_val = CONFIG["commitment_cost"] if commit_cost is None else commit_cost

        for t in range(M):
            mem_arg = stack_mem if (use_stack_engine and not ablate_stack_attn) else None
            stk_bias_arg = edge_stack_bias if (t == 0 and edge_stack_bias is not None) else (0.3 * edge_stack_bias if edge_stack_bias is not None else None)
            p_r, p_i, lambda_t, stack_ctrl = self.cell(z_r, z_i, mem_arg, stack_bias=stk_bias_arg)

            is_forced_exit = ablate_act_1step or (t == M - 1)
            if is_forced_exit:
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
            stack_noop_vec = stack_noop_vec + (p_w_det * stack_ctrl[:, 2].detach())

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
                echo_vec = torch.matmul(sym_belief, cb_norm)
                if last_b_norm is not None:
                    echo_vec = 0.65 * echo_vec + 0.35 * last_b_norm
                g_echo = echo_vec.to(p_r.dtype) * g_gate
                g_echo_r = g_echo[:, :self.dim]
                g_echo_i = g_echo[:, self.dim:]
                step_stream_mask = stream_mask_echo_on
            else:
                g_echo_r = torch.zeros_like(p_r)
                g_echo_i = torch.zeros_like(p_i)
                step_stream_mask = stream_mask_echo_off

            comb_r, comb_i, gwt_gates = self.gwt_router(
                p_r, p_i, read_r, read_i, holo_read_r, holo_read_i, g_echo_r, g_echo_i, stream_mask=step_stream_mask
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

            # Extract Relational Operator R_t from reasoning delta + Dual-Clock boundary edge
            delta_r = comb_r - z0_r
            delta_i = comb_i - z0_i
            if edge_op_r is not None:
                delta_r = delta_r + 0.5 * edge_op_r
                delta_i = delta_i + 0.5 * edge_op_i

            omega_ste, op_idx, op_soft, op_vq_vec, op_flat_det, op_cos_vec = self.op_vq(
                delta_r,
                delta_i,
                active_weights=p_w_det,
                conscience_scale=conscience_scale,
                commit_cost=commit_val,
            )
            if t == 0:
                t0_op_flats = op_flat_det

            if use_rel_op and prev_sym is not None:
                b_norm, step_rel_logits = self.op_vq.bind_transition(prev_sym, omega_ste, cb_norm)
                last_b_norm = b_norm
            else:
                b_norm = torch.zeros(batch_size, self.dim * 2, device=z_r.device, dtype=torch.float32)
                step_rel_logits = None

            prior_sym = prev_sym if t == 0 else None
            step_bias = bias_eff if t == 0 else 0.0

            sym_r, sym_i, norm_r, norm_i, vq_loss_vec, sym_idx, soft_probs, z_flat_det, vq_cos_vec = self.vq_layer(
                comb_r,
                comb_i,
                prior_sym,
                bias_scale=step_bias,
                commit_cost=commit_val,
                active_weights=p_w_det,
                cb_norm=cb_norm,
                conscience_scale=conscience_scale,
                rel_logits=step_rel_logits if t == 0 else None,
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
                current_op = op_idx
                halt_rel_logits = step_rel_logits
            else:
                current_sym = torch.where(is_halt_step, sym_idx, current_sym)
                current_op = torch.where(is_halt_step, op_idx, current_op)
                if step_rel_logits is not None:
                    if halt_rel_logits is None:
                        halt_rel_logits = step_rel_logits
                    else:
                        halt_rel_logits = torch.where(is_halt_step.unsqueeze(-1), step_rel_logits, halt_rel_logits)

            if curr_step_conf is None:
                curr_step_conf = step_conf
            else:
                curr_step_conf = torch.where(is_halt_step, step_conf, curr_step_conf)

            selected_done = selected_done | is_halt_step

            step_soft_probs.append(soft_probs)
            op_step_soft_probs.append(op_soft)
            step_active_weights.append(p_w_det)

            z_r = (1.0 - alpha) * norm_r + alpha * sym_r
            z_i = (1.0 - alpha) * norm_i + alpha * sym_i

            p_cast = p_t.to(z_r.dtype)
            z_w_r = z_w_r + (p_cast * z_r)
            z_w_i = z_w_i + (p_cast * z_i)
            cont_w_r = cont_w_r + (p_cast * norm_r)
            cont_w_i = cont_w_i + (p_cast * norm_i)
            cell_w_r = cell_w_r + (p_cast * p_r)
            cell_w_i = cell_w_i + (p_cast * p_i)
            sym_w_r = sym_w_r + (p_cast * sym_r)
            sym_w_i = sym_w_i + (p_cast * sym_i)
            rel_w = rel_w + (p_t * b_norm)

            vq_loss_sum = vq_loss_sum + (vq_loss_vec * p_w_det).sum()
            vq_cos_sum = vq_cos_sum + (vq_cos_vec * p_w_det).sum()
            op_vq_loss_sum = op_vq_loss_sum + (op_vq_vec * p_w_det).sum()
            op_cos_sum = op_cos_sum + (op_cos_vec * p_w_det).sum()
            p_weight_total = p_weight_total + p_w_det.sum()

            cum_halt = new_cum_halt
            unhalted = unhalted_next

            if ablate_act_1step:
                break

        denom = p_weight_total.clamp(min=1.0)
        vq_loss_total = vq_loss_sum / denom
        vq_cos_mean = vq_cos_sum / denom
        op_vq_loss_total = op_vq_loss_sum / denom
        op_cos_mean = op_cos_sum / denom
        cons_loss_total = cons_loss_sum / cons_weight_total.clamp(min=1.0)
        avg_stack = stack_depth_sum / denom
        avg_stack_ctrl = stack_ctrl_sum / denom
        avg_gwt_gates = gwt_gates_sum / denom

        ethical_loss = self.ethics(
            prev_sym,
            current_sym,
            self.vq_layer.adjacency,
            curr_conf=curr_step_conf,
            rel_logits=halt_rel_logits if use_rel_op else None,
        )

        p_stack = torch.cat(step_p_list, dim=1)
        n_run = p_stack.size(1)
        lam_p = float(CONFIG.get("ponder_prior_halt", 0.45))
        prior_probs = [lam_p * ((1.0 - lam_p) ** i) for i in range(n_run - 1)]
        prior_probs.append((1.0 - lam_p) ** (n_run - 1))
        p_prior = torch.tensor(prior_probs, device=z_r.device, dtype=torch.float32).clamp(min=1e-6)
        p_prior = p_prior / p_prior.sum()
        kl_ponder = (p_stack * (torch.log(p_stack.clamp(min=1e-6)) - torch.log(p_prior.view(1, -1)))).sum(dim=-1)
        step_indices = torch.arange(1, n_run + 1, device=z_r.device, dtype=torch.float32)
        exp_steps = (p_stack * step_indices.view(1, -1)).sum(dim=-1)
        ponder_loss = (kl_ponder + 0.10 * (exp_steps - 1.0)).mean()
        act_steps_mean = steps_taken.mean().detach()

        stem_logits, morph_logits, edge_logits, len_logits = self.decode_next_packet(
            z_w_r,
            z_w_i,
            tgt_stems=tgt_stems,
            tgt_morphs=tgt_morphs,
            max_slots=decode_slots,
            ablate_morph=ablate_morph,
        )

        next_r = 0.5 * cont_w_r + 0.5 * z_w_r
        next_i = 0.5 * cont_w_i + 0.5 * z_w_i
        if ablate_holo:
            next_holo_r, next_holo_i = holo_r, holo_i
        else:
            next_holo_r, next_holo_i = self.holo_mem.write(cell_w_r, cell_w_i, holo_r, holo_i)
        next_hidden = (next_r, next_i, stack_mem, stack_ptr, next_holo_r, next_holo_i)

        if return_stats:
            sym_cat = torch.cat([sym_w_r.float(), sym_w_i.float(), rel_w], dim=-1).to(z_w_r.dtype)
            sym_feat = F.gelu(self.sym_proj(sym_cat))
            sym_logits = self.sym_decoder(self.sym_dec_norm(sym_feat))

            if prev_sym is not None:
                trans_logits = self.vq_layer.adjacency[prev_sym].float()
                if use_rel_op and halt_rel_logits is not None:
                    trans_logits = trans_logits + halt_rel_logits.float()
                rel_acc = (trans_logits.argmax(dim=-1) == current_sym).float().mean().detach()
                halt_rel_prob = F.softmax(trans_logits, dim=-1).gather(1, current_sym.view(-1, 1)).squeeze(1).detach()
            else:
                rel_acc = torch.zeros((), device=z_r.device, dtype=torch.float32)
                halt_rel_prob = torch.zeros(batch_size, device=z_r.device, dtype=torch.float32)

            ptr_entropy = -(stack_ptr * torch.log(stack_ptr + 1e-8)).sum(dim=-1).mean()
            mem_occ_pct = (stack_mem.detach().abs().amax(dim=-1) > 1e-4).float().mean() * 100.0
            extra_stats = {
                "sig_rms": (sig_rms_sum / denom).detach(),
                "carry_mean": carry_mean_val.detach(),
                "carry_std": carry_std_val.detach(),
                "sym_logits": sym_logits,
                "morph_logits": morph_logits,
                "edge_logits": edge_logits,
                "len_logits": len_logits,
                "z_w_r": z_w_r.detach(),
                "z_w_i": z_w_i.detach(),
                "op_vq_loss": op_vq_loss_total,
                "op_cos": op_cos_mean.detach(),
                "op_soft_probs": torch.cat(op_step_soft_probs, dim=0),
                "op_t0_flats": t0_op_flats,
                "curr_op": current_op.detach(),
                "rel_acc": rel_acc,
                "halt_rel_prob": halt_rel_prob,
                "noop_prob": stack_noop_vec.detach(),
            }
            return (
                stem_logits,
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

        return stem_logits, morph_logits, edge_logits, len_logits, next_hidden, current_sym, act_steps_mean, vq_loss_total, ethical_loss, avg_stack


# ==========================================
# 7. Training, Macro-Packet Discovery, BPC Engine & 3-Phase Curriculum
# ==========================================
def save_checkpoint(model, path, macro_trie=None):
    if path is None:
        return
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'stem_vocab': tokenizer.stem_vocab,
        'morph_vocab': tokenizer.morph_vocab,
        'edge_vocab': tokenizer.edge_vocab,
        'macro_trie': macro_trie,
    }, path)


@torch.no_grad()
def crystallize_codebook_kmeans(vq_module, z_pool_list, n_entries, iters=15):
    if not z_pool_list:
        return
    pool = torch.cat(z_pool_list, dim=0).float()
    if pool.size(0) < 2:
        return

    if pool.size(0) >= n_entries:
        perm = torch.randperm(pool.size(0), device=pool.device)[:n_entries]
        centroids = pool[perm].clone()
        for _ in range(iters):
            dists = (
                (pool ** 2).sum(dim=-1, keepdim=True)
                + (centroids ** 2).sum(dim=-1).unsqueeze(0)
                - 2.0 * torch.matmul(pool, centroids.t())
            )
            assign = torch.argmin(dists, dim=-1)
            new_c = torch.zeros_like(centroids)
            counts = torch.zeros(n_entries, 1, device=pool.device, dtype=pool.dtype)
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
        r_idx = torch.randint(0, pool.size(0), (n_entries,), device=pool.device)
        centroids = pool[r_idx] + 0.05 * torch.randn(n_entries, pool.size(1), device=pool.device)
        rms = torch.sqrt(torch.mean(centroids ** 2, dim=-1, keepdim=True) + 1e-4)
        centroids = centroids / rms

    vq_module.codebook.data.copy_(centroids.to(vq_module.codebook.dtype))
    vq_module.normalize_codebook_weights_()
    vq_module.usage_count.zero_()
    vq_module.ema_usage.fill_(1.0 / n_entries)


@torch.no_grad()
def discover_macro_trie(model, atomic_dict, max_phrases=None, max_pkt_len=4):
    """
    Self-Evolving VQ/ACT Chunking:
      Scans atomic corpus with trained Phase-2 model to find recurring 2..4-unit spans
      that exhibit low ACT hesitation, high Stack NoOp stability, and high symbolic transition
      certainty without crossing sentence/continuation boundaries.
    """
    n_total = len(atomic_dict["stems"])
    if n_total < 128:
        return None, []

    was_training = model.training
    model.eval()

    if max_phrases is None:
        max_phrases = int(CONFIG.get("max_macro_phrases", 256))
    scan_cap = int(CONFIG.get("macro_scan_cap", 262144))
    n_scan = min(n_total - 1, scan_cap)
    b_scan = (256 if (HIGH_VRAM and n_scan >= 2048) else 128) if n_scan >= 512 else 1
    s_scan = n_scan // b_scan
    usable = b_scan * s_scan

    st_mat = atomic_dict["stems"][:usable].view(b_scan, s_scan).to(DEVICE)
    mo_mat = atomic_dict["morphs"][:usable].view(b_scan, s_scan).to(DEVICE)
    ed_mat = atomic_dict["edges"][:usable].view(b_scan, s_scan).to(DEVICE)

    acts_mat = torch.zeros(b_scan, s_scan, dtype=torch.float32)
    relp_mat = torch.zeros(b_scan, s_scan, dtype=torch.float32)
    noops_mat = torch.zeros(b_scan, s_scan, dtype=torch.float32)

    cached_cb = model.vq_layer.normalized_codebook()
    cached_padj = F.softmax(model.vq_layer.adjacency.float(), dim=-1)
    hidden, prev_sym = None, None

    with torch.amp.autocast(AMP_DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
        for t in range(s_scan):
            x_s = st_mat[:, t].unsqueeze(1)
            x_m = mo_mat[:, t].unsqueeze(1)
            x_l = torch.ones(b_scan, dtype=torch.long, device=DEVICE)
            curr_edge = ed_mat[:, t]
            out = model(
                x_s,
                x_m,
                x_l,
                prev_edge=curr_edge,
                hidden=hidden,
                prev_sym=prev_sym,
                decode_slots=1,
                return_stats=True,
                use_graph_echo=False,
                cached_P_adj=cached_padj,
                cached_cb_norm=cached_cb,
            )
            hidden = out[1]
            prev_sym = out[2]
            acts_mat[:, t] = out[11].cpu()
            relp_mat[:, t] = out[17]["halt_rel_prob"].cpu()
            noops_mat[:, t] = out[17]["noop_prob"].cpu()

    if was_training:
        model.train()

    st_cpu = st_mat.cpu().tolist()
    mo_cpu = mo_mat.cpu().tolist()
    ed_cpu = ed_mat.cpu().tolist()
    acts_cpu = acts_mat.tolist()
    relp_cpu = relp_mat.tolist()
    noops_cpu = noops_mat.tolist()

    blocked_stems = {tokenizer.pad_id, tokenizer.unk_id, tokenizer.bos_id, tokenizer.sep_id}
    blocked_edges = set(tokenizer.sentence_end_edges) | {tokenizer.edge_cont_id, tokenizer.edge_pad_id, tokenizer.edge_unk_id}

    phrase_stats = defaultdict(lambda: [0, 0.0, 0.0, 0.0])  # [count, sum_act, sum_relp, sum_noop]
    for b in range(b_scan):
        s_row, m_row, e_row = st_cpu[b], mo_cpu[b], ed_cpu[b]
        a_row, r_row, n_row = acts_cpu[b], relp_cpu[b], noops_cpu[b]
        for i in range(s_scan - 1):
            if s_row[i] in blocked_stems or e_row[i] in blocked_edges:
                continue
            key_list = [(s_row[i], m_row[i], e_row[i])]
            act_acc = a_row[i]
            noop_acc = n_row[i]
            rel_acc = 0.0
            for L in range(2, min(max_pkt_len, s_scan - i) + 1):
                j = i + L - 1
                if s_row[j] in blocked_stems or e_row[j] == tokenizer.edge_cont_id:
                    break
                key_list.append((s_row[j], m_row[j], e_row[j]))
                act_acc += a_row[j]
                noop_acc += n_row[j]
                rel_acc += r_row[j]
                if L >= 2:
                    st_entry = phrase_stats[tuple(key_list)]
                    st_entry[0] += 1
                    st_entry[1] += act_acc / float(L)
                    st_entry[2] += rel_acc / float(L - 1)
                    st_entry[3] += noop_acc / float(L)
                if e_row[j] in blocked_edges:
                    break

    base_min_freq = int(CONFIG.get("macro_min_freq", 6))
    min_freq = max(base_min_freq, n_scan // 16384)
    max_act = float(CONFIG.get("macro_max_act", 2.40))
    cfg_min_noop = float(CONFIG.get("macro_min_noop", 0.30))
    mean_corpus_noop = float(noops_mat.mean().item())
    min_noop = min(cfg_min_noop, max(0.18, 0.85 * mean_corpus_noop))
    min_rel = float(CONFIG.get("macro_min_rel_prob", 0.08))

    scored = []
    for tup, (cnt, sum_a, sum_r, sum_n) in phrase_stats.items():
        if cnt < min_freq:
            continue
        mean_a = sum_a / cnt
        mean_r = sum_r / cnt
        mean_n = sum_n / cnt
        if mean_a <= max_act and mean_n >= min_noop and mean_r >= min_rel:
            score = cnt * (len(tup) - 1) * (0.2 + mean_r) * (0.5 + mean_n) / max(1.0, mean_a)
            scored.append((score, cnt, mean_a, mean_r, tup))

    # Adaptive fallback if corpus thresholds were slightly strict
    if len(scored) < 16:
        scored = []
        for tup, (cnt, sum_a, sum_r, sum_n) in phrase_stats.items():
            if cnt >= max(3, min_freq // 2):
                mean_a = sum_a / cnt
                mean_r = sum_r / cnt
                mean_n = sum_n / cnt
                score = cnt * (len(tup) - 1) * (0.1 + mean_r) * (0.5 + mean_n) / max(1.0, mean_a)
                scored.append((score, cnt, mean_a, mean_r, tup))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_selected = scored[:max_phrases]
    if not top_selected:
        return None, []

    trie = {}
    for _, _, _, _, tup in top_selected:
        node = trie
        for unit in tup:
            if unit not in node:
                node[unit] = {}
            node = node[unit]
        node["__end__"] = True

    preview_strs = [
        ("".join(tokenizer.decode_word(u[0], u[1]) + tokenizer.decode_edge(u[2]) for u in tup).strip(), cnt, ma, mr)
        for _, cnt, ma, mr, tup in top_selected[:8]
    ]
    return trie, (preview_strs, len(top_selected), n_scan)


def build_epoch_streams(pkt_dict, batch_size, epoch_cap, stream_len_override=None):
    stems_cpu = pkt_dict["stems"]
    morphs_cpu = pkt_dict["morphs"]
    edges_cpu = pkt_dict["edges"]
    lens_cpu = pkt_dict["lens"]
    chars_cpu = pkt_dict["chars"]

    n_steps = len(stems_cpu) - 1
    if stream_len_override is not None:
        stream_len = max(1, min(n_steps, stream_len_override))
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

    span = stream_len + 1
    return {
        "stems": torch.stack([stems_cpu[off : off + span] for off in offsets], dim=0).to(DEVICE, non_blocking=True),
        "morphs": torch.stack([morphs_cpu[off : off + span] for off in offsets], dim=0).to(DEVICE, non_blocking=True),
        "edges": torch.stack([edges_cpu[off : off + span] for off in offsets], dim=0).to(DEVICE, non_blocking=True),
        "lens": torch.stack([lens_cpu[off : off + span] for off in offsets], dim=0).to(DEVICE, non_blocking=True),
        "chars": torch.stack([chars_cpu[off : off + span] for off in offsets], dim=0).to(DEVICE, non_blocking=True),
    }


def build_val_batches(pkt_slice_list):
    """Builds validation chunks strictly within contiguous packet slices."""
    if not pkt_slice_list:
        return []

    all_chunks = []
    target_len = 64
    for sl in pkt_slice_list:
        n_val = len(sl["stems"]) - 1
        if n_val < 4:
            continue
        s_len = min(target_len, n_val)
        n_chunks = max(1, n_val // s_len)
        for c in range(n_chunks):
            a, b = c * s_len, c * s_len + s_len + 1
            all_chunks.append((
                s_len,
                sl["stems"][a:b],
                sl["morphs"][a:b],
                sl["edges"][a:b],
                sl["lens"][a:b],
                sl["chars"][a:b],
            ))

    if not all_chunks:
        return []

    by_len = defaultdict(list)
    for item in all_chunks:
        by_len[item[0]].append(item[1:])

    batches = []
    for s_len, chunks in by_len.items():
        for start_c in range(0, len(chunks), 64):
            sub = chunks[start_c : start_c + 64]
            batches.append({
                "stems": torch.stack([c[0] for c in sub], dim=0).to(DEVICE),
                "morphs": torch.stack([c[1] for c in sub], dim=0).to(DEVICE),
                "edges": torch.stack([c[2] for c in sub], dim=0).to(DEVICE),
                "lens": torch.stack([c[3] for c in sub], dim=0).to(DEVICE),
                "chars": torch.stack([c[4] for c in sub], dim=0).to(DEVICE),
            })
    return batches


def compute_packet_loss_and_bpc(
    stem_logits,
    morph_logits,
    edge_logits,
    len_logits,
    tgt_stems,
    tgt_morphs,
    tgt_edges,
    tgt_lens,
    tgt_chars,
    macro_active=False,
    label_smoothing=0.0,
):
    B, S_max, V_s = stem_logits.shape
    ts = tgt_stems[:, :S_max]
    tm = tgt_morphs[:, :S_max]
    te = tgt_edges[:, :S_max]
    mask = (ts > 0)
    n_slots = mask.sum().clamp(min=1)

    log_probs_s = F.log_softmax(stem_logits.reshape(-1, V_s).float(), dim=-1)
    ce_s = F.nll_loss(log_probs_s, ts.reshape(-1), reduction="none").view(B, S_max)
    if label_smoothing > 0.0:
        smooth_s = -log_probs_s.mean(dim=-1).view(B, S_max)
        ce_s_opt = (1.0 - label_smoothing) * ce_s + label_smoothing * smooth_s
    else:
        ce_s_opt = ce_s

    ce_m = F.cross_entropy(morph_logits.reshape(-1, morph_logits.size(-1)).float(), tm.reshape(-1), reduction="none").view(B, S_max)
    ce_e = F.cross_entropy(edge_logits.reshape(-1, edge_logits.size(-1)).float(), te.reshape(-1), reduction="none").view(B, S_max)

    loss_stem = (ce_s_opt * mask).sum() / n_slots
    loss_morph = (ce_m * mask).sum() / n_slots
    loss_edge = (ce_e * mask).sum() / n_slots

    if macro_active:
        ce_len = F.cross_entropy(len_logits.float(), (tgt_lens - 1).clamp(0, len_logits.size(-1) - 1), reduction="none")
        loss_len = ce_len.mean()
        total_nll_nats = ((ce_s + ce_m + ce_e) * mask).sum() + ce_len.sum()
    else:
        loss_len = torch.zeros((), device=stem_logits.device, dtype=torch.float32)
        total_nll_nats = ((ce_s + ce_m + ce_e) * mask).sum()

    loss_pred = loss_stem + loss_morph + loss_edge + loss_len

    s_ok = ((stem_logits.detach().argmax(dim=-1) == ts) & mask).sum()
    m_ok = ((morph_logits.detach().argmax(dim=-1) == tm) & mask).sum()
    e_ok = ((edge_logits.detach().argmax(dim=-1) == te) & mask).sum()
    chars_sum = tgt_chars.sum().clamp(min=1.0)

    return loss_pred, total_nll_nats.detach(), chars_sum.detach(), s_ok, m_ok, e_ok, mask.sum()


@torch.no_grad()
def evaluate_val(model, val_batches, macro_active=False, **fwd_kwargs):
    if not val_batches:
        return None, None, None, None, None
    model.eval()
    total_loss_t = torch.zeros((), device=DEVICE)
    total_nll_t = torch.zeros((), device=DEVICE)
    total_chars_t = torch.zeros((), device=DEVICE)
    stem_ok_t = torch.zeros((), dtype=torch.long, device=DEVICE)
    morph_ok_t = torch.zeros((), dtype=torch.long, device=DEVICE)
    edge_ok_t = torch.zeros((), dtype=torch.long, device=DEVICE)
    total_slots = 0
    total_steps = 0

    cached_cb = model.vq_layer.normalized_codebook()
    cached_padj = F.softmax(model.vq_layer.adjacency.float(), dim=-1)

    with torch.amp.autocast(AMP_DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
        for batch in val_batches:
            b_curr = batch["stems"].size(0)
            s_len = batch["stems"].size(1) - 1
            hidden, prev_sym = None, None

            for t in range(s_len):
                in_s = batch["stems"][:, t]
                in_m = batch["morphs"][:, t]
                in_e = batch["edges"][:, t]
                in_l = batch["lens"][:, t]

                tgt_s = batch["stems"][:, t + 1]
                tgt_m = batch["morphs"][:, t + 1]
                tgt_e = batch["edges"][:, t + 1]
                tgt_l = batch["lens"][:, t + 1]
                tgt_c = batch["chars"][:, t + 1]

                last_idx = (in_l - 1).clamp(min=0, max=in_e.size(1) - 1)
                curr_edge = in_e.gather(1, last_idx.unsqueeze(1)).squeeze(1)

                dec_slots = int(tgt_l.max().item()) if macro_active else 1
                s_logits, m_logits, e_logits, len_logits, hidden, prev_sym, _, _, _, _ = model(
                    in_s[:, :int(in_l.max().item())],
                    in_m[:, :int(in_l.max().item())],
                    in_l,
                    prev_edge=curr_edge,
                    hidden=hidden,
                    prev_sym=prev_sym,
                    tgt_stems=tgt_s,
                    tgt_morphs=tgt_m,
                    decode_slots=dec_slots,
                    cached_P_adj=cached_padj,
                    cached_cb_norm=cached_cb,
                    **fwd_kwargs,
                )

                loss_p, nll_p, ch_p, s_ok, m_ok, e_ok, n_sl = compute_packet_loss_and_bpc(
                    s_logits, m_logits, e_logits, len_logits,
                    tgt_s, tgt_m, tgt_e, tgt_l, tgt_c,
                    macro_active=macro_active,
                )
                total_loss_t += loss_p * b_curr
                total_nll_t += nll_p
                total_chars_t += ch_p
                stem_ok_t += s_ok
                morph_ok_t += m_ok
                edge_ok_t += e_ok
                total_slots += int(n_sl.item())
                total_steps += b_curr

    model.train()
    val_loss = total_loss_t.item() / max(1, total_steps)
    val_bpc = total_nll_t.item() / max(1e-6, total_chars_t.item() * math.log(2.0))
    s_acc = 100.0 * stem_ok_t.item() / max(1, total_slots)
    m_acc = 100.0 * morph_ok_t.item() / max(1, total_slots)
    e_acc = 100.0 * edge_ok_t.item() / max(1, total_slots)
    return val_loss, val_bpc, s_acc, m_acc, e_acc


def sample_next_token(logits, recent_ids=None, temperature=0.75, top_k=40, top_p=0.90, rep_penalty=1.15, forbid_pad=True):
    """Top-K / Nucleus (Top-P) sampler with optional PAD masking and repetition penalty."""
    scores = logits.squeeze(0).float().clone()
    if scores.dim() > 1:
        scores = scores[0]
    if forbid_pad and scores.size(-1) > 1:
        scores[0] = -float("inf")
    if recent_ids and rep_penalty > 1.0:
        for tid in set(recent_ids[-24:]):
            if tid < scores.size(-1):
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
        return int(torch.argmax(scores).item())
    return int(torch.multinomial(probs, 1).item())


def sample_edge_token(ej_logits, stem_id, temperature=0.35, top_k=6, top_p=0.85):
    """Samples boundary edge cleanly: forbids <PAD>/<EDGE_UNK> and avoids gluing multi-char words."""
    scores = ej_logits.squeeze(0).float().clone()
    if scores.dim() > 1:
        scores = scores[0]
    scores[tokenizer.edge_pad_id] = -float("inf")
    if tokenizer.edge_unk_id < scores.size(-1):
        scores[tokenizer.edge_unk_id] = -float("inf")
    stem_str = tokenizer.id_to_stem.get(int(stem_id), "")
    if int(stem_id) >= 4 and len(stem_str) >= 2 and not stem_str.isdigit():
        if tokenizer.edge_empty_id < scores.size(-1):
            scores[tokenizer.edge_empty_id] -= 4.0
        if tokenizer.edge_cont_id < scores.size(-1):
            scores[tokenizer.edge_cont_id] -= 2.5
    return sample_next_token(
        scores.unsqueeze(0), None, temperature=temperature, top_k=top_k, top_p=top_p,
        rep_penalty=1.0, forbid_pad=False
    )


@torch.no_grad()
def generate_sample_preview(model, macro_active=False, n_steps_gen=None):
    was_training = model.training
    model.eval()
    if n_steps_gen is None:
        n_steps_gen = max(24, min(48, int(round(65 / max(1.0, CONFIG["comp_ratio"])))))

    with torch.amp.autocast(AMP_DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
        s0 = data_atomic["stems"][0].item()
        m0 = data_atomic["morphs"][0].item()
        e0 = data_atomic["edges"][0].item()
        gen_pkts = [[(s0, m0, e0)]]
        recent_stems = [s0]

        curr_s = torch.tensor([[s0]], device=DEVICE, dtype=torch.long)
        curr_m = torch.tensor([[m0]], device=DEVICE, dtype=torch.long)
        curr_l = torch.tensor([1], device=DEVICE, dtype=torch.long)
        curr_edge = torch.tensor([e0], device=DEVICE, dtype=torch.long)
        gh, gp = None, None

        for _ in range(n_steps_gen):
            out = model(
                curr_s, curr_m, curr_l, prev_edge=curr_edge, hidden=gh, prev_sym=gp,
                decode_slots=1, return_stats=True
            )
            gh, gp = out[1], out[2]
            extra = out[17]
            z_wr, z_wi = extra["z_w_r"], extra["z_w_i"]

            if macro_active:
                len_probs = F.softmax(extra["len_logits"].squeeze(0).float() / 0.85, dim=-1)
                next_L = int(torch.multinomial(len_probs, 1).item()) + 1
            else:
                next_L = 1

            pkt_tuples = []
            slot_s_list, slot_m_list = [], []
            prev_flat = None
            for j in range(next_L):
                feat_j = model.decode_slot_features(z_wr, z_wi, j, prev_slot_flat=prev_flat)
                sj_logits = model.decoder(feat_j)
                mj_logits = model.morph_decoder(feat_j)
                ej_logits = model.edge_decoder(feat_j)

                sj = sample_next_token(sj_logits, recent_stems, temperature=0.75, top_k=40, top_p=0.90, rep_penalty=1.15)
                mj = sample_next_token(mj_logits, None, temperature=0.40, top_k=8, top_p=0.85, rep_penalty=1.0)
                ej = sample_edge_token(ej_logits, sj, temperature=0.35, top_k=6, top_p=0.85)

                pkt_tuples.append((sj, mj, ej))
                slot_s_list.append(sj)
                slot_m_list.append(mj)
                recent_stems.append(sj)

                if j + 1 < next_L:
                    pw_r, pw_i = model.embed_polar_words(
                        torch.tensor([sj], device=DEVICE), torch.tensor([mj], device=DEVICE)
                    )
                    prev_flat = torch.cat([pw_r, pw_i], dim=-1)

            gen_pkts.append(pkt_tuples)
            curr_s = torch.tensor([slot_s_list], device=DEVICE, dtype=torch.long)
            curr_m = torch.tensor([slot_m_list], device=DEVICE, dtype=torch.long)
            curr_l = torch.tensor([next_L], device=DEVICE, dtype=torch.long)
            curr_edge = torch.tensor([pkt_tuples[-1][2]], device=DEVICE, dtype=torch.long)

    if was_training:
        model.train()
    return tokenizer.decode_packets(gen_pkts)


def train(save_path="crsn_omni_model.pth"):
    model = UberCRSN(
        vocab_size,
        CONFIG["embedding_dim"],
        n_morph_classes=n_morphs,
        n_edge_classes=n_edges,
    ).to(DEVICE)

    train_pkts = pack_atomic_to_packets(data_atomic, macro_trie=None, max_pkt_len=CONFIG["max_pkt_len"])
    val_pkts_atomic = [pack_atomic_to_packets(va, macro_trie=None, max_pkt_len=CONFIG["max_pkt_len"]) for va in val_atomic_slices]
    val_batches = build_val_batches(val_pkts_atomic)

    sym_scale = math.sqrt(max(1.0, CONFIG["n_symbols"] / 64.0))
    cb_lr = CONFIG["learning_rate"] * min(1.5, sym_scale)
    adj_lr = CONFIG["learning_rate"] * float(CONFIG.get("adj_lr_mult", 8.0)) * min(2.0, sym_scale)

    codebook_params = [model.vq_layer.codebook, model.op_vq.codebook]
    adj_params = [model.vq_layer.adjacency]
    excluded_names = {"vq_layer.codebook", "op_vq.codebook", "vq_layer.adjacency"}
    other_params = [p for n, p in model.named_parameters() if n not in excluded_names]
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

    mod_param_groups = {"Cell": [], "Att": [], "FFN": [], "VQ": [], "Op": [], "Adj": [], "Holo": [], "Dec": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "cell.attention" in name:
            mod_param_groups["Att"].append(p)
        elif "cell.ffn" in name:
            mod_param_groups["FFN"].append(p)
        elif name.startswith("cell."):
            mod_param_groups["Cell"].append(p)
        elif "vq_layer.codebook" in name:
            mod_param_groups["VQ"].append(p)
        elif "op_vq" in name:
            mod_param_groups["Op"].append(p)
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
    lr_warmup_epochs = max(3, min(15, CONFIG["epochs"] // 30))
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=CONFIG["lr_patience"],
        min_lr=min_lrs,
    )

    log_K = float(math.log(CONFIG["n_symbols"]))
    log_K_op = float(math.log(max(2, CONFIG["n_ops"])))
    ppx_low = CONFIG["target_ppx_low"]
    ppx_high = CONFIG["target_ppx_high"]

    phase1_end = max(5, int(CONFIG["phase1_frac"] * CONFIG["epochs"]))
    phase2_end = max(phase1_end + 5, int(CONFIG["phase2_frac"] * CONFIG["epochs"]))
    eval_every = CONFIG.get("eval_every", 5)
    clip_max = float(CONFIG.get("grad_clip_max", 3.0))

    history = {
        "epoch": [],
        "train_pred": [],
        "train_bpc": [],
        "val_loss": [],
        "val_bpc": [],
        "gen_gap": [],
        "train_acc": [],
        "val_acc": [],
        "sym_lm_loss": [],
        "sym_lm_acc": [],
        "rel_acc": [],
        "act_mean": [],
        "act_std": [],
        "act_dist": [],
        "sym_ppx": [],
        "op_ppx": [],
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
        "gm_op": [],
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
    op_revived_total = 0
    skipped_spikes_since_log = 0
    last_ep_revived = 0
    min_active_converge = max(4, min(16, CONFIG["n_symbols"] // 4))
    best_metric = float("inf")
    best_epoch = -1
    best_state = None
    dirty_best_disk = False
    no_improve_logs = 0
    improved_since_log = False
    improved_since_sched = False

    macro_trie = None
    macro_active = False

    live_div_weight = CONFIG["diversity_weight"]
    live_commit_cost = CONFIG["commitment_cost"]
    live_ponder_penalty = CONFIG["ponder_penalty"]
    ema_gnorm = CONFIG["grad_clip"]
    readiness = 0.0
    ema_bpc = None

    act_low = 1.40
    act_high = max(2.60, 0.80 * CONFIG["max_recursion_depth"])

    revive_accum_counts = torch.zeros(CONFIG["n_symbols"], device=DEVICE)
    op_revive_accum_counts = torch.zeros(CONFIG["n_ops"], device=DEVICE)
    revive_lookups_accum = 0
    min_revive_lookups = max(512, 8 * CONFIG["n_symbols"])
    epoch_z_pool = []
    epoch_op_pool = []
    phase1_z_pool = []
    phase1_op_pool = []
    units_since_log = 0
    clipped_chunks_since_log = 0
    total_chunks_since_log = 0

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"--- Training SACRSN v60 (BF16_AMP={USE_AMP} | FusedAdamW={fused_used} | TBPTT={CONFIG['seq_len']}) ---")

    try:
        for epoch in range(CONFIG["epochs"]):
            t_ep_train_0 = time.time()
            model.train()

            if epoch == phase1_end:
                crystallize_codebook_kmeans(model.vq_layer, phase1_z_pool, CONFIG["n_symbols"])
                crystallize_codebook_kmeans(model.op_vq, phase1_op_pool, CONFIG["n_ops"])
                phase1_z_pool.clear()
                phase1_op_pool.clear()
                revive_accum_counts.zero_()
                op_revive_accum_counts.zero_()
                revive_lookups_accum = 0
                print(f"[Phase 2 Crystallization @ Ep {epoch:04d}] Initialized VQ & RelOp codebooks via GPU K-Means.")
            elif epoch == phase2_end:
                if CONFIG.get("use_macro_packets", True):
                    macro_trie, macro_meta = discover_macro_trie(
                        model,
                        data_atomic,
                        max_phrases=CONFIG.get("max_macro_phrases", 256),
                        max_pkt_len=CONFIG.get("max_pkt_len", 4),
                    )
                    if macro_trie:
                        preview_phrases, n_macro_sel, n_macro_scan = macro_meta
                        train_pkts = pack_atomic_to_packets(data_atomic, macro_trie=macro_trie, max_pkt_len=CONFIG["max_pkt_len"])
                        if val_atomic_slices:
                            val_pkts_macro = [pack_atomic_to_packets(va, macro_trie=macro_trie, max_pkt_len=CONFIG["max_pkt_len"]) for va in val_atomic_slices]
                            val_batches = build_val_batches(val_pkts_macro)
                        macro_active = True
                        mean_u_per_pkt = float(train_pkts["lens"].float().mean().item())
                        pct_multi_pkt = float((train_pkts["lens"] > 1).float().mean().item() * 100.0)
                        best_metric = float("inf")
                        best_epoch = -1
                        best_state = None
                        ema_bpc = None
                        sample_p = ", ".join(f"'{p}'(n={c},ACT={a:.2f})" for p, c, a, _ in preview_phrases[:4])
                        print(
                            f"[Phase 3 Macro-Packets @ Ep {epoch:04d}] Mined {n_macro_sel} phrases (scan={n_macro_scan:,}) -> "
                            f"{mean_u_per_pkt:.2f} units/pkt ({pct_multi_pkt:.1f}% multi) | Top: [{sample_p}]"
                        )
                    else:
                        print(f"[Phase 3 Reasoning @ Ep {epoch:04d}] No phrases passed macro threshold; continuing at L=1.")
                else:
                    print(f"[Phase 3 Reasoning @ Ep {epoch:04d}] Unlocked Multi-Hop Graph Echo & Relational Rules (L=1).")

            if epoch < phase1_end:
                phase_id = 1
                model.sym_alpha.fill_(0.0)
                model.graph_gate.fill_(0.0)
                vq_scale = 0.25
                op_vq_scale = 0.25
                sym_lm_scale = 0.05
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
                vq_scale = CONFIG["vq_weight"]
                op_vq_scale = CONFIG["op_vq_weight"]
                sym_lm_scale = CONFIG["sym_lm_weight"] * p2_prog
                div_scale_phase = 1.0
                cons_weight_live = CONFIG["symbol_consistency_weight"] * p2_prog
                live_eth_weight = 0.35 * CONFIG["ethical_weight"] * (0.3 + 0.7 * p2_prog)
                live_bias_scale = 0.0
                live_conscience = float(CONFIG["conscience_p2"])
            else:
                phase_id = 3
                p3_ramp = min(1.0, (epoch - phase2_end + 1) / 25.0)
                model.graph_gate.fill_(p3_ramp)
                vq_scale = CONFIG["vq_weight"]
                op_vq_scale = CONFIG["op_vq_weight"]
                sym_lm_scale = CONFIG["sym_lm_weight"]
                div_scale_phase = 1.0
                cons_weight_live = CONFIG["symbol_consistency_weight"]
                live_eth_weight = CONFIG["ethical_weight"] * max(0.35, readiness)
                live_bias_scale = float(torch.clamp(model.vq_layer.ema_d_std * 0.30, min=0.02, max=0.20).item()) * readiness
                live_conscience = float(CONFIG["conscience_p3"])

            if epoch < lr_warmup_epochs:
                lr_scale = (epoch + 1) / float(lr_warmup_epochs)
                for pg, blr in zip(opt.param_groups, base_lrs):
                    pg["lr"] = blr * lr_scale

            # Adjust stream_len cleanly if macro packing shortened packet corpus
            eff_stream_len = min(CONFIG["stream_len"], max(4, (len(train_pkts["stems"]) - 1) // max(1, CONFIG["batch_size"]))) if CONFIG.get("stream_len") is not None else None
            streams = build_epoch_streams(
                train_pkts,
                CONFIG["batch_size"],
                CONFIG["epoch_token_cap"],
                stream_len_override=eff_stream_len,
            )
            bs, span_len = streams["stems"].shape[:2]
            stream_len = span_len - 1

            hidden = None
            prev_sym = None

            total_loss_t = torch.zeros((), device=DEVICE)
            total_pred_t = torch.zeros((), device=DEVICE)
            total_nll_t = torch.zeros((), device=DEVICE)
            total_chars_t = torch.zeros((), device=DEVICE)
            total_sym_lm_t = torch.zeros((), device=DEVICE)
            total_vq_t = torch.zeros((), device=DEVICE)
            total_vq_cos_t = torch.zeros((), device=DEVICE)
            total_eth_t = torch.zeros((), device=DEVICE)
            total_cons_t = torch.zeros((), device=DEVICE)
            total_div_t = torch.zeros((), device=DEVICE)
            total_stack_t = torch.zeros((), device=DEVICE)
            total_ptr_ent_t = torch.zeros((), device=DEVICE)
            total_mem_occ_t = torch.zeros((), device=DEVICE)
            total_rel_acc_t = torch.zeros((), device=DEVICE)
            rel_acc_steps = 0

            total_w_pred_t = torch.zeros((), device=DEVICE)
            total_w_symlm_t = torch.zeros((), device=DEVICE)
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

            stem_ok_t = torch.zeros((), dtype=torch.long, device=DEVICE)
            morph_ok_t = torch.zeros((), dtype=torch.long, device=DEVICE)
            edge_ok_t = torch.zeros((), dtype=torch.long, device=DEVICE)
            total_slots_t = torch.zeros((), dtype=torch.long, device=DEVICE)
            total_sym_correct_t = torch.zeros((), dtype=torch.long, device=DEVICE)
            self_loops_t = torch.zeros((), dtype=torch.long, device=DEVICE)
            act_step_hist = torch.zeros(CONFIG["max_recursion_depth"], dtype=torch.long, device=DEVICE)
            act_steps_sum_t = torch.zeros((), device=DEVICE, dtype=torch.float32)
            act_steps_sq_sum_t = torch.zeros((), device=DEVICE, dtype=torch.float32)
            trans_toks = 0
            total_pkts = 0
            grad_norms = []

            progress = epoch / max(1, CONFIG["epochs"] - 1)
            entropy_weight = 0.01 * (1.0 - progress)
            in_revive_phase = (phase_id >= 2) and (epoch < int(0.60 * CONFIG["epochs"]))
            use_echo = (phase_id == 3)

            chunk_loss = 0.0
            chunk_probs = []
            chunk_op_probs = []
            chunk_weights = []
            steps_in_chunk = 0
            chunk_cb_norm = None
            chunk_P_adj = None

            for t in range(stream_len):
                if steps_in_chunk == 0:
                    chunk_cb_norm = model.vq_layer.normalized_codebook()
                    chunk_P_adj = F.softmax(model.vq_layer.adjacency.float(), dim=-1) if use_echo else None

                in_s = streams["stems"][:, t]
                in_m = streams["morphs"][:, t]
                in_e = streams["edges"][:, t]
                in_l = streams["lens"][:, t]

                tgt_s = streams["stems"][:, t + 1]
                tgt_m = streams["morphs"][:, t + 1]
                tgt_e = streams["edges"][:, t + 1]
                tgt_l = streams["lens"][:, t + 1]
                tgt_c = streams["chars"][:, t + 1]

                last_idx = (in_l - 1).clamp(min=0, max=in_e.size(1) - 1)
                curr_edge = in_e.gather(1, last_idx.unsqueeze(1)).squeeze(1)

                in_max_l = int(in_l.max().item()) if macro_active else 1
                dec_slots = int(tgt_l.max().item()) if macro_active else 1

                with torch.amp.autocast(AMP_DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
                    (
                        stem_logits,
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
                        in_s[:, :in_max_l],
                        in_m[:, :in_max_l],
                        in_l,
                        prev_edge=curr_edge,
                        hidden=hidden,
                        prev_sym=prev_sym,
                        tgt_stems=tgt_s,
                        tgt_morphs=tgt_m,
                        decode_slots=dec_slots,
                        bias_scale=live_bias_scale,
                        commit_cost=live_commit_cost,
                        return_stats=True,
                        use_graph_echo=use_echo,
                        cached_P_adj=chunk_P_adj,
                        cached_cb_norm=chunk_cb_norm,
                        conscience_scale=live_conscience,
                    )

                    morph_logits = extra_stats["morph_logits"]
                    edge_logits = extra_stats["edge_logits"]
                    len_logits = extra_stats["len_logits"]

                    loss_pred, nll_step, ch_step, s_ok, m_ok, e_ok, n_sl = compute_packet_loss_and_bpc(
                        stem_logits, morph_logits, edge_logits, len_logits,
                        tgt_s, tgt_m, tgt_e, tgt_l, tgt_c,
                        macro_active=macro_active,
                        label_smoothing=float(CONFIG.get("label_smoothing", 0.0)),
                    )

                    sym_logits = extra_stats["sym_logits"]
                    loss_sym_lm_raw = F.cross_entropy(sym_logits.float(), tgt_s[:, 0])
                    loss_sym_lm = sym_lm_scale * loss_sym_lm_raw

                    loss_ponder = live_ponder_penalty * ponder_loss

                    probs_s0 = F.softmax(stem_logits[:, 0, :].float(), dim=-1)
                    loss_entropy = -entropy_weight * (-(probs_s0 * torch.log(probs_s0 + 1e-8)).sum(dim=-1).mean())

                    op_vq_loss = extra_stats["op_vq_loss"]
                    loss_vq = vq_scale * vq_loss + op_vq_scale * op_vq_loss
                    loss_ethics = live_eth_weight * eth_loss
                    loss_cons = cons_weight_live * cons_loss

                    pop_bal_mult = float(CONFIG.get("stack_pop_balance_mult", 3.0))
                    loss_stk_bal = CONFIG["stack_balance_weight"] * (
                        pop_bal_mult * ((avg_stk_ctrl[0] - avg_stk_ctrl[1]) ** 2)
                        + 1.0 * (F.relu(avg_stk_ctrl[2] - 0.72) ** 2)
                    )

                    step_loss = (
                        loss_pred
                        + loss_sym_lm
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
                    total_rel_acc_t += extra_stats["rel_acc"]
                    rel_acc_steps += 1

                prev_sym = sym_idx.detach()

                chunk_loss = chunk_loss + step_loss
                chunk_probs.append(soft_probs)
                chunk_op_probs.append(extra_stats["op_soft_probs"])
                chunk_weights.append(active_w)

                if phase_id == 1 and t % 4 == 0:
                    phase1_z_pool.append(t0_z_flats[:128])
                    phase1_op_pool.append(extra_stats["op_t0_flats"][:128])
                    if len(phase1_z_pool) > 32:
                        phase1_z_pool.pop(0)
                        phase1_op_pool.pop(0)
                elif in_revive_phase and len(epoch_z_pool) < 64:
                    epoch_z_pool.append(t0_z_flats)
                    epoch_op_pool.append(extra_stats["op_t0_flats"])

                steps_in_chunk += 1

                total_pred_t += loss_pred.detach()
                total_nll_t += nll_step
                total_chars_t += ch_step
                total_sym_lm_t += loss_sym_lm_raw.detach()
                total_vq_t += vq_loss.detach()
                total_vq_cos_t += vq_cos
                total_eth_t += eth_loss.detach()
                total_cons_t += cons_loss.detach()
                total_stack_t += avg_stack.detach()
                total_ptr_ent_t += ptr_ent
                total_mem_occ_t += mem_occ_pct

                total_w_pred_t += loss_pred.detach()
                total_w_symlm_t += loss_sym_lm.detach()
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

                stem_ok_t += s_ok
                morph_ok_t += m_ok
                edge_ok_t += e_ok
                total_slots_t += n_sl
                total_sym_correct_t += (sym_logits.detach().argmax(dim=-1) == tgt_s[:, 0]).sum()

                steps_idx = (steps_vec.long() - 1).clamp(min=0, max=CONFIG["max_recursion_depth"] - 1)
                act_step_hist.index_add_(0, steps_idx, torch.ones_like(steps_idx))
                act_steps_sum_t += steps_vec.sum()
                act_steps_sq_sum_t += (steps_vec ** 2).sum()

                total_pkts += bs
                units_since_log += int(n_sl.item())

                if steps_in_chunk == CONFIG["seq_len"] or t == stream_len - 1:
                    probs_cat = torch.cat(chunk_probs, dim=0)
                    op_probs_cat = torch.cat(chunk_op_probs, dim=0)
                    w_cat = torch.cat(chunk_weights, dim=0)
                    w_sum = w_cat.sum().clamp(min=1.0)
                    w_exp = w_cat.unsqueeze(-1)

                    avg_probs = (probs_cat * w_exp).sum(dim=0) / w_sum
                    marginal_entropy = -(avg_probs * torch.log(avg_probs + 1e-8)).sum()
                    norm_kl = (log_K - marginal_entropy) / max(1e-8, log_K)

                    avg_op_probs = (op_probs_cat * w_exp).sum(dim=0) / w_sum
                    op_marginal_entropy = -(avg_op_probs * torch.log(avg_op_probs + 1e-8)).sum()
                    op_norm_kl = (log_K_op - op_marginal_entropy) / max(1e-8, log_K_op)

                    loss_diversity = (
                        (div_scale_phase * live_div_weight) * norm_kl
                        + (div_scale_phase * CONFIG["op_diversity_weight"]) * op_norm_kl
                    )

                    total_chunk_loss = (chunk_loss / steps_in_chunk) + loss_diversity

                    clip_thresh = max(0.5, min(clip_max, 1.8 * ema_gnorm))
                    opt.zero_grad(set_to_none=True)
                    total_chunk_loss.backward()
                    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_thresh).item()

                    if math.isfinite(gnorm) and gnorm < 5000.0:
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
                        model.vq_layer.normalize_codebook_weights_()
                        model.op_vq.normalize_codebook_weights_()
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
                    chunk_op_probs = []
                    chunk_weights = []
                    steps_in_chunk = 0

            tr_time_since_log += time.time() - t_ep_train_0

            avg_pred = total_pred_t.item() / stream_len
            train_bpc = total_nll_t.item() / max(1e-6, total_chars_t.item() * math.log(2.0))
            avg_sym_lm = total_sym_lm_t.item() / stream_len
            avg_vq = total_vq_t.item() / stream_len
            avg_act_steps = act_steps_sum_t.item() / max(1, total_pkts)
            act_var = max(0.0, (act_steps_sq_sum_t.item() / max(1, total_pkts)) - (avg_act_steps ** 2))
            act_std = math.sqrt(act_var)
            ema_bpc = train_bpc if ema_bpc is None else (0.80 * ema_bpc + 0.20 * train_bpc)

            with torch.no_grad():
                counts = model.vq_layer.usage_count.clone()
                active_syms = int((counts > 0).sum().item())
                dist = counts / (counts.sum() + 1e-10)
                sym_drift = float((dist - model.prev_sym_soft).abs().sum().item() * 0.5)
                model.prev_sym_soft.mul_(0.85).add_(0.15 * dist)
                entropy_val = -(dist * torch.log(dist + 1e-10)).sum()
                true_ppx = torch.exp(entropy_val).item()
                top1_pct = float((counts.max() / (counts.sum() + 1e-10) * 100.0).item())

                op_counts = model.op_vq.usage_count.clone()
                active_ops = int((op_counts > 0).sum().item())
                op_dist = op_counts / (op_counts.sum() + 1e-10)
                op_ppx = torch.exp(-(op_dist * torch.log(op_dist + 1e-10)).sum()).item()

                revive_accum_counts.add_(counts)
                op_revive_accum_counts.add_(op_counts)
                revive_lookups_accum += int(counts.sum().item())

                ep_revived = 0
                if (
                    in_revive_phase
                    and epoch % CONFIG["revive_every"] == 0
                    and revive_lookups_accum >= min_revive_lookups
                    and len(epoch_z_pool) > 0
                ):
                    dead_mask = revive_accum_counts <= 1.0
                    n_dead = int(dead_mask.sum().item())
                    if n_dead > 0 and (true_ppx < ppx_low or n_dead > int(0.05 * CONFIG["n_symbols"])):
                        pool = torch.cat(epoch_z_pool, dim=0)
                        rand_idx = torch.randint(0, pool.size(0), (n_dead,), device=DEVICE)
                        new_vecs = pool[rand_idx] + 0.05 * torch.randn(n_dead, pool.size(1), device=DEVICE)
                        new_vecs = new_vecs / torch.sqrt(torch.mean(new_vecs ** 2, dim=-1, keepdim=True) + 1e-4)
                        model.vq_layer.codebook.data[dead_mask] = new_vecs
                        model.vq_layer.ema_usage.data[dead_mask] = 1.0 / float(CONFIG["n_symbols"])
                        ep_revived = n_dead
                        revived_total += n_dead

                    op_dead_mask = op_revive_accum_counts <= 1.0
                    n_op_dead = int(op_dead_mask.sum().item())
                    if n_op_dead > 0 and len(epoch_op_pool) > 0:
                        op_pool = torch.cat(epoch_op_pool, dim=0)
                        r_op_idx = torch.randint(0, op_pool.size(0), (n_op_dead,), device=DEVICE)
                        new_ops = op_pool[r_op_idx] + 0.05 * torch.randn(n_op_dead, op_pool.size(1), device=DEVICE)
                        new_ops = new_ops / torch.sqrt(torch.mean(new_ops ** 2, dim=-1, keepdim=True) + 1e-4)
                        model.op_vq.codebook.data[op_dead_mask] = new_ops
                        model.op_vq.ema_usage.data[op_dead_mask] = 1.0 / float(CONFIG["n_ops"])
                        op_revived_total += n_op_dead

                    revive_accum_counts.zero_()
                    op_revive_accum_counts.zero_()
                    revive_lookups_accum = 0
                    epoch_z_pool.clear()
                    epoch_op_pool.clear()
                elif not in_revive_phase and len(epoch_z_pool) > 0:
                    epoch_z_pool.clear()
                    epoch_op_pool.clear()

                last_ep_revived = ep_revived
                model.vq_layer.usage_count.zero_()
                model.op_vq.usage_count.zero_()

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

                    ponder_floor = float(CONFIG.get("ponder_min_penalty", 0.008))
                    if act_std > 1.15:
                        live_ponder_penalty = min(0.08, live_ponder_penalty * 1.02)
                    elif act_std < 0.25 and avg_act_steps <= 2.4:
                        live_ponder_penalty = max(ponder_floor, live_ponder_penalty * 0.97)
                    elif avg_act_steps > act_high:
                        live_ponder_penalty = min(0.15, live_ponder_penalty * 1.05)
                    elif avg_act_steps < act_low:
                        live_ponder_penalty = max(ponder_floor, live_ponder_penalty * 0.97)

                if phase_id == 3:
                    target_alpha = min(0.40, max(0.20, 0.40 - 0.25 * max(0.0, avg_vq - 0.25)))
                    model.sym_alpha.mul_(0.85).add_(0.15 * target_alpha)

                    u_score = min(1.0, true_ppx / max(1.0, ppx_low))
                    vq_score = min(1.0, max(0.0, (1.2 - avg_vq) / 0.5))
                    rev_score = max(0.25, 1.0 - last_ep_revived / max(1.0, 0.25 * CONFIG["n_symbols"]))
                    p3_ramp = min(1.0, (epoch - phase2_end + 1) / 15.0)
                    readiness = 0.85 * readiness + 0.15 * (u_score * vq_score * rev_score * p3_ramp)

            is_last_epoch = (epoch == CONFIG["epochs"] - 1)
            is_log_epoch = (epoch % CONFIG["log_every"] == 0) or is_last_epoch
            is_eval_epoch = (epoch % eval_every == 0) or is_log_epoch
            is_phase1_end = (epoch == phase1_end - 1)
            is_phase2_end = (epoch == phase2_end - 1)
            is_phase3_end = is_last_epoch
            printed_sample_this_epoch = False
            is_new_best = False
            is_ready = (
                (phase_id == 3)
                and (readiness >= 0.50 or epoch >= phase2_end + CONFIG["warmup_epochs"])
                and active_syms >= min_active_converge
            )
            n_slots_tot = max(1, int(total_slots_t.item()))
            acc_pct = 100.0 * stem_ok_t.item() / n_slots_tot
            m_acc_pct = 100.0 * morph_ok_t.item() / n_slots_tot
            e_acc_pct = 100.0 * edge_ok_t.item() / n_slots_tot
            sym_acc_pct = 100.0 * total_sym_correct_t.item() / max(1, total_pkts)
            rel_acc_pct = 100.0 * total_rel_acc_t.item() / max(1, rel_acc_steps)

            val_loss, val_bpc, val_acc, val_m_acc, val_e_acc = None, None, None, None, None
            monitor_metric = ema_bpc

            if is_ready and is_eval_epoch:
                t_va_0 = time.time()
                val_loss, val_bpc, val_acc, val_m_acc, val_e_acc = evaluate_val(
                    model, val_batches, macro_active=macro_active
                )
                va_time_since_log += time.time() - t_va_0
                monitor_metric = val_bpc if val_bpc is not None else ema_bpc
                if monitor_metric < best_metric - 1e-3:
                    best_metric = monitor_metric
                    best_epoch = epoch
                    history["best_epoch"] = best_epoch
                    t_io_0 = time.time()
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    dirty_best_disk = True
                    io_time_since_log += time.time() - t_io_0
                    is_new_best = True
                    improved_since_log = True
                    improved_since_sched = True

                if epoch >= lr_warmup_epochs:
                    sched_metric = best_metric if improved_since_sched else monitor_metric
                    plateau_scheduler.step(sched_metric)
                    improved_since_sched = False

            if dirty_best_disk and (is_log_epoch or is_last_epoch) and best_state is not None:
                t_io_0 = time.time()
                torch.save({
                    'model_state_dict': best_state,
                    'config': CONFIG,
                    'stem_vocab': tokenizer.stem_vocab,
                    'morph_vocab': tokenizer.morph_vocab,
                    'edge_vocab': tokenizer.edge_vocab,
                    'macro_trie': macro_trie,
                }, save_path)
                dirty_best_disk = False
                io_time_since_log += time.time() - t_io_0

            if is_log_epoch:
                if not is_ready:
                    t_va_0 = time.time()
                    val_loss, val_bpc, val_acc, val_m_acc, val_e_acc = evaluate_val(
                        model, val_batches, macro_active=macro_active
                    )
                    va_time_since_log += time.time() - t_va_0
                    monitor_metric = val_bpc if val_bpc is not None else ema_bpc

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
                w_sl_v = total_w_symlm_t.item() / stream_len
                w_vq_v = total_w_vq_t.item() / stream_len
                w_eth_v = total_w_eth_t.item() / stream_len
                w_cons_v = total_w_cons_t.item() / stream_len
                w_stk_v = total_w_stk_t.item() / stream_len
                w_pnd_v = total_w_pnd_t.item() / stream_len
                w_div_v = avg_div
                w_tot = max(1e-6, w_pred_v + w_sl_v + w_vq_v + w_eth_v + w_cons_v + w_stk_v + w_pnd_v + w_div_v)

                mod_avgs = {k_m: (mod_gnorm_sums[k_m].item() / max(1, mod_gnorm_count)) for k_m in mod_param_groups}
                for k_m in mod_param_groups:
                    mod_gnorm_sums[k_m].zero_()
                mod_gnorm_count = 0

                loop_pct = 100.0 * self_loops_t.item() / max(1, trans_toks)
                strong_edges = int((F.softmax(model.vq_layer.adjacency.float(), dim=-1) > 0.08).sum().item())
                avg_gnorm = sum(grad_norms) / max(1, len(grad_norms))
                max_gnorm = max(grad_norms) if grad_norms else 0.0
                clip_pct = 100.0 * clipped_chunks_since_log / max(1, total_chunks_since_log)
                clip_thresh_now = max(0.5, min(clip_max, 1.8 * ema_gnorm))

                act_dist = (act_step_hist.float() / max(1, total_pkts) * 100.0).tolist()
                act_dist_str = "/".join(f"{p:2.0f}" for p in act_dist)

                dt = max(1e-4, time.time() - t_start)
                unit_per_sec = units_since_log / dt
                t_start = time.time()
                units_since_log = 0
                clipped_chunks_since_log = 0
                total_chunks_since_log = 0
                lr = opt.param_groups[0]["lr"]

                if prev_logged_lr is not None and lr < prev_logged_lr * 0.75 and epoch >= lr_warmup_epochs:
                    history["lr_drop_epochs"].append(epoch)
                prev_logged_lr = lr

                history["epoch"].append(epoch)
                history["train_pred"].append(avg_pred)
                history["train_bpc"].append(train_bpc)
                history["val_loss"].append(val_loss if val_loss is not None else float("nan"))
                history["val_bpc"].append(val_bpc if val_bpc is not None else float("nan"))
                history["gen_gap"].append((val_bpc - train_bpc) if val_bpc is not None else float("nan"))
                history["train_acc"].append(acc_pct)
                history["val_acc"].append(val_acc if val_acc is not None else float("nan"))
                history["sym_lm_loss"].append(avg_sym_lm)
                history["sym_lm_acc"].append(sym_acc_pct)
                history["rel_acc"].append(rel_acc_pct)
                history["act_mean"].append(avg_act_steps)
                history["act_std"].append(act_std)
                history["act_dist"].append(act_dist)
                history["sym_ppx"].append(true_ppx)
                history["op_ppx"].append(op_ppx)
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
                history["gm_op"].append(mod_avgs["Op"])
                history["gm_adj"].append(mod_avgs["Adj"])
                history["gm_holo"].append(mod_avgs["Holo"])
                history["gm_dec"].append(mod_avgs["Dec"])

                if val_bpc is not None:
                    gen_gap = val_bpc - train_bpc
                    best_tag = "*BEST*" if is_new_best else (f"BestBPC:{best_metric:.3f}@{best_epoch}" if best_epoch >= 0 else "")
                    val_str = f" | ValBPC:{val_bpc:.3f} (S:{val_acc:4.1f}% M:{val_m_acc:4.1f}% E:{val_e_acc:4.1f}% Gap:{gen_gap:+.2f}) {best_tag}"
                else:
                    best_tag = "*BEST*" if is_new_best else (f"BestBPC:{best_metric:.3f}@{best_epoch}" if best_epoch >= 0 else "")
                    val_str = f" | {best_tag}" if best_tag else ""

                vram_str = f" VRAM:{torch.cuda.max_memory_allocated() / (1024 ** 3):.1f}G" if DEVICE == "cuda" else ""
                m_tag = "M" if macro_active else ""

                print(
                    f"Ep {epoch:04d} [P{phase_id}{m_tag}] | Loss:{avg_loss:.3f} BPC:{train_bpc:.3f} "
                    f"(SymLM:{avg_sym_lm:.3f}[{sym_acc_pct:4.1f}%] VQ:{avg_vq:.3f} Div:{avg_div:.3f} Eth:{avg_eth:.3f}) | "
                    f"Acc(S/M/E):{acc_pct:4.1f}/{m_acc_pct:4.1f}/{e_acc_pct:4.1f}%{val_str} | "
                    f"{unit_per_sec:,.0f} u/s (Tr:{tr_time_since_log:.1f}s Va:{va_time_since_log:.1f}s Io:{io_time_since_log:.1f}s){vram_str}"
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
                    f"VQ:{mod_avgs['VQ']:.3f} Op:{mod_avgs['Op']:.3f} Adj:{mod_avgs['Adj']:.3f} Holo:{mod_avgs['Holo']:.2f} Dec:{mod_avgs['Dec']:.2f}] | "
                    f"LossPct[Pr:{100*w_pred_v/w_tot:2.0f}% SL:{100*w_sl_v/w_tot:2.0f}% VQ:{100*w_vq_v/w_tot:2.0f}% Et:{100*w_eth_v/w_tot:2.0f}% "
                    f"Cn:{100*w_cons_v/w_tot:2.0f}% St:{100*w_stk_v/w_tot:2.0f}% Pn:{100*w_pnd_v/w_tot:2.0f}% Dv:{100*w_div_v/w_tot:2.0f}%]"
                )
                print(
                    f"        | Syms:{active_syms:3d}/{CONFIG['n_symbols']} SymPPX:{true_ppx:5.1f} VQCos:{avg_vq_cos:.2f} "
                    f"Ops:{active_ops:2d}/{CONFIG['n_ops']} OpPPX:{op_ppx:4.1f} RelAcc:{rel_acc_pct:4.1f}% "
                    f"Drift:{sym_drift:.3f} Top1:{top1_pct:4.1f}% Loop:{loop_pct:4.1f}% Rev:{revived_total:3d}/{op_revived_total:2d} "
                    f"Edges:{strong_edges:4d}(AdjPPX:{adj_ppx:5.1f}) | "
                    f"Tune[Ready:{readiness*100:3.0f}% DivW:{live_div_weight:.2f} Temp:{model.vq_layer.live_temp:.2f} Cmt:{live_commit_cost:.2f} "
                    f"Alpha:{model.sym_alpha.item():.2f} Bias:{live_bias_scale:.3f} GGate:{model.graph_gate.item():.2f} PndP:{live_ponder_penalty:.4f} "
                    f"Consc:{live_conscience:.2f} Clip:{clip_thresh_now:.2f}({clip_pct:2.0f}%) Pat:{no_improve_logs}/{CONFIG['early_stop_patience']}]"
                )
                revived_total = 0
                op_revived_total = 0
                skipped_spikes_since_log = 0
                tr_time_since_log = 0.0
                va_time_since_log = 0.0
                io_time_since_log = 0.0

            if is_new_best:
                if val_bpc is not None:
                    print(
                        f"   >>> *BEST* EPOCH {epoch:04d} | Val BPC: {best_metric:.4f} "
                        f"(ValLoss: {val_loss:.3f} StemAcc: {val_acc:4.1f}%) | Cached best weights"
                    )
                else:
                    print(
                        f"   >>> *BEST* EPOCH {epoch:04d} | Train BPC (EMA): {best_metric:.4f} "
                        f"(BPC: {train_bpc:.4f} StemAcc: {acc_pct:4.1f}%) | Cached best weights"
                    )
                print(f"   Sample (*BEST* Ep {epoch:04d}): {repr(generate_sample_preview(model, macro_active=macro_active))}")
                printed_sample_this_epoch = True

            if is_log_epoch and (epoch % CONFIG["sample_every"] == 0) and not printed_sample_this_epoch and not (is_phase1_end or is_phase2_end or is_phase3_end):
                print(f"   Sample: {repr(generate_sample_preview(model, macro_active=macro_active))}")
                printed_sample_this_epoch = True

            if is_phase1_end:
                print(f"   [End of Phase 1 Sample @ Ep {epoch:04d}]: {repr(generate_sample_preview(model, macro_active=False))}")
            elif is_phase2_end:
                print(f"   [End of Phase 2 Sample @ Ep {epoch:04d}]: {repr(generate_sample_preview(model, macro_active=False))}")
            elif is_phase3_end:
                print(f"   [End of Phase 3 Sample @ Ep {epoch:04d}]: {repr(generate_sample_preview(model, macro_active=macro_active))}")

            if is_ready:
                small_text_converged = (
                    not val_atomic_slices
                    and epoch >= phase2_end + 30
                    and train_bpc < 0.04
                    and acc_pct >= 98.5
                )
                if train_bpc < 0.02 or small_text_converged:
                    print("\n--- CONVERGENCE REACHED ---")
                    if best_state is None or is_new_best:
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        if not val_atomic_slices:
                            best_metric = train_bpc
                            best_epoch = epoch
                            history["best_epoch"] = best_epoch
                    else:
                        model.load_state_dict(best_state)
                    save_checkpoint(model, save_path, macro_trie=macro_trie)
                    print(f"   >>> *BEST* EPOCH {best_epoch:04d} | Metric BPC: {best_metric:.4f} (StemAcc: {acc_pct:.1f}%) | Saved checkpoint -> {save_path}")
                    if not is_phase3_end:
                        print(f"   [End of Phase 3 Sample @ Ep {epoch:04d}]: {repr(generate_sample_preview(model, macro_active=macro_active))}")
                    break

            if is_log_epoch and progress >= 0.4 and lr <= min_lrs[0] * 2.1 and no_improve_logs >= CONFIG["early_stop_patience"]:
                print(f"\n--- EARLY STOPPING (LR converged & plateaued for {no_improve_logs} checks) ---")
                if not is_phase3_end:
                    print(f"   [End of Phase 3 Sample @ Ep {epoch:04d}]: {repr(generate_sample_preview(model, macro_active=macro_active))}")
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    if best_state is not None:
        model.load_state_dict(best_state)
        metric_name = "Val BPC" if val_atomic_slices else "Train BPC (EMA)"
        print(f"Restored best healthy checkpoint ({metric_name}: {best_metric:.4f} at Ep {best_epoch}).")

    return model, history, macro_trie, macro_active, val_batches


# ==========================================
# 8. Visualization Suite (0..6 Diagnostics)
# ==========================================
def plot_training_dynamics(history):
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
            ax.axvline(p2_end, color="teal", linestyle="--", linewidth=1.1, alpha=0.7, label="P3 Macro+Rel" if show_legend_labels else None)
        if best_ep >= 0:
            ax.axvline(best_ep, color="gold", linestyle="-.", linewidth=1.4, alpha=0.9, label=f"Best Ep {best_ep}" if show_legend_labels else None)
        for idx_d, d_ep in enumerate(lr_drops):
            ax.axvline(d_ep, color="navy", linestyle=":", linewidth=1.0, alpha=0.6, label="LR Drop" if (show_legend_labels and idx_d == 0) else None)

    # Panel (0, 0): Bits-Per-Character (BPC) & Pure SymLM
    ax1 = axes[0, 0]
    tr_bpc = np.array(history["train_bpc"], dtype=np.float64)
    va_bpc = np.array(history["val_bpc"], dtype=np.float64)
    ax1.plot(epochs, tr_bpc, color="royalblue", linewidth=2.0, label="Train BPC")
    if np.isfinite(va_bpc).any():
        ax1.plot(epochs, va_bpc, color="crimson", linewidth=2.0, label="Val BPC")
        valid_m = np.isfinite(va_bpc)
        ep_arr = np.array(epochs)
        ax1.fill_between(ep_arr[valid_m], tr_bpc[valid_m], va_bpc[valid_m], color="crimson", alpha=0.10, label="BPC Gap")
    ax1_r = ax1.twinx()
    ax1_r.plot(epochs, history["sym_lm_loss"], color="darkorange", linewidth=1.3, linestyle="--", alpha=0.85, label="Pure SymLM CE")
    ax1_r.set_ylabel("Pure SymLM CE (nats)", color="darkorange")
    _add_phase_markers(ax1, show_legend_labels=True)
    ax1.set_title("Bits-Per-Character (BPC) Trajectory & Symbolic LM")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Bits Per Character (BPC)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax1_r.legend(loc="upper left", fontsize=8)

    # Panel (0, 1): ACT Thinking-Step Evolution
    ax2 = axes[0, 1]
    act_mat = np.array(history["act_dist"], dtype=np.float64)
    step_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
    for d_i in range(act_mat.shape[1]):
        ax2.plot(epochs, act_mat[:, d_i], color=step_colors[d_i % len(step_colors)], linewidth=1.8, label=f"{d_i+1}-Step %")
    ax2_r = ax2.twinx()
    ax2_r.plot(epochs, history["act_mean"], color="black", linestyle="--", linewidth=1.5, label="Mean Steps")
    ax2_r.set_ylabel("Mean ACT Steps", color="black")
    ax2_r.set_ylim(0.5, CONFIG["max_recursion_depth"] + 0.3)
    _add_phase_markers(ax2)
    ax2.set_title("ACT Depth Distribution Over Training")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Token Share (%)")
    ax2.set_ylim(-2, 102)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", fontsize=8)
    ax2_r.legend(loc="upper right", fontsize=8)

    # Panel (1, 0): Codebook, Operator PPX & Relational Transition Accuracy
    ax3 = axes[1, 0]
    ax3.plot(epochs, history["sym_ppx"], color="darkorchid", linewidth=2.0, label="SymPPX")
    ax3.plot(epochs, history["op_ppx"], color="magenta", linewidth=1.5, linestyle="--", label="OpPPX")
    ax3.axhspan(CONFIG["target_ppx_low"], CONFIG["target_ppx_high"], color="green", alpha=0.10, label="Target SymPPX Band")
    ax3.plot(epochs, np.minimum(history["adj_ppx"], float(CONFIG["n_symbols"])), color="darkorange", linewidth=1.6, linestyle="-.", label="AdjPPX (capped)")
    ax3_r = ax3.twinx()
    ax3_r.plot(epochs, history["rel_acc"], color="teal", linewidth=1.5, label="RelTrans Acc%")
    ax3_r.plot(epochs, history["top1_pct"], color="crimson", linewidth=1.3, linestyle=":", label="Top1 %")
    ax3_r.set_ylabel("RelAcc% / Top1%")
    _add_phase_markers(ax3)
    ax3.set_title("Codebook, Operator PPX & Relational Transition Accuracy")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Perplexity")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper left", fontsize=8)
    ax3_r.legend(loc="upper right", fontsize=8)

    # Panel (1, 1): Memory Routing & Stack Depth
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

    # Panel (2, 0): Per-Module Gradient Norms (Log Scale)
    ax5 = axes[2, 0]
    mod_specs = [
        ("gm_cell", "Cell", "royalblue"),
        ("gm_ffn", "FFN", "darkorange"),
        ("gm_dec", "Dec/Emb", "crimson"),
        ("gm_holo", "Holo", "teal"),
        ("gm_att", "Att", "purple"),
        ("gm_vq", "VQ", "seagreen"),
        ("gm_op", "RelOp", "magenta"),
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

    # Panel (2, 1): Stem Accuracy, Pure Symbolic Accuracy & LR Schedule
    ax6 = axes[2, 1]
    ax6.plot(epochs, history["train_acc"], color="royalblue", linewidth=2.0, label="Train Stem Acc %")
    ax6.plot(epochs, history["sym_lm_acc"], color="darkorange", linewidth=1.5, linestyle="--", label="Pure SymLM Acc %")
    va_acc = np.array(history["val_acc"], dtype=np.float64)
    if np.isfinite(va_acc).any():
        ax6.plot(epochs, va_acc, color="crimson", linewidth=2.0, label="Val Stem Acc %")
    ax6_r = ax6.twinx()
    ax6_r.semilogy(epochs, np.maximum(np.array(history["lr"], dtype=np.float64), 1e-7), color="black", linestyle=":", linewidth=1.5, label="LR (log)")
    ax6_r.set_ylabel("Learning Rate (log)")
    _add_phase_markers(ax6)
    ax6.set_title("Stem Accuracy, Pure Symbolic Accuracy & LR Schedule")
    ax6.set_xlabel("Epoch")
    ax6.set_ylabel("Accuracy (%)")
    ax6.grid(True, alpha=0.25)
    ax6.legend(loc="upper left", fontsize=8)
    ax6_r.legend(loc="lower right", fontsize=8)

    fig.suptitle("SACRSN v60 — Dual-Clock Polar + Macro-Packet Training Dashboard", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("0_training_dynamics.png", dpi=150)
    print("Saved 0_training_dynamics.png")
    plt.close()


def build_symbol_token_table(model, atomic_dict, max_steps=None):
    n_avail = len(atomic_dict["stems"]) - 1
    if max_steps is None:
        max_steps = min(n_avail, max(CONFIG["diag_max_tokens"], 20000))
    else:
        max_steps = min(n_avail, max_steps)

    model.eval()
    symbol_to_tok = defaultdict(list)
    sym_stem_counts = defaultdict(Counter)
    sym_op_counts = defaultdict(Counter)
    hidden, prev_sym = None, None

    cached_cb = model.vq_layer.normalized_codebook()
    cached_padj = F.softmax(model.vq_layer.adjacency.float(), dim=-1)

    with torch.no_grad(), torch.amp.autocast(AMP_DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
        for i in range(max_steps):
            s_id = atomic_dict["stems"][i].item()
            m_id = atomic_dict["morphs"][i].item()
            e_id = atomic_dict["edges"][i].item()
            x_s = torch.tensor([[s_id]], device=DEVICE)
            x_m = torch.tensor([[m_id]], device=DEVICE)
            x_l = torch.tensor([1], device=DEVICE)
            curr_edge = torch.tensor([e_id], device=DEVICE, dtype=torch.long)

            out = model(
                x_s, x_m, x_l, prev_edge=curr_edge, hidden=hidden, prev_sym=prev_sym,
                return_stats=True, cached_P_adj=cached_padj, cached_cb_norm=cached_cb
            )
            hidden = out[1]
            prev_sym = out[2]
            curr_op = out[17]["curr_op"].item()

            stem_str = tokenizer.id_to_stem.get(s_id, "<UNK>")
            sym_id = prev_sym.item()
            symbol_to_tok[sym_id].append(stem_str)
            sym_stem_counts[sym_id][s_id] += 1
            sym_op_counts[sym_id][curr_op] += 1

    return symbol_to_tok, sym_stem_counts, sym_op_counts


def visualize_all(model, macro_active=False):
    print("\n--- Generating Diagnostics & Images ---")
    model.eval()

    diag_steps = min(len(data_atomic["stems"]) - 1, CONFIG["diag_max_tokens"])
    symbol_to_tok, sym_stem_counts, sym_op_counts = build_symbol_token_table(model, data_atomic, diag_steps)

    all_counts = np.zeros(CONFIG["n_symbols"], dtype=np.int64)
    purity_map = {}
    total_hits = 0
    purity_hits = 0
    for s_idx in range(CONFIG["n_symbols"]):
        cntr = sym_stem_counts.get(s_idx)
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
    top10_k = max(1, CONFIG["n_symbols"] // 10)
    top10_share = 100.0 * float(sorted_counts[:top10_k].sum()) / max(1, total_hits)
    med_hits = float(np.median(nonzero_counts)) if len(nonzero_counts) > 0 else 0.0
    avg_purity = 100.0 * purity_hits / max(1, total_hits)
    print(
        f"Codebook Health : Active={len(nonzero_counts)}/{CONFIG['n_symbols']} "
        f"| Top10%Share={top10_share:.1f}% | MedianHits={med_hits:.0f} | AvgStemPurity={avg_purity:.1f}%"
    )

    top_syms = sorted(sym_stem_counts.items(), key=lambda kv: sum(kv[1].values()), reverse=True)[:5]
    for s_id, cntr in top_syms:
        s_tot = sum(cntr.values())
        top3_str = ", ".join(
            f"'{tokenizer.id_to_stem.get(sid, '<UNK>')}'({cnt})"
            for sid, cnt in cntr.most_common(3)
        )
        top_op_s = sym_op_counts[s_id].most_common(1)[0][0] if sym_op_counts[s_id] else -1
        p_s = purity_map[s_id]
        print(f"   Hub S_{s_id:<4} (hits={s_tot:<4}, purity={p_s:4.1f}%, dom_op=R_{top_op_s}): [{top3_str}]")

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
        fig.colorbar(sm, ax=ax, fraction=0.032, pad=0.02, label="Concept Stem Purity (% Top-Stem Share)")

    ax.set_title(f"1_semantic_topology (Concept Stems Active: {len(symbol_to_tok)}/{CONFIG['n_symbols']}, Shown: {len(G.nodes())})")
    plt.tight_layout()
    plt.savefig("1_semantic_topology.png", dpi=150)
    print("Saved 1_semantic_topology.png")
    plt.close()

    # Autoregressive Inference Scan (supports Macro-Packets)
    hidden, prev_sym = None, None
    s0 = data_atomic["stems"][0].item()
    m0 = data_atomic["morphs"][0].item()
    e0 = data_atomic["edges"][0].item()
    curr_s = torch.tensor([[s0]], device=DEVICE)
    curr_m = torch.tensor([[m0]], device=DEVICE)
    curr_l = torch.tensor([1], device=DEVICE)
    curr_edge = torch.tensor([e0], device=DEVICE, dtype=torch.long)
    curr_pkt_tuples = [(s0, m0, e0)]

    stack_history, act_history = [], []
    push_history, pop_history, noop_history = [], [], []
    pred_ent_history = []
    latent_traj = []
    scan_tok_labels = []
    gen_pkts = [curr_pkt_tuples]
    recent_stems = [s0]
    scan_len = max(45, min(90, int(round(140 / max(1.0, CONFIG["comp_ratio"])))))

    cached_cb = model.vq_layer.normalized_codebook()
    cached_padj = F.softmax(model.vq_layer.adjacency.float(), dim=-1)

    print("Running Inference Scan...")
    with torch.no_grad(), torch.amp.autocast(AMP_DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
        for _ in range(scan_len):
            if len(curr_pkt_tuples) == 1:
                lbl = tokenizer.format_unit(*curr_pkt_tuples[0])
            else:
                lbl = "[" + "·".join(tokenizer.decode_word(u[0], u[1]) for u in curr_pkt_tuples) + "]"
            scan_tok_labels.append(lbl)

            (
                stem_logits,
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
                extra,
            ) = model(
                curr_s, curr_m, curr_l, prev_edge=curr_edge, hidden=hidden, prev_sym=prev_sym,
                decode_slots=1, return_stats=True, cached_P_adj=cached_padj, cached_cb_norm=cached_cb
            )

            stack_history.append(avg_stack.item())
            act_history.append(steps_vec.item())
            ctrl_list = avg_stk_ctrl.detach().cpu().tolist()
            push_history.append(ctrl_list[0])
            pop_history.append(ctrl_list[1])
            noop_history.append(ctrl_list[2])

            probs = F.softmax(stem_logits[:, 0, :].squeeze(0).float(), dim=-1)
            pred_ent = -(probs * torch.log(probs + 1e-8)).sum().item()
            pred_ent_history.append(pred_ent)

            zr = hidden[0].detach().float().cpu().squeeze(0)
            zi = hidden[1].detach().float().cpu().squeeze(0)
            latent_traj.append(torch.cat([zr, zi], dim=-1))

            z_wr, z_wi = extra["z_w_r"], extra["z_w_i"]
            if macro_active:
                len_probs = F.softmax(extra["len_logits"].squeeze(0).float() / 0.85, dim=-1)
                next_L = int(torch.multinomial(len_probs, 1).item()) + 1
            else:
                next_L = 1

            next_tuples, slot_s_list, slot_m_list = [], [], []
            prev_flat = None
            for j in range(next_L):
                feat_j = model.decode_slot_features(z_wr, z_wi, j, prev_slot_flat=prev_flat)
                sj = sample_next_token(model.decoder(feat_j), recent_stems, temperature=0.80, top_k=40, top_p=0.92, rep_penalty=1.10)
                mj = sample_next_token(model.morph_decoder(feat_j), None, temperature=0.40, top_k=8, top_p=0.85, rep_penalty=1.0)
                ej = sample_edge_token(model.edge_decoder(feat_j), sj, temperature=0.35, top_k=6, top_p=0.85)
                next_tuples.append((sj, mj, ej))
                slot_s_list.append(sj)
                slot_m_list.append(mj)
                recent_stems.append(sj)
                if j + 1 < next_L:
                    pw_r, pw_i = model.embed_polar_words(torch.tensor([sj], device=DEVICE), torch.tensor([mj], device=DEVICE))
                    prev_flat = torch.cat([pw_r, pw_i], dim=-1)

            gen_pkts.append(next_tuples)
            curr_pkt_tuples = next_tuples
            curr_s = torch.tensor([slot_s_list], device=DEVICE, dtype=torch.long)
            curr_m = torch.tensor([slot_m_list], device=DEVICE, dtype=torch.long)
            curr_l = torch.tensor([next_L], device=DEVICE, dtype=torch.long)
            curr_edge = torch.tensor([next_tuples[-1][2]], device=DEVICE, dtype=torch.long)

    act_counts = Counter(int(round(a)) for a in act_history)
    act_dist_str = ", ".join(
        f"{k} steps: {100.0 * v / len(act_history):.0f}%" for k, v in sorted(act_counts.items())
    )
    pkt_lens_gen = [len(p) for p in gen_pkts]
    print(f"ACT Step Breakdown : {act_dist_str}")
    print(f"Stack Depth Range  : [{min(stack_history):.2f}, {max(stack_history):.2f}] | Avg Gen PktLen={np.mean(pkt_lens_gen):.2f}")
    print(f"Generated          : {tokenizer.decode_packets(gen_pkts)}\n")

    x_idx = np.arange(len(stack_history))

    # 2_stack_mri.png
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 6.8), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.0]})
    ax_top.plot(x_idx, stack_history, color='purple', linewidth=2.0, label='Stack Depth')
    ax_top.fill_between(x_idx, stack_history, color='purple', alpha=0.12)
    ax_top.set_title("2_stack_mri (Memory Depth & Per-Packet Push/Pop/NoOp Control)")
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

    # 3_act_profile.png
    fig, ax_act = plt.subplots(figsize=(16, 4.8))
    ax_act.bar(x_idx, act_history, color='orange', alpha=0.85, label='ACT Steps')
    ax_act.set_ylim(0, CONFIG["max_recursion_depth"] + 0.5)
    ax_act.set_ylabel("Thinking Steps")
    ax_act.grid(True, axis='y', alpha=0.25)

    ax_ent = ax_act.twinx()
    ax_ent.plot(x_idx, pred_ent_history, color='navy', linewidth=1.6, marker='o', markersize=3, alpha=0.85, label='Stem Entropy (nats)')
    ax_ent.set_ylabel("Stem Predictive Entropy (nats)", color='navy')

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

    # 4_phase_plot.png
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
        if any(ch in lbl for ch in (":", ".", "!", "?", "\\n")) or idx_t % 8 == 0:
            ax_pca.annotate(lbl, (pc[idx_t, 0], pc[idx_t, 1]), fontsize=7, alpha=0.8, xytext=(3, 3), textcoords="offset points")
    fig.colorbar(sc, ax=ax_pca, label="Packet Step")
    ax_pca.set_title(f"4_phase_plot (2D PCA of {Z_traj.size(1)}-D Complex Latent)")
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    ax_pca.grid(True, alpha=0.25)
    ax_pca.legend(loc="upper right", fontsize=8)

    ax_nv.plot(np.arange(T_pts), norms, color='teal', linewidth=1.8, label='State Norm ||z_t||')
    if T_pts > 1:
        ax_nv.plot(np.arange(1, T_pts), velocities, color='darkorange', linewidth=1.5, alpha=0.9, label='Step Velocity ||z_t - z_{t-1}||')
    ax_nv.set_title("Latent State Norm & Step-to-Step Velocity")
    ax_nv.set_xlabel("Packet Step")
    ax_nv.set_ylabel("L2 Magnitude")
    ax_nv.grid(True, alpha=0.25)
    ax_nv.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig("4_phase_plot.png", dpi=150)
    print("Saved 4_phase_plot.png")
    plt.close()

    return sym_stem_counts, sym_op_counts, sorted_counts


# ==========================================
# 9. Logic Rule Extraction (Typed Relational + Dual-Clock Edge Triples)
# ==========================================
def extract_logic_rules(model, atomic_dict, sym_stem_counts=None):
    print("\n--- Extracting Explicit Logic Rules (Relational Operator + Dual-Clock Edge Triples) ---")
    model.eval()
    rule_book_1 = defaultdict(list)
    rule_book_op = defaultdict(Counter)
    rule_book_edge = defaultdict(Counter)
    rule_book_2 = defaultdict(list)

    hidden = None
    prev_sym = None
    prev_prev_sym = None
    diag_steps = min(len(atomic_dict["stems"]) - 1, CONFIG["diag_max_tokens"])
    min_rule_1 = max(2, diag_steps // 1000)
    min_rule_2 = max(2, diag_steps // 1500)

    local_counts = defaultdict(Counter) if sym_stem_counts is None else sym_stem_counts
    cached_cb = model.vq_layer.normalized_codebook()
    cached_padj = F.softmax(model.vq_layer.adjacency.float(), dim=-1)

    with torch.no_grad(), torch.amp.autocast(AMP_DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
        for i in range(diag_steps):
            s_id = atomic_dict["stems"][i].item()
            m_id = atomic_dict["morphs"][i].item()
            e_id = atomic_dict["edges"][i].item()
            x_s = torch.tensor([[s_id]], device=DEVICE)
            x_m = torch.tensor([[m_id]], device=DEVICE)
            x_l = torch.tensor([1], device=DEVICE)
            ce = torch.tensor([e_id], device=DEVICE, dtype=torch.long)

            out = model(
                x_s, x_m, x_l, prev_edge=ce, hidden=hidden, prev_sym=prev_sym,
                return_stats=True, cached_P_adj=cached_padj, cached_cb_norm=cached_cb
            )
            hidden = out[1]
            sym_idx = out[2]
            steps_v = out[11].item()
            curr_op = out[17]["curr_op"].item()

            if sym_stem_counts is None:
                local_counts[sym_idx.item()][s_id] += 1
            curr = sym_idx.item()
            if prev_sym is not None:
                src = prev_sym.item()
                rule_book_1[(src, curr)].append(steps_v)
                rule_book_op[(src, curr)][curr_op] += 1
                rule_book_edge[(src, curr)][e_id] += 1
                if prev_prev_sym is not None:
                    rule_book_2[(prev_prev_sym, src, curr)].append(steps_v)
                prev_prev_sym = src
            prev_sym = sym_idx

    def sym_tag(s):
        c = local_counts.get(s)
        if not c:
            return f"S_{s:<4}"
        top_stem = tokenizer.id_to_stem.get(c.most_common(1)[0][0], "<UNK>")
        if len(top_stem) > 8:
            top_stem = top_stem[:7] + "…"
        return f"S_{s:<4}({top_stem})"

    print(f"\n[1-Hop Relational Rules] {'FROM':<16} | {'OP | EDGE':<14} | {'TO':<16} | {'COUNT':<6} | {'AVG STEPS'}")
    print("-" * 80)
    sorted_rules_1 = sorted(rule_book_1.items(), key=lambda x: len(x[1]), reverse=True)
    printed = 0
    for (src, dst), steps_list in sorted_rules_1:
        if len(steps_list) >= min_rule_1:
            dom_op, op_cnt = rule_book_op[(src, dst)].most_common(1)[0]
            dom_edge, _ = rule_book_edge[(src, dst)].most_common(1)[0]
            edge_lbl = tokenizer.format_token(tokenizer.id_to_edge.get(dom_edge, ""))
            if len(edge_lbl) > 5:
                edge_lbl = edge_lbl[:4] + "…"
            op_pct = 100.0 * op_cnt / max(1, len(steps_list))
            print(
                f"{sym_tag(src):<16} --[R_{dom_op:<2}|{edge_lbl:<5}]--> {sym_tag(dst):<16} | "
                f"{len(steps_list):<6} | {sum(steps_list)/len(steps_list):.2f} (op {op_pct:3.0f}%)"
            )
            printed += 1
            if printed >= 25:
                break

    print(f"\n[2-Hop Concept Chains] {'S_A -> S_B -> S_C':<52} | {'COUNT':<6} | {'AVG STEPS'}")
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
def dream_mode(model, sym_stem_counts=None, sym_op_counts=None):
    print("\n--- Dream Mode (Relational Operator Conceptual Walk) ---")
    model.eval()
    if sym_stem_counts is None or sym_op_counts is None:
        _, sym_stem_counts, sym_op_counts = build_symbol_token_table(model, data_atomic)

    adj_logits = model.vq_layer.adjacency.detach().float()
    cached_cb = model.vq_layer.normalized_codebook()
    cached_op_cb = model.op_vq.normalized_codebook()
    cached_padj = F.softmax(adj_logits, dim=-1)

    s0 = data_atomic["stems"][0].item()
    m0 = data_atomic["morphs"][0].item()
    e0 = data_atomic["edges"][0].item()

    x_s = torch.tensor([[s0]], device=DEVICE)
    x_m = torch.tensor([[m0]], device=DEVICE)
    x_l = torch.tensor([1], device=DEVICE)
    pe = torch.tensor([e0], device=DEVICE, dtype=torch.long)

    with torch.no_grad(), torch.amp.autocast(AMP_DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
        out = model(
            x_s, x_m, x_l, prev_edge=pe, return_stats=True,
            cached_P_adj=cached_padj, cached_cb_norm=cached_cb
        )
        stem_logits = out[0][:, 0, :]
        hidden = out[1]
        prev_sym = out[2]
        extra = out[17]
        morph_logits = extra["morph_logits"][:, 0, :]
        edge_logits = extra["edge_logits"][:, 0, :]
        sym_logits = extra["sym_logits"]
        curr_op = extra["curr_op"].item()

    curr_sym = prev_sym.item()
    out_stems, out_morphs, out_edges = [s0], [m0], [e0]

    target_branch_ppx = min(4.0, max(2.0, 0.04 * CONFIG["n_symbols"]))
    dream_temps, dream_ppxs = [], []
    emitted_from_table = 0

    for _ in range(50):
        op_cntr = sym_op_counts.get(curr_sym)
        if op_cntr:
            op_ids = list(op_cntr.keys())
            op_w = np.array([op_cntr[oid] for oid in op_ids], dtype=np.float64)
            op_w = op_w / op_w.sum()
            chosen_op = int(np.random.choice(op_ids, p=op_w))
        else:
            chosen_op = curr_op

        with torch.no_grad():
            omega = cached_op_cb[chosen_op].unsqueeze(0)
            curr_sym_t = torch.tensor([curr_sym], device=DEVICE, dtype=torch.long)
            _, rel_logits = model.op_vq.bind_transition(curr_sym_t, omega, cached_cb)
            row_logits = adj_logits[curr_sym] + rel_logits.squeeze(0)

        lo, hi = 0.15, 2.50
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            p_mid = F.softmax(row_logits / mid, dim=-1)
            ent_mid = -(p_mid * torch.log(p_mid + 1e-10)).sum()
            if torch.exp(ent_mid).item() < target_branch_ppx:
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

        lm_probs = F.softmax(stem_logits.squeeze(0).float(), dim=-1).cpu().numpy()
        sym_lm_probs = F.softmax(sym_logits.squeeze(0).float(), dim=-1).cpu().numpy()
        lm_probs[0] = 0.0
        sym_lm_probs[0] = 0.0

        stem_counter = sym_stem_counts.get(next_sym)
        if stem_counter:
            cand_ids = [sid for sid in stem_counter.keys() if sid > 0]
            if cand_ids:
                cand_counts = np.array([stem_counter[sid] for sid in cand_ids], dtype=np.float64)
                cand_emit = cand_counts / cand_counts.sum()
                cand_lm = np.maximum(lm_probs[cand_ids], 1e-9) ** 0.30
                cand_sym = np.maximum(sym_lm_probs[cand_ids], 1e-9) ** 0.25
                combined = cand_emit * cand_lm * cand_sym
                combined = combined / combined.sum()
                s_idx = int(np.random.choice(cand_ids, p=combined))
                emitted_from_table += 1
            else:
                s_idx = int(np.argmax(0.5 * lm_probs + 0.5 * sym_lm_probs))
        else:
            s_idx = int(np.argmax(0.5 * lm_probs + 0.5 * sym_lm_probs))

        m_idx = sample_next_token(morph_logits, None, temperature=0.40, top_k=8, top_p=0.85, rep_penalty=1.0)
        e_idx = sample_edge_token(edge_logits, s_idx, temperature=0.35, top_k=6, top_p=0.85)

        out_stems.append(s_idx)
        out_morphs.append(m_idx)
        out_edges.append(e_idx)
        curr_sym = next_sym

        x_next_s = torch.tensor([[s_idx]], device=DEVICE)
        x_next_m = torch.tensor([[m_idx]], device=DEVICE)
        pe_next = torch.tensor([e_idx], device=DEVICE, dtype=torch.long)

        with torch.no_grad(), torch.amp.autocast(AMP_DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
            out = model(
                x_next_s, x_next_m, x_l, prev_edge=pe_next, hidden=hidden,
                prev_sym=torch.tensor([curr_sym], device=DEVICE, dtype=torch.long),
                return_stats=True, cached_P_adj=cached_padj, cached_cb_norm=cached_cb
            )
            stem_logits = out[0][:, 0, :]
            hidden = out[1]
            extra = out[17]
            morph_logits = extra["morph_logits"][:, 0, :]
            edge_logits = extra["edge_logits"][:, 0, :]
            sym_logits = extra["sym_logits"]
            curr_op = extra["curr_op"].item()

    print(
        f"Dream Stats  : AvgTemp={sum(dream_temps)/len(dream_temps):.2f} | "
        f"AvgBranchPPX={sum(dream_ppxs)/len(dream_ppxs):.2f} (Target={target_branch_ppx:.2f}) | "
        f"GroundedSyms={emitted_from_table}/50"
    )
    print(f"Dream Output : {tokenizer.decode_atomic(out_stems, out_morphs, out_edges)}\n")


def anomaly_detector(model):
    print("\n--- Anomaly Detection Test (With Warmup Context) ---")
    model.eval()

    if val_text_str is not None and len(val_text_str) > 200:
        mid = len(val_text_str) // 4
        snippet = val_text_str[mid : mid + 500]
        words = snippet.split()
        if len(words) > 2:
            words = words[1:]
    else:
        words = TEXT_DATA[:500].split()

    n_w = len(words)
    n_warm = min(4, max(0, n_w - 4)) if n_w >= 8 else min(2, max(0, n_w - 2))
    warmup_words = words[:n_warm]
    prefix_words = words[n_warm : min(n_w, n_warm + 7)]

    warmup_str = " ".join(warmup_words)
    eval_prefix_str = " ".join(prefix_words)
    full_prefix_str = " ".join(warmup_words + prefix_words)

    anomaly_word = "banana" if "banana" not in TEXT_DATA.lower() else "xyzzy"
    corrupt_text = f"{full_prefix_str} {anomaly_word}" if full_prefix_str else anomaly_word
    print(f"Warmup: '{warmup_str}' | Test: '{eval_prefix_str} {anomaly_word}'")

    atom_corr = tokenizer.encode_atomic(corrupt_text)
    n_u = len(atom_corr["stems"])
    if n_u < 3:
        print("Input too short for anomaly scan.")
        return

    w_len = min(n_u - 2, len(tokenizer.encode_atomic(warmup_str)["stems"])) if warmup_str else 0
    eval_start = max(1, w_len)
    p_len = min(n_u - 1, max(eval_start + 1, len(tokenizer.encode_atomic(full_prefix_str)["stems"])))

    hidden, prev_sym = None, None
    prev_s_logits, prev_m_logits, prev_e_logits = None, None, None

    lm_surprisals = []
    eth_violations = []
    act_trace = []

    cached_cb = model.vq_layer.normalized_codebook()
    cached_padj = F.softmax(model.vq_layer.adjacency.float(), dim=-1)

    with torch.no_grad(), torch.amp.autocast(AMP_DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
        for i in range(n_u):
            s_i = atom_corr["stems"][i].view(1, 1).to(DEVICE)
            m_i = atom_corr["morphs"][i].view(1, 1).to(DEVICE)
            e_i = atom_corr["edges"][i].view(1).to(DEVICE)
            l_i = torch.ones(1, dtype=torch.long, device=DEVICE)

            if i > 0 and prev_s_logits is not None:
                lm_nll = (
                    F.cross_entropy(prev_s_logits.float(), s_i.view(1)).item()
                    + F.cross_entropy(prev_m_logits.float(), m_i.view(1)).item()
                    + F.cross_entropy(prev_e_logits.float(), e_i.view(1)).item()
                )
            else:
                lm_nll = None

            s_log, m_log, e_log, _, hidden, prev_sym, act_steps, _, eth_loss, _ = model(
                s_i, m_i, l_i, prev_edge=e_i, hidden=hidden, prev_sym=prev_sym,
                cached_P_adj=cached_padj, cached_cb_norm=cached_cb
            )
            prev_s_logits = s_log[:, 0, :]
            prev_m_logits = m_log[:, 0, :]
            prev_e_logits = e_log[:, 0, :]

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
            f"RelSymEth: {eth_pre:.2f}->{eth_tail_max:.2f} ({eth_tail_max/max(1e-6,eth_pre):.2f}x) | "
            f"ACT: {act_pre:.2f}->{act_tail:.2f}]"
        )

    x_labels = [
        tokenizer.format_unit(atom_corr["stems"][k].item(), atom_corr["morphs"][k].item(), atom_corr["edges"][k].item())
        for k in range(eval_start, n_u)
    ]
    x_idx = np.arange(len(combined_scores))

    fig, (ax_v, ax_a) = plt.subplots(2, 1, figsize=(13, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    ax_v.plot(x_idx, combined_scores, color='skyblue', linestyle='-', linewidth=2, label=f'Combined Violation (LM + {eth_weight_vis:.1f}*RelEth)')
    ax_v.plot(x_idx, lm_surprisals, color='orange', linestyle='--', linewidth=1.3, alpha=0.85, label='LM Joint Surprisal (NLL)')
    ax_v.plot(x_idx, scaled_eth, color='purple', linestyle='-.', linewidth=1.3, alpha=0.85, label=f'Relational Topology ({eth_weight_vis:.1f}*RelEth)')
    ax_v.scatter(x_idx, combined_scores, color='crimson', s=50, zorder=5, label='Anomaly Score')

    ax_a.bar(x_idx, act_trace, color='orange', alpha=0.80, label='ACT Thinking Steps')
    ax_a.set_ylim(0, CONFIG["max_recursion_depth"] + 0.5)
    ax_a.set_ylabel("ACT Steps")
    ax_a.grid(True, alpha=0.25)

    if len(combined_scores) >= 2:
        ax_v.axvline(split_idx - 0.5, color='gray', linestyle=':', linewidth=1.5, label='Anomaly Boundary')
        ax_a.axvline(split_idx - 0.5, color='gray', linestyle=':', linewidth=1.5)

    ax_a.set_xticks(x_idx)
    ax_a.set_xticklabels(x_labels, rotation=35, ha='right', fontsize=8)
    ax_v.set_title("Topological, Relational & ACT Violation Score (Dual-Clock Polar Anomaly Detection)")
    ax_v.set_ylabel("Violation Magnitude")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend(loc="upper left", fontsize=8, ncol=2)
    ax_a.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig("5_anomaly_detection.png", dpi=150)
    print("Saved 5_anomaly_detection.png")
    plt.close()


def context_difficulty_probe(model):
    """Checks whether ACT allocates more recursion steps on high-difficulty predictions across clean slices."""
    targets = val_atomic_slices if val_atomic_slices else [data_atomic]
    total_avail = sum(max(0, len(t["stems"]) - 1) for t in targets)
    if total_avail < 32:
        return
    print("\n--- Context-Difficulty ACT Probe ---")
    model.eval()
    budget = min(total_avail, 1024)

    cached_cb = model.vq_layer.normalized_codebook()
    cached_padj = F.softmax(model.vq_layer.adjacency.float(), dim=-1)

    next_nlls, pred_ents, in_nlls, steps_aligned, steps_in = [], [], [], [], []
    scanned = 0
    with torch.no_grad(), torch.amp.autocast(AMP_DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
        for sl in targets:
            if scanned >= budget:
                break
            hidden, prev_sym, prev_s_logits = None, None, None
            sl_n = min(len(sl["stems"]) - 1, budget - scanned)
            for i in range(sl_n):
                s_i = sl["stems"][i].view(1, 1).to(DEVICE)
                m_i = sl["morphs"][i].view(1, 1).to(DEVICE)
                e_i = sl["edges"][i].view(1).to(DEVICE)
                l_i = torch.ones(1, dtype=torch.long, device=DEVICE)
                y_next = sl["stems"][i + 1].view(1).to(DEVICE)

                in_nll = (
                    F.cross_entropy(prev_s_logits.float(), sl["stems"][i].view(1).to(DEVICE)).item()
                    if (i > 0 and prev_s_logits is not None)
                    else None
                )

                s_log, _, _, _, hidden, prev_sym, act_steps, _, _, _ = model(
                    s_i, m_i, l_i, prev_edge=e_i, hidden=hidden, prev_sym=prev_sym,
                    cached_P_adj=cached_padj, cached_cb_norm=cached_cb
                )
                s_log_0 = s_log[:, 0, :]
                st = act_steps.item()
                next_nll = F.cross_entropy(s_log_0.float(), y_next).item()
                probs = F.softmax(s_log_0.float(), dim=-1)
                ent = -(probs * torch.log(probs + 1e-8)).sum().item()

                next_nlls.append(next_nll)
                pred_ents.append(ent)
                steps_aligned.append(st)
                if in_nll is not None:
                    in_nlls.append(in_nll)
                    steps_in.append(st)

                prev_s_logits = s_log_0
                scanned += 1

    if next_nlls:
        med_next = float(np.median(next_nlls))
        easy_next = [st for nll, st in zip(next_nlls, steps_aligned) if nll <= med_next]
        hard_next = [st for nll, st in zip(next_nlls, steps_aligned) if nll > med_next]
        if easy_next and hard_next:
            print(
                f"Val ACT (Aligned Stem NLL) : Easy(NLL<={med_next:.2f})={np.mean(easy_next):.3f} steps | "
                f"Hard(NLL>{med_next:.2f})={np.mean(hard_next):.3f} steps"
            )

        med_ent = float(np.median(pred_ents))
        low_ent = [st for h, st in zip(pred_ents, steps_aligned) if h <= med_ent]
        high_ent = [st for h, st in zip(pred_ents, steps_aligned) if h > med_ent]
        if low_ent and high_ent:
            print(
                f"Val ACT (Model Entropy)    : LowEnt(H<={med_ent:.2f})={np.mean(low_ent):.3f} steps | "
                f"HighEnt(H>{med_ent:.2f})={np.mean(high_ent):.3f} steps"
            )

    if in_nlls:
        med_in = float(np.median(in_nlls))
        easy_in = [st for nll, st in zip(in_nlls, steps_in) if nll <= med_in]
        hard_in = [st for nll, st in zip(in_nlls, steps_in) if nll > med_in]
        if easy_in and hard_in:
            print(
                f"Val ACT (Input Surprisal)  : EasyIn(NLL<={med_in:.2f})={np.mean(easy_in):.3f} steps | "
                f"HardIn(NLL>{med_in:.2f})={np.mean(hard_in):.3f} steps"
            )


def ablation_probe(model, val_batches, macro_active=False):
    """Quantifies every cognitive & tokenization subsystem in exact Bits-Per-Character (BPC)."""
    eval_batches = val_batches
    split_tag = "Val Set"
    eval_macro = macro_active
    if not eval_batches:
        sub_n = min(len(data_atomic["stems"]), 4096)
        sub_atom = {k: v[:sub_n] for k, v in data_atomic.items()}
        eval_batches = build_val_batches([pack_atomic_to_packets(sub_atom, macro_trie=None, max_pkt_len=CONFIG["max_pkt_len"])])
        eval_macro = False
        split_tag = "TrainSlice"
    if not eval_batches:
        return []

    print(f"\n--- Split Module Ablation Probe ({split_tag}, BPC Scorecard) ---")
    configs = [
        ("Full Model", {}),
        ("- No Stack (Both)", {"ablate_stack": True}),
        ("- No Stack Attn", {"ablate_stack_attn": True}),
        ("- No Stack Read", {"ablate_stack_read": True}),
        ("- No HoloMem", {"ablate_holo": True}),
        ("- No GraphEcho", {"ablate_graph": True}),
        ("- No RelOp Bind", {"ablate_rel_op": True}),
        ("- No DualClock Edge", {"ablate_edge": True}),
        ("- No Morph Phase", {"ablate_morph": True}),
        ("- No VQ Feedback", {"ablate_vq": True}),
        ("- No Rec Carry", {"ablate_carry": True}),
        ("- Fixed ACT (1x)", {"ablate_act_1step": True}),
    ]
    results = []
    base_bpc = None
    print(f"{'Configuration':<21} | {'BPC':<7} | {'Loss':<7} | {'StemAcc':<7} | {'dBPC vs Full'}")
    print("-" * 68)

    for label, kwargs in configs:
        loss_v, bpc_v, s_acc, _, _ = evaluate_val(model, eval_batches, macro_active=eval_macro, **kwargs)
        if bpc_v is None:
            continue
        if base_bpc is None:
            base_bpc = bpc_v
        delta = bpc_v - base_bpc
        print(f"{label:<21} | {bpc_v:<7.4f} | {loss_v:<7.4f} | {s_acc:6.1f}% | {delta:+10.4f}")
        results.append((label, bpc_v, loss_v, s_acc, delta))

    if eval_macro and val_atomic_slices:
        unpacked_batches = build_val_batches([
            pack_atomic_to_packets(va, macro_trie=None, max_pkt_len=CONFIG["max_pkt_len"])
            for va in val_atomic_slices
        ])
        u_loss, u_bpc, u_acc, _, _ = evaluate_val(model, unpacked_batches, macro_active=False)
        if u_bpc is not None:
            u_delta = u_bpc - base_bpc
            print(f"{'- Unpacked (L=1)':<21} | {u_bpc:<7.4f} | {u_loss:<7.4f} | {u_acc:6.1f}% | {u_delta:+10.4f}")
            results.append(("- Unpacked (L=1)", u_bpc, u_loss, u_acc, u_delta))

    return results


def plot_codebook_and_ablation(sorted_counts, ablation_results):
    fig, (ax_zipf, ax_abl) = plt.subplots(1, 2, figsize=(15.0, 6.2))

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
    ax_zipf.set_title("Concept Codebook Utilization (Zipf Rank-Frequency)")
    ax_zipf.set_xlabel("Codebook Symbol Rank")
    ax_zipf.set_ylabel("Hit Count (log scale)")
    ax_zipf.grid(True, alpha=0.25)
    ax_zipf.legend(loc="upper right", fontsize=8)

    if len(ablation_results) > 1:
        sub = ablation_results[1:]
        labels = [r[0] for r in sub]
        deltas = [r[4] for r in sub]
        colors = ['crimson' if d > 0.002 else ('seagreen' if d < -0.001 else 'slategray') for d in deltas]
        y_pos = np.arange(len(labels))
        bars = ax_abl.barh(y_pos, deltas, color=colors, alpha=0.85)
        ax_abl.axvline(0.0, color='black', linewidth=1.0)
        ax_abl.set_yticks(y_pos)
        ax_abl.set_yticklabels(labels, fontsize=9)
        ax_abl.invert_yaxis()
        max_abs = max(0.02, max(abs(d) for d in deltas))
        ax_abl.set_xlim(min(0.0, min(deltas) * 1.15), max_abs * 1.20)
        ax_abl.set_xlabel("Delta Val BPC vs. Full Model (+ = module helps)")
        ax_abl.set_title("Split Module & Tokenization Ablation (BPC Penalty)")
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
        ax_abl.set_title("Split Module & Tokenization Ablation")

    plt.tight_layout()
    plt.savefig("6_codebook_and_ablation.png", dpi=150)
    print("Saved 6_codebook_and_ablation.png")
    plt.close()


def generate_pddl_blocksworld_episodes(n_episodes=60, steps_per_ep=6):
    """Generates valid deterministic 3-block Blocksworld state-action transitions."""
    rng = random.Random(SEED)
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
    """Optional standalone Blocksworld/PDDL rule-inference sanity benchmark."""
    print("\n=== Running Optional Blocksworld/PDDL Symbolic Rule Sanity Benchmark ===")
    pddl_text = generate_pddl_blocksworld_episodes(n_episodes=80, steps_per_ep=6)
    tokens = pddl_text.replace("\n", " \n ").split(" ")
    tokens = [t for t in tokens if t != ""]
    vocab = ["<PAD>"] + sorted(set(tokens))
    tok_to_id = {t: i for i, t in enumerate(vocab)}
    ids = torch.tensor([tok_to_id[t] for t in tokens], dtype=torch.long)

    orig_cfg = CONFIG.copy()
    try:
        CONFIG["n_symbols"] = 64
        CONFIG["n_ops"] = 12
        CONFIG["stack_size"] = 12
        CONFIG["max_recursion_depth"] = 4
        CONFIG["embedding_dim"] = 64
        CONFIG["act_threshold"] = 0.85
        CONFIG["decoder_dropout"] = 0.0
        CONFIG["cell_dropout"] = 0.0
        CONFIG["emb_dropout"] = 0.0
        CONFIG["label_smoothing"] = 0.0

        probe_model = UberCRSN(len(vocab), 64, n_morph_classes=2, n_edge_classes=2).to(DEVICE)
        probe_model.sym_alpha.fill_(0.30)
        probe_model.graph_gate.fill_(1.0)
        opt = torch.optim.AdamW(probe_model.parameters(), lr=2e-3, weight_decay=1e-5)

        split = int(0.85 * len(ids))
        tr_ids = ids[:split].to(DEVICE)
        va_ids = ids[split:].to(DEVICE)
        seq = 16
        bs = max(1, (len(tr_ids) - 1) // seq)
        streams = torch.stack([tr_ids[b * seq : b * seq + seq + 1] for b in range(bs)], dim=0)
        ones_m = torch.ones(bs, 1, dtype=torch.long, device=DEVICE)
        ones_l = torch.ones(bs, dtype=torch.long, device=DEVICE)
        ones_e = torch.ones(bs, dtype=torch.long, device=DEVICE)

        for _ in range(epochs):
            probe_model.train()
            hidden, prev_sym = None, None
            loss_acc = 0.0
            for t in range(seq):
                x = streams[:, t].unsqueeze(1)
                y = streams[:, t + 1]
                out = probe_model(x, ones_m, ones_l, prev_edge=ones_e, hidden=hidden, prev_sym=prev_sym, return_stats=True)
                logits = out[0][:, 0, :]
                hidden = out[1]
                prev_sym = out[2]
                ponder_l = out[3]
                vq_l = out[4]
                eth_l = out[5]
                cons_l = out[15]
                extra = out[17]
                sym_lm_l = F.cross_entropy(extra["sym_logits"].float(), y)
                loss = (
                    F.cross_entropy(logits.float(), y)
                    + 0.20 * sym_lm_l
                    + 0.03 * ponder_l
                    + 0.5 * vq_l
                    + 0.3 * extra["op_vq_loss"]
                    + 0.1 * eth_l
                    + 0.02 * cons_l
                )
                loss_acc = loss_acc + loss
            opt.zero_grad(set_to_none=True)
            (loss_acc / seq).backward()
            torch.nn.utils.clip_grad_norm_(probe_model.parameters(), 1.5)
            opt.step()
            probe_model.vq_layer.normalize_codebook_weights_()
            probe_model.op_vq.normalize_codebook_weights_()

        probe_model.eval()
        hidden, prev_sym = None, None
        correct, sym_correct, total = 0, 0, 0
        act_on_next, act_on_other = [], []
        m1 = torch.ones(1, 1, dtype=torch.long, device=DEVICE)
        l1 = torch.ones(1, dtype=torch.long, device=DEVICE)
        e1 = torch.ones(1, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            for i in range(len(va_ids) - 1):
                x = va_ids[i].view(1, 1)
                y = va_ids[i + 1].item()
                out = probe_model(x, m1, l1, prev_edge=e1, hidden=hidden, prev_sym=prev_sym, return_stats=True)
                logits = out[0][:, 0, :]
                hidden = out[1]
                prev_sym = out[2]
                act_steps = out[11].item()
                sym_logits = out[17]["sym_logits"]
                correct += int(logits.argmax(dim=-1).item() == y)
                sym_correct += int(sym_logits.argmax(dim=-1).item() == y)
                total += 1
                if vocab[va_ids[i].item()] in ("->", "NEXT", "ACT"):
                    act_on_next.append(act_steps)
                else:
                    act_on_other.append(act_steps)

        acc = 100.0 * correct / max(1, total)
        sym_acc = 100.0 * sym_correct / max(1, total)
        print(
            f"PDDL Probe Val Accuracy: {acc:.1f}% (PureSym={sym_acc:.1f}%, {correct}/{total}) | "
            f"ACT(->/ACT/NEXT)={np.mean(act_on_next) if act_on_next else 0:.2f} vs "
            f"ACT(StateTok)={np.mean(act_on_other) if act_on_other else 0:.2f}"
        )
    finally:
        CONFIG.clear()
        CONFIG.update(orig_cfg)


# ==========================================
# 11. Main Execution
# ==========================================
def main():
    logger = initialize_runtime(data_file=DATA_FILE, log_file=TERMINAL_LOG_FILE)
    filename = "crsn_omni_model.pth"

    try:
        trained_model, train_history, macro_trie, macro_active, val_batches = train(save_path=filename)

        print(f"\n--- Saving Final Model to {filename} ---")
        save_checkpoint(trained_model, filename, macro_trie=macro_trie)
        print("Saved.")

        plot_training_dynamics(train_history)
        sym_stem_counts, sym_op_counts, sym_zipf_counts = visualize_all(trained_model, macro_active=macro_active)
        extract_logic_rules(trained_model, data_atomic, sym_stem_counts=sym_stem_counts)

        dream_mode(trained_model, sym_stem_counts=sym_stem_counts, sym_op_counts=sym_op_counts)
        anomaly_detector(trained_model)
        context_difficulty_probe(trained_model)
        ablation_results = ablation_probe(trained_model, val_batches, macro_active=macro_active)
        plot_codebook_and_ablation(sym_zipf_counts, ablation_results)

        if CONFIG.get("run_pddl_probe", False):
            run_pddl_probe()

        print(f"\nSaved complete terminal output log to '{TERMINAL_LOG_FILE}'.")
    finally:
        logger.close()

    try:
        import importlib
        colab_files = importlib.import_module("google.colab.files")
        for out_file in (
            filename,
            TERMINAL_LOG_FILE,
            "0_training_dynamics.png",
            "1_semantic_topology.png",
            "2_stack_mri.png",
            "3_act_profile.png",
            "4_phase_plot.png",
            "5_anomaly_detection.png",
            "6_codebook_and_ablation.png",
        ):
            if os.path.isfile(out_file):
                colab_files.download(out_file)
    except (ImportError, ModuleNotFoundError):
        pass


if __name__ == "__main__":
    main()