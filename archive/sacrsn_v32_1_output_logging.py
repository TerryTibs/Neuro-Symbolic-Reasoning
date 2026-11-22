"""
SACRSN v32.1 — stability/diagnostic and output-integrity build

This is a corrected research prototype based on the supplied SACRSN v31.2
source. It is not a theorem prover and does not establish general reasoning.

Key corrections:
- VQ is genuinely bypassed when use_vq=False (no random codebook in control).
- "steps" means optimizer updates; each update samples one batch.
- ACT is explicitly named soft halting aggregation, not computational early exit.
  act_epsilon is removed because the old implementation never used it.
- Graph loss is an auxiliary latent-symbol transition objective only.
- Hard and soft codebook usage are reported separately.
- Train-only vocabulary; validation-only characters are handled as unknown.
- Correct sampling bounds and input/config validation.
- Per-module gradient diagnostics and finite checks before/after optimizer step.
- Stack remains a bounded differentiable memory, not a discrete pushdown stack.
- Stack semantics are tested by built-in deterministic self-tests.
- Fresh initialisation only; no checkpoint loading or resume path.
- Each run receives a unique output directory.

Requirements:
    pip install torch numpy

Run:
    python sacrsn_v32.py
    python sacrsn_v32.py --steps 1000 --ablation full
    python sacrsn_v32.py --ablation control
    python sacrsn_v32.py --text-file corpus.txt --steps 3000
    python sacrsn_v32.py --self-test

All outputs are written beneath runs/v32_0/fresh_...
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
import uuid
import sys
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


VERSION = "32.1"
UNKNOWN_TOKEN = "\uFFFD"

DEFAULT_TEXT = """True, without falsehood, certain and most true.
That which is above is like to that which is below,
and that which is below is like to that which is above.
The father of all perfection in the whole world is here.
Its force or power is entire if it be converted into earth."""


@dataclass
class Config:
    embedding_dim: int = 64
    n_symbols: int = 64
    max_recursion_depth: int = 8
    stack_size: int = 16

    use_stack: bool = True
    use_vq: bool = True
    use_graph: bool = True
    use_diversity: bool = True
    use_ponder: bool = True

    commitment_cost: float = 0.25
    diversity_weight: float = 0.05
    graph_weight: float = 0.10
    vq_weight: float = 1.0
    ponder_weight: float = 0.001

    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    batch_size: int = 16
    sequence_length: int = 32
    steps: int = 1000

    validation_fraction: float = 0.20
    seed: int = 42
    log_every: int = 25

    codebook_temperature: float = 1.0
    eps: float = 1e-8
    early_stopping_patience: int = 20
    minimum_improvement: float = 1e-5
    ablation: str = "full"


ABLATIONS = {
    "control": dict(use_vq=False, use_graph=False, use_diversity=False,
                    use_ponder=False, use_stack=False),
    "vq": dict(use_vq=True, use_graph=False, use_diversity=False,
               use_ponder=False, use_stack=False),
    "diversity": dict(use_vq=True, use_graph=False, use_diversity=True,
                      use_ponder=False, use_stack=False),
    "graph": dict(use_vq=True, use_graph=True, use_diversity=True,
                  use_ponder=False, use_stack=False),
    "full": dict(use_vq=True, use_graph=True, use_diversity=True,
                 use_ponder=True, use_stack=True),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Warn rather than forcing deterministic kernels that may be unavailable.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def complex_cat(z: torch.Tensor) -> torch.Tensor:
    return torch.cat((z.real, z.imag), dim=-1)


def finite_or_raise(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        bad = (~torch.isfinite(tensor)).sum().item()
        raise FloatingPointError(f"{name}: {bad} non-finite value(s)")


def validate_config(cfg: Config) -> None:
    positive = {
        "embedding_dim": cfg.embedding_dim,
        "n_symbols": cfg.n_symbols,
        "max_recursion_depth": cfg.max_recursion_depth,
        "stack_size": cfg.stack_size,
        "batch_size": cfg.batch_size,
        "sequence_length": cfg.sequence_length,
        "steps": cfg.steps,
        "log_every": cfg.log_every,
    }
    for name, value in positive.items():
        if value < 1:
            raise ValueError(f"{name} must be >= 1")
    if not 0.0 < cfg.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if cfg.learning_rate <= 0 or cfg.grad_clip <= 0:
        raise ValueError("learning_rate and grad_clip must be positive")
    if cfg.codebook_temperature <= 0:
        raise ValueError("codebook_temperature must be positive")


# ---------------------------------------------------------------------
# Complex-valued layers
# ---------------------------------------------------------------------

class ComplexMagnitudeNorm(nn.Module):
    """Normalises magnitudes only; preserves complex phase."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = z.abs()
        mean = mag.mean(dim=-1, keepdim=True)
        var = mag.var(dim=-1, unbiased=False, keepdim=True)
        norm = (mag - mean) * torch.rsqrt(var + self.eps)
        norm = F.softplus(norm * self.weight + self.bias)
        return torch.polar(norm, torch.angle(z))


class ModReLU(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = z.abs()
        scale = F.relu(mag + self.bias) / (mag + self.eps)
        return z * scale


class ComplexLinear(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.real = nn.Linear(dim, dim, bias=False)
        self.imag = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.real.weight)
        nn.init.orthogonal_(self.imag.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        real = self.real(z.real) - self.imag(z.imag)
        imag = self.real(z.imag) + self.imag(z.real)
        return torch.complex(real, imag)


class ComplexAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q = ComplexLinear(dim)
        self.k = ComplexLinear(dim)
        self.v = ComplexLinear(dim)
        self.scale = (2.0 * dim) ** -0.5

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = complex_cat(self.q(query)).unsqueeze(1)
        k = complex_cat(self.k(memory))
        scores = torch.matmul(q, k.transpose(1, 2)) * self.scale
        weights = F.softmax(scores, dim=-1)
        values = self.v(memory)
        out_r = torch.matmul(weights, values.real).squeeze(1)
        out_i = torch.matmul(weights, values.imag).squeeze(1)
        return torch.complex(out_r, out_i)


# ---------------------------------------------------------------------
# Differentiable bounded memory stack
# ---------------------------------------------------------------------

class DifferentiableStack(nn.Module):
    """
    A soft pointer-controlled bounded memory.

    It is deliberately NOT described as a conventional pushdown stack:
    pop moves the pointer but does not erase or shift stored memory.
    Push writes a weighted update at the probability mass moved upward.
    """
    def __init__(self, dim: int, size: int):
        super().__init__()
        if size < 1:
            raise ValueError("stack size must be >= 1")
        self.dim = dim
        self.size = size

    def normalise_pointer(self, ptr: torch.Tensor) -> torch.Tensor:
        mass = ptr.sum(dim=-1, keepdim=True)
        normed = ptr / mass.clamp_min(1e-8)
        fallback = torch.zeros_like(ptr)
        fallback[:, 0] = 1.0
        valid = torch.isfinite(mass) & (mass > 1e-8)
        return torch.where(valid, normed, fallback)

    def forward(self, z, memory, ptr, control):
        if ptr.shape[-1] != self.size:
            raise ValueError("pointer dimension does not match stack size")
        if memory.shape[1] != self.size or memory.shape[-1] != 2 * self.dim:
            raise ValueError("memory shape does not match stack configuration")

        push, pop, noop = control.unbind(dim=-1)
        push, pop, noop = (v.unsqueeze(-1) for v in (push, pop, noop))
        ptr = self.normalise_pointer(ptr)

        up = torch.zeros_like(ptr)
        down = torch.zeros_like(ptr)
        if self.size > 1:
            up[:, 1:] = ptr[:, :-1]
            down[:, :-1] = ptr[:, 1:]
        up[:, -1] += ptr[:, -1]       # retain upper-boundary mass
        down[:, 0] += ptr[:, 0]       # retain lower-boundary mass

        candidate = self.normalise_pointer(push * up + pop * down + noop * ptr)

        write_distribution = (push * up).clamp(0.0, 1.0)
        z_flat = complex_cat(z).unsqueeze(1)
        write = write_distribution.unsqueeze(-1)
        new_memory = write * z_flat + (1.0 - write) * memory

        read_flat = (new_memory * candidate.unsqueeze(-1)).sum(dim=1)
        read = torch.complex(read_flat[..., :self.dim], read_flat[..., self.dim:])
        return read, new_memory, candidate


# ---------------------------------------------------------------------
# VQ and auxiliary graph
# ---------------------------------------------------------------------

class SymbolQuantizer(nn.Module):
    def __init__(self, dim: int, n_symbols: int, temperature: float = 1.0):
        super().__init__()
        self.n_symbols = n_symbols
        self.temperature = temperature
        self.codebook = nn.Parameter(torch.empty(n_symbols, 2 * dim))
        nn.init.uniform_(self.codebook, -0.25, 0.25)

    def forward(self, z):
        flat = complex_cat(z)
        distances = (
            flat.square().sum(-1, keepdim=True)
            + self.codebook.square().sum(-1).unsqueeze(0)
            - 2.0 * flat @ self.codebook.t()
        ).clamp_min(0.0)
        soft = F.softmax(-distances / self.temperature, dim=-1)
        indices = distances.argmin(dim=-1)
        quantized = F.embedding(indices, self.codebook)
        codebook_loss = F.mse_loss(quantized, flat.detach())
        commitment_loss = F.mse_loss(flat, quantized.detach())
        straight_through = flat + (quantized - flat).detach()
        d = z.shape[-1]
        z_q = torch.complex(straight_through[..., :d],
                            straight_through[..., d:])
        return z_q, indices, soft, codebook_loss, commitment_loss


class TransitionGraph(nn.Module):
    """Auxiliary P(symbol_t | symbol_(t-1)); not connected to decoder."""
    def __init__(self, n_symbols: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_symbols, n_symbols))

    def loss(self, previous, current):
        if previous is None:
            return self.logits.sum() * 0.0
        return F.cross_entropy(self.logits[previous.reshape(-1)],
                               current.reshape(-1))

    def probabilities(self):
        return F.softmax(self.logits, dim=-1)


# ---------------------------------------------------------------------
# Recursive cell and model
# ---------------------------------------------------------------------

class AdaptiveRecursiveCell(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = ComplexLinear(dim)
        self.norm = ComplexMagnitudeNorm(dim)
        self.activation = ModReLU(dim)
        self.attention = ComplexAttention(dim)
        self.halt = nn.Linear(2 * dim, 1)
        self.stack_control = nn.Linear(2 * dim, 3)
        nn.init.constant_(self.halt.bias, -1.5)

    def forward(self, z, history):
        base_processed = self.activation(self.norm(self.linear(z)))
        attention_contribution = torch.zeros_like(base_processed)
        if history is not None and history.size(1):
            attention_contribution = self.attention(base_processed, history)
        processed = base_processed + attention_contribution
        flat = complex_cat(processed)
        halt_probability = torch.sigmoid(self.halt(flat))
        stack_control = F.softmax(self.stack_control(flat), dim=-1)
        return (processed, halt_probability, stack_control,
                base_processed, attention_contribution)


@dataclass
class ModelState:
    recurrent: torch.Tensor
    stack_memory: torch.Tensor
    stack_pointer: torch.Tensor
    previous_symbol: Optional[torch.Tensor]


class SACRSN(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        self.config = cfg
        d = cfg.embedding_dim
        self.embedding_magnitude = nn.Embedding(vocab_size, d)
        self.embedding_phase = nn.Embedding(vocab_size, d)
        self.cell = AdaptiveRecursiveCell(d)
        self.quantizer = SymbolQuantizer(d, cfg.n_symbols, cfg.codebook_temperature)
        self.graph = TransitionGraph(cfg.n_symbols)
        self.decoder = nn.Linear(2 * d, vocab_size)
        self.stack = DifferentiableStack(d, cfg.stack_size) if cfg.use_stack else None

    def initial_state(self, batch_size: int, device) -> ModelState:
        d = self.config.embedding_dim
        recurrent = torch.zeros(batch_size, d, dtype=torch.complex64, device=device)
        memory = torch.zeros(batch_size, self.config.stack_size, 2 * d, device=device)
        pointer = torch.zeros(batch_size, self.config.stack_size, device=device)
        pointer[:, 0] = 1.0
        return ModelState(recurrent, memory, pointer, None)

    def embed(self, token_ids):
        magnitude = F.softplus(self.embedding_magnitude(token_ids))
        phase = math.pi * torch.tanh(self.embedding_phase(token_ids))
        return torch.polar(magnitude, phase)

    def forward_step(self, token_ids, state=None, collect_stability=False):
        batch = token_ids.shape[0]
        if state is None:
            state = self.initial_state(batch, token_ids.device)

        z = 0.5 * self.embed(token_ids) + 0.5 * state.recurrent
        memory, pointer = state.stack_memory, state.stack_pointer
        history = []
        weighted_state = torch.zeros_like(z)
        remaining = torch.ones(batch, 1, device=z.device)
        expected_steps = torch.zeros(batch, 1, device=z.device)

        halt_masses, vq_losses, soft_assignments = [], [], []
        hard_indices = []
        stack_depths = []
        stability_trace = []

        for step in range(self.config.max_recursion_depth):
            history_tensor = torch.stack(history, dim=1) if history else None
            z_proc, halt_prob, stack_control, base_processed, attention_contribution = self.cell(z, history_tensor)

            if self.stack is not None:
                stack_read, memory, pointer = self.stack(
                    z_proc, memory, pointer, stack_control)
                z_combined = z_proc + stack_read
                positions = torch.arange(self.config.stack_size,
                                         device=z.device, dtype=z.real.dtype)
                stack_depths.append((pointer * positions).sum(-1))
            else:
                stack_read = torch.zeros_like(z_proc)
                z_combined = z_proc

            # Critical correction: no quantizer call or random codebook path
            # is used by the no-VQ control condition.
            if self.config.use_vq:
                z_symbol, indices, soft, cb_loss, commit_loss = self.quantizer(z_combined)
                vq_loss = cb_loss + self.config.commitment_cost * commit_loss
                hard_indices.append(indices)
                soft_assignments.append(soft)
            else:
                z_symbol = z_combined
                indices = None
                vq_loss = z_combined.real.sum() * 0.0

            vq_losses.append(vq_loss)

            # Soft halting aggregation: all recursion steps are executed.
            # This is not computational early-exit ACT.
            halt_mass = remaining if step == self.config.max_recursion_depth - 1 \
                else remaining * halt_prob
            halt_masses.append(halt_mass)
            weighted_state = weighted_state + halt_mass * z_combined
            expected_steps = expected_steps + remaining
            remaining = (remaining - halt_mass).clamp_min(0.0)

            history.append(z_proc)
            z_next = 0.7 * z_combined + 0.3 * z_symbol
            if collect_stability:
                def rms(value):
                    return float(value.detach().abs().square().mean().sqrt().cpu())
                stability_trace.append({
                    "depth": step + 1,
                    "input_rms": rms(z),
                    "base_cell_output_rms": rms(base_processed),
                    "attention_contribution_rms": rms(attention_contribution),
                    "post_attention_rms": rms(z_proc),
                    "stack_read_rms": rms(stack_read),
                    "post_stack_rms": rms(z_combined),
                    "quantized_or_bypass_output_rms": rms(z_symbol),
                    "next_recurrent_input_rms": rms(z_next),
                })
            z = z_next

        if self.config.use_vq:
            final_indices = hard_indices[-1]
            if self.config.use_graph and state.previous_symbol is not None:
                graph_loss = self.graph.loss(state.previous_symbol, final_indices)
            else:
                graph_loss = self.graph.logits.sum() * 0.0
            soft_tensor = torch.stack(soft_assignments, dim=0)
            hard_tensor = torch.stack(hard_indices, dim=0)
        else:
            final_indices = None
            graph_loss = self.graph.logits.sum() * 0.0
            soft_tensor = None
            hard_tensor = None

        logits = self.decoder(complex_cat(weighted_state))
        halt_distribution = torch.stack(halt_masses, dim=0)
        total_halt_mass = halt_distribution.sum(dim=0)

        next_state = ModelState(weighted_state, memory, pointer, final_indices)
        metrics = {
            "ponder": expected_steps.mean(),
            "vq_loss": torch.stack(vq_losses).mean(),
            "graph_loss": graph_loss,
            "soft_assignments": soft_tensor,
            "hard_indices": hard_tensor,
            "symbol_indices": final_indices,
            "halt_mass": total_halt_mass,
            "halt_distribution": halt_distribution,
            "expected_steps": expected_steps,
            "stack_depth": torch.stack(stack_depths).mean() if stack_depths
                           else logits.new_tensor(0.0),
            "stack_pointer_mass": pointer.sum(dim=-1).mean(),
            "stability_trace": stability_trace if collect_stability else None,
        }
        return logits, next_state, metrics


# ---------------------------------------------------------------------
# Data, objective and evaluation
# ---------------------------------------------------------------------

def split_text(text: str, val_fraction: float):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) < 12:
        raise ValueError("Text corpus is too short; provide at least 12 characters.")
    split = int(len(text) * (1.0 - val_fraction))
    split = max(2, min(len(text) - 2, split))
    train_text, val_text = text[:split], text[split:]

    # Build vocabulary only from training data. Unknown validation characters
    # map to a dedicated token, avoiding validation-to-vocabulary leakage.
    chars = sorted(set(train_text))
    if UNKNOWN_TOKEN not in chars:
        chars.append(UNKNOWN_TOKEN)
    mapping = {c: i for i, c in enumerate(chars)}
    unk = mapping[UNKNOWN_TOKEN]
    train_ids = torch.tensor([mapping[c] for c in train_text], dtype=torch.long)
    val_ids = torch.tensor([mapping.get(c, unk) for c in val_text], dtype=torch.long)
    return chars, mapping, train_ids, val_ids, train_text, val_text


def sample_batch(data, batch_size, sequence_length, device):
    max_start = len(data) - sequence_length - 1
    if max_start < 0:
        raise ValueError(
            f"Training split ({len(data)} chars) must exceed "
            f"sequence_length+1 ({sequence_length + 1})."
        )
    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([data[s:s + sequence_length] for s in starts]).to(device)
    y = torch.stack([data[s + 1:s + sequence_length + 1] for s in starts]).to(device)
    return x, y


def sequence_objective(model, x, y, cfg):
    state = None
    prediction_losses, vq_losses, graph_losses, ponder_losses = [], [], [], []
    soft_assignments, hard_indices = [], []
    halt_errors, stack_errors = [], []

    for t in range(x.size(1)):
        logits, state, metrics = model.forward_step(x[:, t], state)
        prediction_losses.append(F.cross_entropy(logits, y[:, t]))
        vq_losses.append(metrics["vq_loss"])
        graph_losses.append(metrics["graph_loss"])
        ponder_losses.append(metrics["ponder"])
        halt_errors.append((metrics["halt_mass"] - 1.0).abs().mean())
        stack_errors.append((state.stack_pointer.sum(-1) - 1.0).abs().mean())
        if metrics["soft_assignments"] is not None:
            # [depth, batch, symbols] -> [depth*batch, symbols]
            soft_assignments.append(metrics["soft_assignments"].reshape(-1, cfg.n_symbols))
            hard_indices.append(metrics["hard_indices"].reshape(-1))

    prediction = torch.stack(prediction_losses).mean()
    vq = torch.stack(vq_losses).mean()
    graph = torch.stack(graph_losses).mean()
    ponder = torch.stack(ponder_losses).mean()

    zero = prediction * 0.0
    entropy = zero
    soft_perplexity = zero
    hard_perplexity = zero
    hard_active = zero
    soft_active = zero
    diversity = zero

    if cfg.use_vq and soft_assignments:
        soft = torch.cat(soft_assignments, dim=0)
        usage = soft.mean(0)
        usage = usage / usage.sum().clamp_min(cfg.eps)
        entropy = -(usage * usage.clamp_min(cfg.eps).log()).sum()
        soft_perplexity = entropy.exp()
        soft_active = (usage > (1.0 / cfg.n_symbols) * 0.25).sum()

        hard = torch.cat(hard_indices)
        hard_counts = torch.bincount(hard, minlength=cfg.n_symbols).float()
        hard_usage = hard_counts / hard_counts.sum().clamp_min(1.0)
        hard_entropy = -(hard_usage * hard_usage.clamp_min(cfg.eps).log()).sum()
        hard_perplexity = hard_entropy.exp()
        hard_active = (hard_counts > 0).sum()

        diversity = -entropy if cfg.use_diversity else zero

    total = prediction
    if cfg.use_vq:
        total = total + cfg.vq_weight * vq
    if cfg.use_graph:
        total = total + cfg.graph_weight * graph
    if cfg.use_ponder:
        total = total + cfg.ponder_weight * ponder
    if cfg.use_diversity:
        total = total + cfg.diversity_weight * diversity

    return total, {
        "prediction": prediction, "vq": vq, "graph": graph,
        "ponder": ponder, "diversity": diversity, "entropy": entropy,
        "soft_usage_perplexity": soft_perplexity,
        "hard_usage_perplexity": hard_perplexity,
        "soft_active_symbols": soft_active,
        "hard_active_symbols": hard_active,
        "halt_mass_error": torch.stack(halt_errors).mean(),
        "stack_mass_error": torch.stack(stack_errors).mean(),
    }


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    if len(data) < 2:
        return float("nan")
    state, losses = None, []
    for i in range(len(data) - 1):
        logits, state, _ = model.forward_step(data[i:i + 1].to(device), state)
        losses.append(F.cross_entropy(logits, data[i + 1:i + 2].to(device)).item())
    return float(np.mean(losses))


# ---------------------------------------------------------------------
# Diagnostics and persistence
# ---------------------------------------------------------------------

def gradient_diagnostics(model):
    total_sq = 0.0
    per_module = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        finite_or_raise(f"gradient {name}", parameter.grad)
        norm = float(parameter.grad.detach().norm(2).item())
        total_sq += norm * norm
        root = name.split(".")[0]
        per_module[root] = per_module.get(root, 0.0) + norm * norm
    return math.sqrt(total_sq), {
        k: math.sqrt(v) for k, v in sorted(per_module.items())
    }


def check_parameters_finite(model):
    for name, parameter in model.named_parameters():
        finite_or_raise(f"parameter {name}", parameter.data)


def append_csv(path: Path, row: dict):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def train_model(model, train_data, val_data, cfg, device, output_dir):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.steps, eta_min=cfg.learning_rate * 0.05)

    best_val, best_step, best_state = float("inf"), 0, None
    evaluations_without_improvement = 0
    history = []
    started = time.time()
    print(f"Device: {device} | parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Fresh initialisation: no checkpoint is loaded.")
    print("A step is one optimiser update, not a full corpus epoch.")

    step = 0
    for step in range(1, cfg.steps + 1):
        model.train()
        x, y = sample_batch(train_data, cfg.batch_size, cfg.sequence_length, device)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = sequence_objective(model, x, y, cfg)
        finite_or_raise("training loss", loss)
        loss.backward()

        pre_clip, module_grads = gradient_diagnostics(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip,
                                       error_if_nonfinite=True)
        post_clip, _ = gradient_diagnostics(model)
        optimizer.step()
        scheduler.step()
        check_parameters_finite(model)

        row = {
            "step": step,
            "total_loss": float(loss.detach()),
            "prediction_loss": float(metrics["prediction"].detach()),
            "vq_loss": float(metrics["vq"].detach()),
            "graph_loss": float(metrics["graph"].detach()),
            "ponder_expected_steps": float(metrics["ponder"].detach()),
            "diversity_loss": float(metrics["diversity"].detach()),
            "soft_usage_perplexity": float(metrics["soft_usage_perplexity"].detach()),
            "hard_usage_perplexity": float(metrics["hard_usage_perplexity"].detach()),
            "soft_active_symbols": int(metrics["soft_active_symbols"].detach()),
            "hard_active_symbols": int(metrics["hard_active_symbols"].detach()),
            "pre_clip_grad_norm": pre_clip,
            "post_clip_grad_norm": post_clip,
            "clipping_ratio": post_clip / max(pre_clip, 1e-12),
            "halt_mass_error": float(metrics["halt_mass_error"].detach()),
            "stack_mass_error": float(metrics["stack_mass_error"].detach()),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "grad_by_module_json": json.dumps(module_grads, sort_keys=True),
            "quantizer_codebook_grad_norm": (
                float(model.quantizer.codebook.grad.detach().norm(2).item())
                if model.config.use_vq and model.quantizer.codebook.grad is not None
                else 0.0
            ),
        }

        validation_due = step == 1 or step % cfg.log_every == 0 or step == cfg.steps
        if validation_due:
            val_nll = evaluate(model, val_data, device)
            row["validation_nll"] = val_nll
            print(f"Step {step:05d}/{cfg.steps} | train={row['prediction_loss']:.4f} "
                  f"| val={val_nll:.4f} | hard-ppx={row['hard_usage_perplexity']:.2f} "
                  f"| active={row['hard_active_symbols']} | grad={pre_clip:.3g}")

            if math.isfinite(val_nll) and val_nll < best_val - cfg.minimum_improvement:
                best_val, best_step = val_nll, step
                evaluations_without_improvement = 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                torch.save({
                    "model_state_dict": best_state,
                    "config": asdict(cfg),
                    "best_validation_nll": best_val,
                    "best_step": best_step,
                    "version": VERSION,
                }, output_dir / "best_model.pt")
            else:
                evaluations_without_improvement += 1

        history.append(row)
        append_csv(output_dir / "training_metrics.csv", row)

        if validation_due and evaluations_without_improvement >= cfg.early_stopping_patience:
            print("Early stopping: validation did not improve.")
            break

    completed = step
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "steps_completed": completed,
        "version": VERSION,
    }, output_dir / "final_model.pt")

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - started
    report = {
        "version": VERSION,
        "config": asdict(cfg),
        "steps_completed": completed,
        "best_step": best_step,
        "best_validation_nll": best_val,
        "elapsed_seconds": elapsed,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "history": history,
        "interpretation": "Character-level prediction plus auxiliary latent-symbol objectives; not evidence of general reasoning.",
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(json_safe(report), indent=2), encoding="utf-8")
    return report


@torch.no_grad()
def collect_symbol_transitions(model, data, device):
    model.eval()
    state, counts, transitions = None, Counter(), Counter()
    previous = None
    if not model.config.use_vq:
        return counts, transitions
    for token in data:
        _, state, metrics = model.forward_step(token.view(1).to(device), state)
        current = int(metrics["symbol_indices"].item())
        counts[current] += 1
        if previous is not None:
            transitions[(previous, current)] += 1
        previous = current
    return counts, transitions


@torch.no_grad()
def collect_act_diagnostics(model, data, device):
    model.eval()
    state, depths, mass_errors = None, [], []
    for token in data:
        _, state, metrics = model.forward_step(token.view(1).to(device), state)
        depths.append(float(metrics["ponder"]))
        mass_errors.append(float((metrics["halt_mass"] - 1.0).abs().mean()))
    return {
        "mean_expected_depth": float(np.mean(depths)),
        "median_expected_depth": float(np.median(depths)),
        "minimum_expected_depth": float(np.min(depths)),
        "maximum_expected_depth": float(np.max(depths)),
        "maximum_halt_mass_error": float(np.max(mass_errors)),
        "all_recursion_steps_executed": True,
        "computational_early_exit": False,
        "description": "Differentiable weighted aggregation; not early-exit ACT.",
    }


@torch.no_grad()
def anomaly_report(model, data, device, output_dir):
    model.eval()
    state, scores = None, []
    for i in range(len(data) - 1):
        logits, state, _ = model.forward_step(data[i:i + 1].to(device), state)
        target = data[i + 1:i + 2].to(device)
        scores.append(float(F.cross_entropy(logits, target)))
    if not scores:
        return {}
    report = {
        "method": "next-character negative log likelihood",
        "score_count": len(scores),
        "mean": float(np.mean(scores)),
        "standard_deviation": float(np.std(scores)),
        "p95_threshold": float(np.quantile(scores, 0.95)),
        "maximum": float(np.max(scores)),
        "warning": "This is a tiny-split statistical threshold, not semantic anomaly detection.",
    }
    (output_dir / "anomaly_calibration.json").write_text(json.dumps(report, indent=2),
                                                          encoding="utf-8")
    return report


@torch.no_grad()
def generate_text(model, start_text, chars, mapping, max_new_tokens, device,
                  temperature=0.8):
    model.eval()
    if not start_text:
        start_text = chars[0]
    ids = [mapping.get(c, mapping[UNKNOWN_TOKEN]) for c in start_text]
    state = None
    for token in ids:
        _, state, _ = model.forward_step(torch.tensor([token], device=device), state)
    current = torch.tensor([ids[-1]], device=device)
    output = list(start_text)
    for _ in range(max_new_tokens):
        logits, state, _ = model.forward_step(current, state)
        probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
        current = torch.multinomial(probs, 1).squeeze(-1)
        output.append(chars[int(current.item())])
    return "".join(output)


@torch.no_grad()
def collect_stability_diagnostics(model, data, device):
    """Measure recurrent/component magnitudes at every recursion depth."""
    model.eval()
    collected = {}
    state = None
    for token in data:
        _, state, metrics = model.forward_step(
            token.view(1).to(device), state, collect_stability=True
        )
        for item in metrics["stability_trace"]:
            depth = item["depth"]
            bucket = collected.setdefault(depth, {k: [] for k in item if k != "depth"})
            for key, value in item.items():
                if key != "depth":
                    bucket[key].append(value)
    per_depth = {}
    for depth, bucket in collected.items():
        per_depth[str(depth)] = {
            key: {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "maximum": float(np.max(values)),
            }
            for key, values in bucket.items()
        }
    return {
        "method": "Per-token forward diagnostic; RMS complex magnitude at each recursion depth",
        "tokens_measured": len(data),
        "per_depth": per_depth,
        "interpretation": "Compare growth across depth and component boundaries. These are descriptive measurements, not proof of causal instability.",
    }


def save_diagnostics(model, train_data, val_data, device, output_dir):
    stability = collect_stability_diagnostics(model, train_data, device)
    write_json(output_dir / "stability_diagnostics.json", stability)
    counts, transitions = collect_symbol_transitions(model, train_data, device)
    symbol_report = {
        "symbol_counts": dict(counts),
        "transitions": [
            {"from_symbol": a, "to_symbol": b, "count": n}
            for (a, b), n in transitions.most_common()
        ],
        "warning": "Empirical latent transitions, not extracted logical rules.",
    }
    (output_dir / "symbol_transitions.json").write_text(
        json.dumps(symbol_report, indent=2), encoding="utf-8")

    if model.config.use_vq:
        np.savetxt(output_dir / "transition_matrix.csv",
                   model.graph.probabilities().detach().cpu().numpy(),
                   delimiter=",", fmt="%.7f")
    act = collect_act_diagnostics(model, val_data, device)
    (output_dir / "act_diagnostics.json").write_text(json.dumps(act, indent=2),
                                                     encoding="utf-8")

    stack_report = {
        "stack_enabled": model.stack is not None,
        "stack_size": model.config.stack_size,
        "semantics": "Soft bounded pointer-controlled memory; pop does not erase/shift memory.",
        "not_claimed": "A conventional discrete pushdown stack.",
    }
    (output_dir / "stack_diagnostics.json").write_text(
        json.dumps(stack_report, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------
# Built-in tests
# ---------------------------------------------------------------------

def run_self_tests():
    torch.manual_seed(123)
    # Pointer mass is conserved, including both boundaries and size=1.
    for size in (1, 2, 5):
        stack = DifferentiableStack(dim=3, size=size)
        batch = 4
        ptr = torch.softmax(torch.randn(batch, size), dim=-1)
        mem = torch.randn(batch, size, 6)
        z = torch.complex(torch.randn(batch, 3), torch.randn(batch, 3))
        ctrl = torch.softmax(torch.randn(batch, 3), dim=-1)
        read, new_mem, new_ptr = stack(z, mem, ptr, ctrl)
        assert read.shape == (batch, 3)
        assert new_mem.shape == mem.shape
        assert torch.allclose(new_ptr.sum(-1), torch.ones(batch), atol=1e-6)
        assert torch.isfinite(read.real).all() and torch.isfinite(read.imag).all()

    # VQ disabled must be independent of the codebook and emit no symbol IDs.
    cfg = Config(embedding_dim=8, n_symbols=7, max_recursion_depth=3,
                 stack_size=4, use_stack=False, use_vq=False,
                 use_graph=False, use_diversity=False, use_ponder=False)
    model = SACRSN(vocab_size=5, cfg=cfg)
    tokens = torch.tensor([1, 2])
    out1, state1, m1 = model.forward_step(tokens)
    with torch.no_grad():
        model.quantizer.codebook.add_(1000.0)
    out2, state2, m2 = model.forward_step(tokens)
    assert m1["symbol_indices"] is None and m2["symbol_indices"] is None
    assert torch.allclose(out1, out2, atol=1e-6)
    assert state1.previous_symbol is None and state2.previous_symbol is None

    # ACT mass is one, and all configured recursive steps are evaluated.
    cfg2 = Config(embedding_dim=8, n_symbols=7, max_recursion_depth=4,
                  stack_size=4, use_stack=False, use_vq=True,
                  use_graph=False, use_diversity=False, use_ponder=True)
    model2 = SACRSN(vocab_size=5, cfg=cfg2)
    _, _, met = model2.forward_step(tokens)
    assert torch.allclose(met["halt_mass"], torch.ones(2, 1), atol=1e-6)
    assert met["halt_distribution"].shape[0] == cfg2.max_recursion_depth

    # Sampling must include the final legal start position without shape errors.
    data = torch.arange(20)
    x, y = sample_batch(data, 4, 5, torch.device("cpu"))
    assert x.shape == y.shape == (4, 5)
    assert torch.equal(x[:, 1:], y[:, :-1])

    print("SELF-TESTS PASSED: stack mass, VQ bypass, halt mass, batch sampling.")


def write_manifest(cfg, chars, train_data, val_data, text, output_dir, device):
    manifest = {
        "version": VERSION,
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": str(uuid.uuid4()),
        "seed": cfg.seed,
        "device": str(device),
        "config": asdict(cfg),
        "vocabulary_size_including_unk": len(chars),
        "training_characters": len(train_data),
        "validation_characters": len(val_data),
        "corpus_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "split": "contiguous; vocabulary fitted on training segment only",
        "fresh_initialisation": True,
        "checkpoint_loaded": False,
        "source_file": Path(__file__).name,
        "limitations": [
            "Small corpus results are not evidence of generalisation.",
            "Graph is auxiliary and does not affect decoder logits.",
            "All recursion steps execute; no computational ACT early exit.",
            "Stack is soft bounded memory, not a discrete pushdown stack.",
        ],
    }
    (output_dir / "config.json").write_text(json.dumps(manifest, indent=2),
                                            encoding="utf-8")



class Tee:
    """Mirror terminal output to both the original stream and a UTF-8 log."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    @property
    def encoding(self):
        return "utf-8"


def write_json(path: Path, payload) -> None:
    """Write JSON atomically so an interrupted process cannot leave half a file."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=f"SACRSN v{VERSION}")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of optimiser updates (not corpus epochs).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Deprecated alias for --steps.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--text-file", type=str, default=None)
    parser.add_argument("--ablation", choices=tuple(ABLATIONS), default="full")
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_tests()
        return

    cfg = Config(seed=args.seed, ablation=args.ablation)
    if args.steps is not None:
        cfg.steps = args.steps
    if args.epochs is not None:
        if args.steps is not None:
            parser.error("Use only one of --steps or --epochs.")
        cfg.steps = args.epochs
    if args.sequence_length is not None:
        cfg.sequence_length = args.sequence_length
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.log_every is not None:
        cfg.log_every = args.log_every
    for key, value in ABLATIONS[args.ablation].items():
        setattr(cfg, key, value)
    validate_config(cfg)

    set_seed(cfg.seed)
    corpus = Path(args.text_file).read_text(encoding="utf-8") if args.text_file else DEFAULT_TEXT
    chars, mapping, train_data, val_data, train_text, val_text = split_text(
        corpus, cfg.validation_fraction)

    minimum = cfg.sequence_length + 2
    if len(train_data) < minimum:
        raise ValueError(
            f"Training split has {len(train_data)} characters, but at least "
            f"{minimum} are required for sequence_length={cfg.sequence_length}. "
            "Use a larger corpus or reduce --sequence-length."
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_tag = uuid.uuid4().hex[:8]
    output_dir = (Path(args.output_dir) if args.output_dir else
                  Path("runs") / "v32_1" / f"fresh_{stamp}_seed{cfg.seed}_{run_tag}")
    output_dir.mkdir(parents=True, exist_ok=False)

    # From this point onward, mirror stdout and stderr into the run directory.
    log_path = output_dir / "terminal_output.txt"
    original_stdout, original_stderr = sys.stdout, sys.stderr
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    started_wall = time.time()
    stage_status = {}
    report = None
    error_text = None
    try:
        write_manifest(cfg, chars, train_data, val_data, corpus, output_dir, device)
        model = SACRSN(len(chars), cfg).to(device)

        print("=" * 72)
        print(f"SACRSN v{VERSION} — stability/diagnostic and output-integrity build")
        print("=" * 72)
        print(f"Ablation: {cfg.ablation}")
        print(f"Device: {device}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Vocabulary (incl. unknown): {len(chars)}")
        print(f"Training/validation characters: {len(train_data)}/{len(val_data)}")
        print(f"Optimiser updates: {cfg.steps}")
        print("Fresh initialisation: YES | checkpoint loading: DISABLED")
        print(f"Output: {output_dir.resolve()}")
        print(f"Terminal output is being saved to: {log_path.resolve()}")

        print("\n[STAGE 1/5] Training")
        report = train_model(model, train_data, val_data, cfg, device, output_dir)
        stage_status["training"] = {"status": "completed", "steps_completed": report["steps_completed"]}

        print("\n[STAGE 2/5] Model diagnostics")
        try:
            save_diagnostics(model, train_data, val_data, device, output_dir)
            stage_status["model_diagnostics"] = {"status": "completed"}
        except Exception as exc:
            stage_status["model_diagnostics"] = {"status": "failed", "error": repr(exc)}
            print(f"ERROR in model diagnostics: {exc!r}")
            traceback.print_exc()
            # Ensure predictable diagnostic files exist, with explicit failure state.
            write_json(output_dir / "diagnostics_error.json", {
                "stage": "model_diagnostics", "error": repr(exc),
                "traceback": traceback.format_exc()
            })

        print("\n[STAGE 3/5] Anomaly calibration")
        anomaly = None
        try:
            anomaly = anomaly_report(model, val_data, device, output_dir)
            stage_status["anomaly_calibration"] = {"status": "completed"}
        except Exception as exc:
            stage_status["anomaly_calibration"] = {"status": "failed", "error": repr(exc)}
            print(f"ERROR in anomaly calibration: {exc!r}")
            traceback.print_exc()
            write_json(output_dir / "anomaly_calibration.json", {
                "status": "failed", "error": repr(exc),
                "traceback": traceback.format_exc()
            })

        print("\n[STAGE 4/5] Text generation")
        try:
            generated = generate_text(model, train_text[:1], chars, mapping, 300, device)
            (output_dir / "generated_sample.txt").write_text(generated, encoding="utf-8")
            stage_status["text_generation"] = {"status": "completed"}
        except Exception as exc:
            stage_status["text_generation"] = {"status": "failed", "error": repr(exc)}
            print(f"ERROR in text generation: {exc!r}")
            traceback.print_exc()
            (output_dir / "generated_sample.txt").write_text(
                f"Generation failed: {exc!r}\n", encoding="utf-8")

        print("\n[STAGE 5/5] Summary")
        summary = {
            "version": VERSION,
            "ablation": cfg.ablation,
            "steps_completed": report["steps_completed"],
            "best_step": report["best_step"],
            "best_validation_nll": report["best_validation_nll"],
            "elapsed_seconds": report["elapsed_seconds"],
            "anomaly_calibration": anomaly,
            "stage_status": stage_status,
            "output_directory": str(output_dir.resolve()),
            "interpretation": "Diagnostic character-model experiment only; not evidence of reasoning.",
        }
        write_json(output_dir / "summary.json", summary)

        print("\n" + "=" * 72)
        print("RUN PIPELINE FINISHED")
        print("=" * 72)
        print(f"Best validation NLL: {report['best_validation_nll']:.6f}")
        print(f"Best step: {report['best_step']}")
        print(f"Completed updates: {report['steps_completed']}")
        print(f"Output directory: {output_dir.resolve()}")
        print("Interpret results cautiously: the default corpus is far too small")
        print("to support claims about general reasoning or robust generalisation.")

    except BaseException as exc:
        error_text = repr(exc)
        print(f"\nFATAL RUN ERROR: {exc!r}")
        traceback.print_exc()
        raise
    finally:
        # Persist a machine-readable end state even when training/diagnostics fail.
        finished = time.time()
        status = {
            "version": VERSION,
            "status": "failed" if error_text else (
                "completed_with_diagnostic_errors" if any(
                    v.get("status") == "failed" for v in stage_status.values()
                ) else "completed"
            ),
            "started_unix": started_wall,
            "finished_unix": finished,
            "elapsed_seconds_wall": finished - started_wall,
            "error": error_text,
            "stages": stage_status,
            "output_directory": str(output_dir.resolve()),
        }
        try:
            write_json(output_dir / "run_status.json", status)
            # Include all expected files, marking conditional or not-yet-created files.
            expected = [
                "config.json", "terminal_output.txt", "training_metrics.csv",
                "training_report.json", "best_model.pt", "final_model.pt",
                "symbol_transitions.json", "act_diagnostics.json",
                "stack_diagnostics.json", "stability_diagnostics.json",
                "anomaly_calibration.json",
                "generated_sample.txt", "summary.json", "run_status.json",
            ]
            if cfg.use_vq:
                expected.append("transition_matrix.csv")
            artifact_rows = []
            for name in expected:
                path = output_dir / name
                artifact_rows.append({
                    "file": name,
                    "exists": path.is_file(),
                    "size_bytes": path.stat().st_size if path.is_file() else None,
                    "required": True,
                })
            write_json(output_dir / "artifact_manifest.json", {
                "status": "complete" if all(x["exists"] for x in artifact_rows) else "incomplete",
                "expected_artifact_count": len(artifact_rows),
                "present_artifact_count": sum(x["exists"] for x in artifact_rows),
                "artifacts": artifact_rows,
                "note": "A present file may contain an explicit failure record if its stage failed; inspect run_status.json and terminal_output.txt.",
            })
        except Exception:
            traceback.print_exc()
        finally:
            try:
                print(f"\n[FINAL] Log saved: {log_path.resolve()}")
                print(f"[FINAL] Run status: {status['status']}")
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                sys.stdout, sys.stderr = original_stdout, original_stderr
                log_file.close()


if __name__ == "__main__":
    main()
