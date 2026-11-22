
"""
SACRSN v31.2 — Stability, Observability and Ablation Edition

Preserves the v31 complex-valued recurrent architecture:
- Complex-valued embeddings and recurrent processing
- Complex attention
- Differentiable bounded stack
- Vector quantisation and learned discrete symbols
- Adaptive computation time (ACT)
- Latent-symbol transition graph
- Character-level prediction decoder

v31.2 engineering changes:
- Stack boundary probability conservation
- Defensive pointer normalisation
- One graph transition per completed token
- Graph training and transition diagnostics use the same definition
- ACT halting mass and expected-depth diagnostics
- Loss-component instrumentation
- Pre-clipping and post-clipping gradient measurements
- Symbol utilisation and perplexity measurements
- Fresh initialisation by default
- Best-validation checkpoint preservation
- Independent experiment directories
- Configurable ablation modes
- Finite-value checks

IMPORTANT:
This is a research prototype, not a validated theorem prover.
ACT currently aggregates all recursion steps. It does not implement
computational early exit.
The transition graph is an auxiliary latent-symbol predictor.
It does not currently control decoder predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
import uuid

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_TEXT = """True, without falsehood, certain and most true.
That which is above is like to that which is below,
and that which is below is like to that which is above.
The father of all perfection in the whole world is here.
Its force or power is entire if it be converted into earth."""


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    embedding_dim: int = 64
    n_symbols: int = 64

    max_recursion_depth: int = 8
    act_epsilon: float = 0.01

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

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    batch_size: int = 16
    sequence_length: int = 32
    epochs: int = 1000

    validation_fraction: float = 0.20
    seed: int = 42
    log_every: int = 25

    codebook_temperature: float = 1.0
    eps: float = 1e-8

    early_stopping_patience: int = 100
    minimum_improvement: float = 1e-5

    ablation: str = "full"


ABLATIONS = {
    "control": {
        "use_vq": False,
        "use_graph": False,
        "use_diversity": False,
        "use_ponder": False,
        "use_stack": False,
    },
    "vq": {
        "use_vq": True,
        "use_graph": False,
        "use_diversity": False,
        "use_ponder": False,
        "use_stack": False,
    },
    "diversity": {
        "use_vq": True,
        "use_graph": False,
        "use_diversity": True,
        "use_ponder": False,
        "use_stack": False,
    },
    "graph": {
        "use_vq": True,
        "use_graph": True,
        "use_diversity": True,
        "use_ponder": False,
        "use_stack": False,
    },
    "full": {
        "use_vq": True,
        "use_graph": True,
        "use_diversity": True,
        "use_ponder": True,
        "use_stack": True,
    },
}


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def complex_cat(z: torch.Tensor) -> torch.Tensor:
    return torch.cat((z.real, z.imag), dim=-1)


def finite_or_raise(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(
            f"Non-finite values detected in {name}."
        )


# ============================================================
# Complex-valued layers
# ============================================================

class ComplexLayerNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, z):

        mag = torch.abs(z)

        mean = mag.mean(dim=-1, keepdim=True)

        var = mag.var(
            dim=-1,
            unbiased=False,
            keepdim=True
        )

        norm = (mag - mean) * torch.rsqrt(var + self.eps)

        norm = F.softplus(
            norm * self.weight + self.bias
        )

        phase = torch.angle(z)

        return torch.polar(norm, phase)


class ModReLU(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, z):

        mag = torch.abs(z)

        scale = F.relu(
            mag + self.bias
        ) / (mag + self.eps)

        return z * scale


class ComplexLinear(nn.Module):

    def __init__(self, dim: int):
        super().__init__()

        self.real = nn.Linear(
            dim, dim, bias=False
        )

        self.imag = nn.Linear(
            dim, dim, bias=False
        )

        nn.init.orthogonal_(self.real.weight)
        nn.init.orthogonal_(self.imag.weight)

    def forward(self, z):

        real = (
            self.real(z.real)
            - self.imag(z.imag)
        )

        imag = (
            self.real(z.imag)
            + self.imag(z.real)
        )

        return torch.complex(real, imag)


class ComplexAttention(nn.Module):

    def __init__(self, dim: int):
        super().__init__()

        self.q = ComplexLinear(dim)
        self.k = ComplexLinear(dim)
        self.v = ComplexLinear(dim)

        self.scale = (2 * dim) ** -0.5

    def forward(self, query, memory):

        q = complex_cat(
            self.q(query)
        ).unsqueeze(1)

        k = complex_cat(
            self.k(memory)
        )

        scores = torch.matmul(
            q, k.transpose(1, 2)
        ) * self.scale

        weights = F.softmax(scores, dim=-1)

        values = self.v(memory)

        out_r = torch.matmul(
            weights, values.real
        ).squeeze(1)

        out_i = torch.matmul(
            weights, values.imag
        ).squeeze(1)

        return torch.complex(out_r, out_i)


# ============================================================
# Differentiable bounded stack
# ============================================================

class DifferentiableStack(nn.Module):
    """
    Push moves probability towards higher indices.
    Pop moves probability towards lower indices.

    Boundary mass is retained.
    The stack is non-circular.
    """

    def __init__(self, dim: int, size: int):
        super().__init__()

        if size < 1:
            raise ValueError("Stack size must be >= 1.")

        self.dim = dim
        self.size = size

    def normalise_pointer(self, ptr):

        eps = 1e-8

        mass = ptr.sum(
            dim=-1,
            keepdim=True
        )

        normalised = ptr / mass.clamp_min(eps)

        fallback = torch.zeros_like(ptr)
        fallback[:, 0] = 1.0

        valid = (
            torch.isfinite(mass)
            & (mass > eps)
        )

        return torch.where(
            valid,
            normalised,
            fallback
        )

    def forward(self, z, memory, ptr, control):

        if ptr.shape[-1] != self.size:
            raise ValueError("Invalid pointer dimensions.")

        push, pop, noop = control.unbind(dim=-1)

        push = push.unsqueeze(-1)
        pop = pop.unsqueeze(-1)
        noop = noop.unsqueeze(-1)

        ptr = self.normalise_pointer(ptr)

        up = torch.zeros_like(ptr)
        down = torch.zeros_like(ptr)

        # Push movement.
        up[:, 1:] = ptr[:, :-1]

        # Upper boundary retention.
        up[:, -1] += ptr[:, -1]

        # Pop movement.
        down[:, :-1] = ptr[:, 1:]

        # Lower boundary retention.
        down[:, 0] += ptr[:, 0]

        candidate = (
            push * up
            + pop * down
            + noop * ptr
        )

        candidate = self.normalise_pointer(candidate)

        write_distribution = push * up

        z_flat = complex_cat(z).unsqueeze(1)

        new_memory = (
            write_distribution.unsqueeze(-1) * z_flat
            + (
                1.0 - write_distribution.unsqueeze(-1)
            ) * memory
        )

        read_flat = (
            new_memory
            * candidate.unsqueeze(-1)
        ).sum(dim=1)

        read = torch.complex(
            read_flat[..., :self.dim],
            read_flat[..., self.dim:]
        )

        return read, new_memory, candidate


# ============================================================
# Vector quantisation
# ============================================================

class SymbolQuantizer(nn.Module):

    def __init__(
        self,
        dim,
        n_symbols,
        temperature=1.0
    ):
        super().__init__()

        self.n_symbols = n_symbols
        self.temperature = temperature

        self.codebook = nn.Parameter(
            torch.empty(n_symbols, 2 * dim)
        )

        nn.init.uniform_(
            self.codebook, -0.5, 0.5
        )

    def forward(self, z):

        flat = complex_cat(z)

        distances = (
            flat.square().sum(-1, keepdim=True)
            + self.codebook.square().sum(-1).unsqueeze(0)
            - 2.0 * flat @ self.codebook.t()
        ).clamp_min(0.0)

        soft = F.softmax(
            -distances / max(
                self.temperature, 1e-4
            ),
            dim=-1
        )

        indices = distances.argmin(dim=-1)

        quantized = F.embedding(
            indices, self.codebook
        )

        codebook_loss = F.mse_loss(
            quantized,
            flat.detach()
        )

        commitment_loss = F.mse_loss(
            flat,
            quantized.detach()
        )

        st = flat + (
            quantized - flat
        ).detach()

        d = z.shape[-1]

        z_q = torch.complex(
            st[..., :d],
            st[..., d:]
        )

        return (
            z_q,
            indices,
            soft,
            codebook_loss,
            commitment_loss
        )


# ============================================================
# Latent transition graph
# ============================================================

class TransitionGraph(nn.Module):

    def __init__(self, n_symbols):
        super().__init__()

        self.logits = nn.Parameter(
            torch.zeros(n_symbols, n_symbols)
        )

    def loss(self, previous, current):

        if previous is None:
            return self.logits.sum() * 0.0

        return F.cross_entropy(
            self.logits[previous.reshape(-1)],
            current.reshape(-1)
        )

    def probabilities(self):

        return F.softmax(
            self.logits,
            dim=-1
        )


# ============================================================
# Recursive processing cell
# ============================================================

class AdaptiveRecursiveCell(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.linear = ComplexLinear(dim)
        self.norm = ComplexLayerNorm(dim)
        self.activation = ModReLU(dim)

        self.attention = ComplexAttention(dim)

        self.halt = nn.Linear(
            2 * dim, 1
        )

        self.stack_control = nn.Linear(
            2 * dim, 3
        )

        nn.init.constant_(
            self.halt.bias, -1.5
        )

    def forward(self, z, history):

        processed = self.activation(
            self.norm(
                self.linear(z)
            )
        )

        if history is not None and history.size(1) > 0:

            attended = self.attention(
                processed, history
            )

            processed = processed + attended

        flat = complex_cat(processed)

        halt_probability = torch.sigmoid(
            self.halt(flat)
        )

        stack_control = F.softmax(
            self.stack_control(flat),
            dim=-1
        )

        return (
            processed,
            halt_probability,
            stack_control
        )


# ============================================================
# Model state
# ============================================================

@dataclass
class ModelState:
    recurrent: torch.Tensor
    stack_memory: torch.Tensor
    stack_pointer: torch.Tensor
    previous_symbol: Optional[torch.Tensor]


# ============================================================
# SACRSN v31.2 model
# ============================================================

class SACRSNv31(nn.Module):

    def __init__(self, vocab_size, config):

        super().__init__()

        self.config = config

        d = config.embedding_dim

        self.embedding_magnitude = nn.Embedding(
            vocab_size, d
        )

        self.embedding_phase = nn.Embedding(
            vocab_size, d
        )

        self.cell = AdaptiveRecursiveCell(d)

        self.quantizer = SymbolQuantizer(
            d,
            config.n_symbols,
            config.codebook_temperature
        )

        self.graph = TransitionGraph(
            config.n_symbols
        )

        self.decoder = nn.Linear(
            2 * d, vocab_size
        )

        self.stack = (
            DifferentiableStack(
                d, config.stack_size
            )
            if config.use_stack
            else None
        )

    def initial_state(self, batch_size, device):

        d = self.config.embedding_dim

        recurrent = torch.zeros(
            batch_size,
            d,
            dtype=torch.complex64,
            device=device
        )

        stack_memory = torch.zeros(
            batch_size,
            self.config.stack_size,
            2 * d,
            device=device
        )

        pointer = torch.zeros(
            batch_size,
            self.config.stack_size,
            device=device
        )

        pointer[:, 0] = 1.0

        return ModelState(
            recurrent,
            stack_memory,
            pointer,
            None
        )

    def embed(self, token_ids):

        magnitude = F.softplus(
            self.embedding_magnitude(token_ids)
        )

        phase = math.pi * torch.tanh(
            self.embedding_phase(token_ids)
        )

        return torch.polar(
            magnitude, phase
        )

    def forward_step(self, token_ids, state=None):

        batch = token_ids.shape[0]

        if state is None:
            state = self.initial_state(
                batch, token_ids.device
            )

        z = (
            0.5 * self.embed(token_ids)
            + 0.5 * state.recurrent
        )

        stack_memory = state.stack_memory
        stack_pointer = state.stack_pointer

        recursive_history = []

        weighted_state = torch.zeros_like(z)

        remaining = torch.ones(
            batch, 1, device=z.device
        )

        expected_steps = torch.zeros(
            batch, 1, device=z.device
        )

        halt_masses = []
        vq_losses = []
        soft_assignments = []
        stack_depths = []

        symbol_indices = None

        for step in range(
            self.config.max_recursion_depth
        ):

            history_tensor = (
                torch.stack(
                    recursive_history, dim=1
                )
                if recursive_history
                else None
            )

            z_proc, halt_prob, stack_control = self.cell(
                z, history_tensor
            )

            if self.stack is not None:

                stack_read, stack_memory, stack_pointer = self.stack(
                    z_proc,
                    stack_memory,
                    stack_pointer,
                    stack_control
                )

                z_combined = z_proc + stack_read

                positions = torch.arange(
                    self.config.stack_size,
                    device=z.device,
                    dtype=z.real.dtype
                )

                depth = (
                    stack_pointer * positions
                ).sum(-1)

                stack_depths.append(depth)

            else:
                z_combined = z_proc

            (
                z_symbol,
                symbol_indices,
                soft,
                codebook_loss,
                commitment_loss
            ) = self.quantizer(z_combined)

            if self.config.use_vq:

                vq_loss = (
                    codebook_loss
                    + self.config.commitment_cost
                    * commitment_loss
                )

            else:
                vq_loss = z_combined.real.sum() * 0.0

            vq_losses.append(vq_loss)
            soft_assignments.append(soft)

            # Differentiable ACT remainder.
            if step == self.config.max_recursion_depth - 1:
                halt_mass = remaining
            else:
                halt_mass = remaining * halt_prob

            halt_masses.append(halt_mass)

            weighted_state = (
                weighted_state
                + halt_mass * z_combined
            )

            expected_steps = (
                expected_steps + remaining
            )

            remaining = (
                remaining - halt_mass
            ).clamp_min(0.0)

            recursive_history.append(z_proc)

            z = (
                0.7 * z_combined
                + 0.3 * z_symbol
            )

        # Exactly one graph transition per completed token.
        # The graph target matches transition diagnostics.
        if self.config.use_graph:

            graph_loss = self.graph.loss(
                state.previous_symbol,
                symbol_indices
            )

        else:
            graph_loss = self.graph.logits.sum() * 0.0

        logits = self.decoder(
            complex_cat(weighted_state)
        )

        total_halt_mass = torch.stack(
            halt_masses, dim=0
        ).sum(dim=0)

        halt_distribution = torch.stack(
            halt_masses, dim=0
        )

        next_state = ModelState(
            weighted_state,
            stack_memory,
            stack_pointer,
            symbol_indices
        )

        metrics = {

            "ponder": expected_steps.mean(),

            "vq_loss": torch.stack(
                vq_losses
            ).mean(),

            "graph_loss": graph_loss,

            "soft_assignments": torch.stack(
                soft_assignments,
                dim=0
            ),

            "stack_depth": (
                torch.stack(
                    stack_depths
                ).mean()
                if stack_depths
                else logits.new_tensor(0.0)
            ),

            "symbol_indices": symbol_indices,

            "halt_mass": total_halt_mass,

            "halt_distribution": halt_distribution,

            "expected_steps": expected_steps,

            "stack_pointer_mass": (
                stack_pointer.sum(dim=-1).mean()
            ),

        }

        return logits, next_state, metrics


# ============================================================
# Data preparation
# ============================================================

def split_text(text, val_fraction):

    chars = sorted(set(text))

    if len(chars) < 2:
        raise ValueError(
            "Training text needs at least two characters."
        )

    mapping = {
        c: i for i, c in enumerate(chars)
    }

    ids = torch.tensor(
        [mapping[c] for c in text],
        dtype=torch.long
    )

    split = max(
        2,
        min(
            len(ids) - 1,
            int(len(ids) * (1.0 - val_fraction))
        )
    )

    return (
        chars,
        mapping,
        ids[:split],
        ids[split:]
    )


def sample_batch(
    data,
    batch_size,
    sequence_length,
    device
):

    if len(data) <= sequence_length + 1:
        raise ValueError(
            "Training data too short for sequence length."
        )

    starts = torch.randint(
        0,
        len(data) - sequence_length - 1,
        (batch_size,)
    )

    x = torch.stack([
        data[s:s + sequence_length]
        for s in starts
    ]).to(device)

    y = torch.stack([
        data[s + 1:s + sequence_length + 1]
        for s in starts
    ]).to(device)

    return x, y


# ============================================================
# Sequence objective
# ============================================================

def sequence_objective(model, x, y, cfg):

    state = None

    prediction_losses = []
    vq_losses = []
    graph_losses = []
    ponder_losses = []

    assignments = []

    halt_mass_errors = []
    stack_mass_errors = []

    for t in range(x.size(1)):

        logits, state, metrics = model.forward_step(
            x[:, t], state
        )

        prediction_losses.append(
            F.cross_entropy(
                logits, y[:, t]
            )
        )

        vq_losses.append(
            metrics["vq_loss"]
        )

        graph_losses.append(
            metrics["graph_loss"]
        )

        ponder_losses.append(
            metrics["ponder"]
        )

        assignments.append(
            metrics["soft_assignments"].mean(dim=0)
        )

        halt_mass_errors.append(
            (
                metrics["halt_mass"] - 1.0
            ).abs().mean()
        )

        stack_mass_errors.append(
            (
                state.stack_pointer.sum(dim=-1) - 1.0
            ).abs().mean()
        )

    prediction = torch.stack(
        prediction_losses
    ).mean()

    vq = torch.stack(
        vq_losses
    ).mean()

    graph = torch.stack(
        graph_losses
    ).mean()

    ponder = torch.stack(
        ponder_losses
    ).mean()

    assignments = torch.cat(
        assignments, dim=0
    )

    usage = assignments.mean(dim=0)

    usage = usage / usage.sum().clamp_min(
        cfg.eps
    )

    entropy = -(
        usage * torch.log(
            usage.clamp_min(cfg.eps)
        )
    ).sum()

    if cfg.use_diversity:
        diversity = -entropy
    else:
        diversity = entropy * 0.0

    total = prediction

    if cfg.use_vq:
        total = total + cfg.vq_weight * vq

    if cfg.use_graph:
        total = total + cfg.graph_weight * graph

    if cfg.use_ponder:
        total = total + cfg.ponder_weight * ponder

    if cfg.use_diversity:
        total = total + cfg.diversity_weight * diversity

    perplexity = torch.exp(entropy)

    active_symbols = (
        usage > (1.0 / cfg.n_symbols) * 0.25
    ).sum()

    return total, {

        "prediction": prediction,
        "vq": vq,
        "graph": graph,
        "ponder": ponder,
        "diversity": diversity,
        "entropy": entropy,
        "usage_perplexity": perplexity,
        "active_symbols": active_symbols,
        "halt_mass_error": torch.stack(
            halt_mass_errors
        ).mean(),
        "stack_mass_error": torch.stack(
            stack_mass_errors
        ).mean(),

    }


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, data, device):

    model.eval()

    if len(data) < 2:
        return float("nan")

    state = None
    losses = []

    for i in range(len(data) - 1):

        logits, state, _ = model.forward_step(
            data[i:i + 1].to(device),
            state
        )

        loss = F.cross_entropy(
            logits,
            data[i + 1:i + 2].to(device)
        )

        losses.append(loss.item())

    return float(np.mean(losses))


# ============================================================
# Gradient diagnostics
# ============================================================

def gradient_norm(model):

    total = 0.0

    for parameter in model.parameters():

        if parameter.grad is None:
            continue

        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(
                "Non-finite gradient detected."
            )

        norm = parameter.grad.detach().norm(2).item()

        total += norm ** 2

    return math.sqrt(total)


# ============================================================
# CSV logging
# ============================================================

def append_csv(path, row):

    exists = path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys())
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# Training
# ============================================================

def train_model(
    model,
    train_data,
    val_data,
    cfg,
    device,
    output_dir
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.epochs, 1),
        eta_min=cfg.learning_rate * 0.05
    )

    best_val = float("inf")
    best_state = None
    best_epoch = 0

    patience_counter = 0

    history = []

    start_time = time.time()

    print(
        f"Device: {device} | "
        f"Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    print(
        "Fresh initialisation: no checkpoint is loaded."
    )

    for epoch in range(1, cfg.epochs + 1):

        model.train()

        x, y = sample_batch(
            train_data,
            cfg.batch_size,
            cfg.sequence_length,
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss, metrics = sequence_objective(
            model, x, y, cfg
        )

        finite_or_raise(
            "training loss", loss
        )

        loss.backward()

        pre_clip_norm = gradient_norm(model)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.grad_clip
        )

        post_clip_norm = gradient_norm(model)

        clipping_ratio = (
            post_clip_norm / max(
                pre_clip_norm, 1e-12
            )
        )

        optimizer.step()
        scheduler.step()

        row = {

            "epoch": epoch,

            "total_loss": float(loss.detach()),

            "prediction_loss": float(
                metrics["prediction"].detach()
            ),

            "vq_loss": float(
                metrics["vq"].detach()
            ),

            "graph_loss": float(
                metrics["graph"].detach()
            ),

            "ponder": float(
                metrics["ponder"].detach()
            ),

            "diversity_loss": float(
                metrics["diversity"].detach()
            ),

            "symbol_entropy": float(
                metrics["entropy"].detach()
            ),

            "usage_perplexity": float(
                metrics["usage_perplexity"].detach()
            ),

            "active_symbols": int(
                metrics["active_symbols"].detach()
            ),

            "pre_clip_grad_norm": pre_clip_norm,

            "post_clip_grad_norm": post_clip_norm,

            "clipping_ratio": clipping_ratio,

            "halt_mass_error": float(
                metrics["halt_mass_error"].detach()
            ),

            "stack_mass_error": float(
                metrics["stack_mass_error"].detach()
            ),

            "learning_rate": optimizer.param_groups[0]["lr"],

        }

        validation_due = (
            epoch == 1
            or epoch % cfg.log_every == 0
            or epoch == cfg.epochs
        )

        if validation_due:

            val_nll = evaluate(
                model,
                val_data,
                device
            )

            row["validation_nll"] = val_nll

            print(
                f"Epoch {epoch:04d}/{cfg.epochs} "
                f"| total={row['total_loss']:.4f} "
                f"| pred={row['prediction_loss']:.4f} "
                f"| val={val_nll:.4f} "
                f"| ppx={row['usage_perplexity']:.2f} "
                f"| active={row['active_symbols']} "
                f"| grad={pre_clip_norm:.3f}"
            )

            if (
                math.isfinite(val_nll)
                and val_nll < best_val - cfg.minimum_improvement
            ):

                best_val = val_nll
                best_epoch = epoch
                patience_counter = 0

                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

                torch.save(
                    {
                        "model_state_dict": best_state,
                        "config": asdict(cfg),
                        "best_validation_nll": best_val,
                        "best_epoch": best_epoch,
                    },
                    output_dir / "best_model.pt"
                )

            else:
                patience_counter += 1

            if (
                patience_counter >= cfg.early_stopping_patience
            ):
                print(
                    "Early stopping: validation has not improved."
                )
                history.append(row)
                append_csv(
                    output_dir / "training_metrics.csv",
                    row
                )
                break

        history.append(row)

        append_csv(
            output_dir / "training_metrics.csv",
            row
        )

    # Save the final state separately from the best state.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "epoch": epoch,
        },
        output_dir / "final_model.pt"
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - start_time

    report = {

        "version": "31.2",

        "config": asdict(cfg),

        "epochs_completed": epoch,

        "best_epoch": best_epoch,

        "best_validation_nll": best_val,

        "elapsed_seconds": elapsed,

        "parameter_count": sum(
            p.numel() for p in model.parameters()
        ),

        "history": history,

        "interpretation": (
            "Character-level sequence prediction and "
            "latent-symbol transition learning. "
            "Does not establish general reasoning."
        ),

    }

    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    return report


# ============================================================
# Symbol transition diagnostics
# ============================================================

@torch.no_grad()
def collect_symbol_transitions(
    model,
    data,
    device
):

    model.eval()

    state = None

    transitions = Counter()
    symbol_counts = Counter()

    previous = None

    for token in data:

        _, state, metrics = model.forward_step(
            token.view(1).to(device),
            state
        )

        current = int(
            metrics["symbol_indices"].item()
        )

        symbol_counts[current] += 1

        if previous is not None:

            transitions[
                (previous, current)
            ] += 1

        previous = current

    return symbol_counts, transitions


# ============================================================
# ACT diagnostics
# ============================================================

@torch.no_grad()
def collect_act_diagnostics(
    model,
    data,
    device
):

    model.eval()

    state = None

    depth_values = []
    mass_errors = []

    for token in data:

        _, state, metrics = model.forward_step(
            token.view(1).to(device),
            state
        )

        depth_values.append(
            float(metrics["ponder"])
        )

        mass_errors.append(
            float(
                (
                    metrics["halt_mass"] - 1.0
                ).abs().mean()
            )
        )

    return {

        "mean_expected_depth": float(
            np.mean(depth_values)
        ),

        "median_expected_depth": float(
            np.median(depth_values)
        ),

        "minimum_expected_depth": float(
            np.min(depth_values)
        ),

        "maximum_expected_depth": float(
            np.max(depth_values)
        ),

        "maximum_mass_error": float(
            np.max(mass_errors)
        ),

        "computational_early_exit": False,

    }


# ============================================================
# Anomaly calibration
# ============================================================

@torch.no_grad()
def calibrated_anomaly_report(
    model,
    val_data,
    device,
    output_dir
):

    model.eval()

    state = None
    scores = []

    for i in range(len(val_data) - 1):

        logits, state, _ = model.forward_step(
            val_data[i:i + 1].to(device),
            state
        )

        target = val_data[
            i + 1:i + 2
        ].to(device)

        scores.append(
            float(
                F.cross_entropy(
                    logits, target
                )
            )
        )

    if not scores:
        return {}

    report = {

        "method": "next-character negative log likelihood",

        "score_count": len(scores),

        "mean": float(np.mean(scores)),

        "standard_deviation": float(
            np.std(scores)
        ),

        "p95_threshold": float(
            np.quantile(scores, 0.95)
        ),

        "maximum": float(np.max(scores)),

        "interpretation": (
            "Scores above the threshold are unusual relative "
            "to this validation split. This is not semantic "
            "or ethical anomaly detection."
        ),

    }

    (output_dir / "anomaly_calibration.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    return report


# ============================================================
# Diagnostics export
# ============================================================

def save_diagnostics(
    model,
    train_data,
    val_data,
    device,
    output_dir
):

    counts, transitions = collect_symbol_transitions(
        model,
        train_data,
        device
    )

    transition_rows = [

        {
            "from_symbol": a,
            "to_symbol": b,
            "count": count
        }

        for (a, b), count in transitions.most_common()

    ]

    symbol_report = {

        "symbol_counts": dict(counts),

        "transitions": transition_rows,

        "warning": (
            "Empirical latent transitions, "
            "not extracted logical rules."
        ),

    }

    (output_dir / "symbol_transitions.json").write_text(
        json.dumps(symbol_report, indent=2),
        encoding="utf-8"
    )

    graph_probabilities = (
        model.graph.probabilities()
        .detach()
        .cpu()
        .numpy()
    )

    np.savetxt(
        output_dir / "transition_matrix.csv",
        graph_probabilities,
        delimiter=",",
        fmt="%.7f"
    )

    act_report = collect_act_diagnostics(
        model,
        val_data,
        device
    )

    (output_dir / "act_diagnostics.json").write_text(
        json.dumps(act_report, indent=2),
        encoding="utf-8"
    )

    pointer_report = {

        "stack_enabled": model.stack is not None,

        "stack_size": model.config.stack_size,

        "pointer_integrity": (
            "Measured during training objective."
        ),

    }

    (output_dir / "stack_diagnostics.json").write_text(
        json.dumps(pointer_report, indent=2),
        encoding="utf-8"
    )


# ============================================================
# Text generation
# ============================================================

@torch.no_grad()
def generate_text(
    model,
    start_text,
    chars,
    mapping,
    max_new_tokens,
    device,
    temperature=0.8
):

    model.eval()

    if not start_text:
        start_text = chars[0]

    ids = [
        mapping.get(c, 0)
        for c in start_text
    ]

    state = None

    for token in ids:

        _, state, _ = model.forward_step(
            torch.tensor(
                [token],
                device=device
            ),
            state
        )

    current = torch.tensor(
        [ids[-1]],
        device=device
    )

    output = list(start_text)

    for _ in range(max_new_tokens):

        logits, state, _ = model.forward_step(
            current, state
        )

        probabilities = F.softmax(
            logits / max(temperature, 1e-4),
            dim=-1
        )

        current = torch.multinomial(
            probabilities, 1
        ).squeeze(-1)

        output.append(
            chars[int(current.item())]
        )

    return "".join(output)


# ============================================================
# Experiment metadata
# ============================================================

def write_experiment_manifest(
    cfg,
    chars,
    train_data,
    val_data,
    output_dir,
    device
):

    manifest = {

        "version": "31.2",

        "created": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "run_id": str(uuid.uuid4()),

        "seed": cfg.seed,

        "device": str(device),

        "config": asdict(cfg),

        "vocabulary_size": len(chars),

        "training_characters": len(train_data),

        "validation_characters": len(val_data),

        "split_method": (
            "Contiguous character-level split."
        ),

        "fresh_initialisation": True,

        "checkpoint_loaded": False,

        "source": "sacrsn_v31_2.py",

    }

    (output_dir / "config.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="SACRSN v31.2"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None
    )

    parser.add_argument(
        "--text-file",
        type=str,
        default=None
    )

    parser.add_argument(
        "--ablation",
        choices=list(ABLATIONS.keys()),
        default="full"
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None
    )

    args = parser.parse_args()

    cfg = Config(seed=args.seed)

    if args.epochs is not None:
        cfg.epochs = args.epochs

    if args.sequence_length is not None:
        cfg.sequence_length = args.sequence_length

    cfg.ablation = args.ablation

    for key, value in ABLATIONS[args.ablation].items():
        setattr(cfg, key, value)

    if cfg.epochs < 1:
        raise ValueError("Epoch count must be positive.")

    if cfg.max_recursion_depth < 1:
        raise ValueError("Recursion depth must be positive.")

    set_seed(cfg.seed)

    if args.text_file:

        text = Path(
            args.text_file
        ).read_text(encoding="utf-8")

    else:
        text = DEFAULT_TEXT

    chars, mapping, train_data, val_data = split_text(
        text,
        cfg.validation_fraction
    )

    if args.device == "cuda" and not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    device = torch.device(

        "cuda"
        if (
            args.device == "cuda"
            or (
                args.device == "auto"
                and torch.cuda.is_available()
            )
        )
        else "cpu"

    )

    stamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    run_id = uuid.uuid4().hex[:8]

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("runs")
        / "v31_2"
        / f"fresh_{stamp}_seed{cfg.seed}_{run_id}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    write_experiment_manifest(
        cfg,
        chars,
        train_data,
        val_data,
        output_dir,
        device
    )

    model = SACRSNv31(
        len(chars),
        cfg
    ).to(device)

    print("=" * 60)
    print("SACRSN v31.2")
    print("=" * 60)

    print(f"Ablation: {cfg.ablation}")
    print(f"Device: {device}")
    print(f"Vocabulary: {len(chars)}")
    print(f"Training characters: {len(train_data)}")
    print(f"Validation characters: {len(val_data)}")
    print(f"Seed: {cfg.seed}")
    print("Fresh initialisation: YES")
    print("Checkpoint loading: DISABLED")

    report = train_model(
        model,
        train_data,
        val_data,
        cfg,
        device,
        output_dir
    )

    save_diagnostics(
        model,
        train_data,
        val_data,
        device,
        output_dir
    )

    anomaly = calibrated_anomaly_report(
        model,
        val_data,
        device,
        output_dir
    )

    generated = generate_text(
        model,
        chars[0],
        chars,
        mapping,
        300,
        device
    )

    (output_dir / "generated_sample.txt").write_text(
        generated,
        encoding="utf-8"
    )

    summary = {

        "version": "31.2",

        "ablation": cfg.ablation,

        "best_epoch": report["best_epoch"],

        "best_validation_nll": (
            report["best_validation_nll"]
        ),

        "elapsed_seconds": (
            report["elapsed_seconds"]
        ),

        "anomaly_calibration": anomaly,

        "output_directory": str(
            output_dir.resolve()
        ),

    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best validation NLL: "
        f"{report['best_validation_nll']:.5f}"
    )

    print(
        f"Best epoch: {report['best_epoch']}"
    )

    print(
        f"Output: {output_dir.resolve()}"
    )

    print()
    print(
        "This run measures character prediction and "
        "latent transition learning."
    )

    print(
        "It does not establish general-purpose reasoning."
    )


if __name__ == "__main__":
    main()
