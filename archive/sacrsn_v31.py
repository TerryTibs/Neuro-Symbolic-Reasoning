"""
SACRSN v31.1 — Repaired Observable Edition
A repaired research prototype preserving the original v31 design themes.

Important scope:
- This is a character-level sequence model with learned discrete symbols and
  transition statistics. It is NOT, by itself, a theorem prover or validated
  general-purpose neuro-symbolic reasoner.
- The transition graph models observed latent-symbol transitions. It is not
  normative ethics or formal logic.
- Anomaly scores are calibrated against a held-out validation split only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
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


@dataclass
class Config:
    embedding_dim: int = 64
    n_symbols: int = 64
    max_recursion_depth: int = 8
    act_epsilon: float = 0.01
    stack_size: int = 16
    use_stack: bool = True
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


class ComplexLayerNorm(nn.Module):
    """Normalises magnitude while preserving phase; stable at near-zero inputs."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(z)
        mean = mag.mean(dim=-1, keepdim=True)
        var = mag.var(dim=-1, unbiased=False, keepdim=True)
        norm = (mag - mean) * torch.rsqrt(var + self.eps)
        norm = F.softplus(norm * self.weight + self.bias)
        phase = torch.angle(z)
        return torch.polar(norm, phase)


class ModReLU(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(z)
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
        return torch.complex(
            self.real(z.real) - self.imag(z.imag),
            self.real(z.imag) + self.imag(z.real),
        )


class ComplexAttention(nn.Module):
    """Attention over the current token's recursive state history."""
    def __init__(self, dim: int):
        super().__init__()
        self.q = ComplexLinear(dim)
        self.k = ComplexLinear(dim)
        self.v = ComplexLinear(dim)
        self.scale = (2 * dim) ** -0.5

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # query [B,D], memory [B,T,D]
        q = complex_cat(self.q(query)).unsqueeze(1)
        k = complex_cat(self.k(memory))
        scores = torch.matmul(q, k.transpose(1, 2)) * self.scale
        weights = F.softmax(scores, dim=-1)
        values = self.v(memory)
        out_r = torch.matmul(weights, values.real).squeeze(1)
        out_i = torch.matmul(weights, values.imag).squeeze(1)
        return torch.complex(out_r, out_i)


class DifferentiableStack(nn.Module):
    """
    Bounded differentiable stack.
    Pointer movement is non-circular: push at the top/bottom boundary is
    clamped rather than wrapping around to the opposite end.
    """
    def __init__(self, dim: int, size: int):
        super().__init__()
        self.dim = dim
        self.size = size

    def forward(self, z, memory, ptr, control):
        # control is [B,3] = push, pop, no-op probabilities
        push, pop, noop = control.unbind(dim=-1)
        push, pop, noop = push[:, None], pop[:, None], noop[:, None]

        up = torch.zeros_like(ptr)
        down = torch.zeros_like(ptr)
        up[:, 1:] = ptr[:, :-1]
        down[:, :-1] = ptr[:, 1:]

        candidate = push * up + pop * down + noop * ptr
        candidate = candidate / candidate.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # Write at the destination pointer after a push.
        write_mask = (push * up).unsqueeze(-1)
        z_flat = complex_cat(z).unsqueeze(1)
        new_memory = write_mask * z_flat + (1.0 - write_mask) * memory
        read_flat = (new_memory * candidate.unsqueeze(-1)).sum(dim=1)
        read = torch.complex(read_flat[..., :self.dim], read_flat[..., self.dim:])
        return read, new_memory, candidate


class SymbolQuantizer(nn.Module):
    """Straight-through VQ with differentiable batch-level usage statistics."""
    def __init__(self, dim: int, n_symbols: int, temperature: float = 1.0):
        super().__init__()
        self.n_symbols = n_symbols
        self.temperature = temperature
        self.codebook = nn.Parameter(torch.empty(n_symbols, 2 * dim))
        nn.init.uniform_(self.codebook, -0.5, 0.5)

    def forward(self, z):
        flat = complex_cat(z)
        distances = (
            flat.square().sum(-1, keepdim=True)
            + self.codebook.square().sum(-1).unsqueeze(0)
            - 2.0 * flat @ self.codebook.t()
        ).clamp_min(0.0)
        soft = F.softmax(-distances / max(self.temperature, 1e-4), dim=-1)
        indices = distances.argmin(dim=-1)
        quantized = F.embedding(indices, self.codebook)

        codebook_loss = F.mse_loss(quantized, flat.detach())
        commitment_loss = F.mse_loss(flat, quantized.detach())
        vq_loss = codebook_loss + commitment_loss

        # Straight-through estimator: forward uses quantized values;
        # gradients to encoder flow as if identity.
        st = flat + (quantized - flat).detach()
        d = z.shape[-1]
        z_q = torch.complex(st[..., :d], st[..., d:])
        return z_q, indices, soft, codebook_loss, commitment_loss


class TransitionGraph(nn.Module):
    """Directed latent-symbol transition model; not a formal logic system."""
    def __init__(self, n_symbols: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_symbols, n_symbols))

    def loss(self, previous: Optional[torch.Tensor], current: torch.Tensor):
        if previous is None:
            return self.logits.sum() * 0.0
        return F.cross_entropy(self.logits[previous.reshape(-1)], current.reshape(-1))

    def probabilities(self):
        return F.softmax(self.logits, dim=-1)


class AdaptiveRecursiveCell(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = ComplexLinear(dim)
        self.norm = ComplexLayerNorm(dim)
        self.activation = ModReLU(dim)
        self.attention = ComplexAttention(dim)
        self.halt = nn.Linear(2 * dim, 1)
        self.stack_control = nn.Linear(2 * dim, 3)
        nn.init.constant_(self.halt.bias, -1.5)

    def forward(self, z, history):
        processed = self.activation(self.norm(self.linear(z)))
        if history is not None and history.size(1) > 0:
            attended = self.attention(processed, history)
            processed = processed + attended
        flat = complex_cat(processed)
        halt_probability = torch.sigmoid(self.halt(flat))
        stack_control = F.softmax(self.stack_control(flat), dim=-1)
        return processed, halt_probability, stack_control


@dataclass
class ModelState:
    recurrent: torch.Tensor
    stack_memory: torch.Tensor
    stack_pointer: torch.Tensor
    previous_symbol: Optional[torch.Tensor]


class SACRSNv31(nn.Module):
    def __init__(self, vocab_size: int, config: Config):
        super().__init__()
        self.config = config
        d = config.embedding_dim
        self.embedding_magnitude = nn.Embedding(vocab_size, d)
        self.embedding_phase = nn.Embedding(vocab_size, d)
        self.cell = AdaptiveRecursiveCell(d)
        self.quantizer = SymbolQuantizer(d, config.n_symbols, config.codebook_temperature)
        self.graph = TransitionGraph(config.n_symbols)
        self.decoder = nn.Linear(2 * d, vocab_size)
        self.stack = DifferentiableStack(d, config.stack_size) if config.use_stack else None

    def initial_state(self, batch_size: int, device) -> ModelState:
        recurrent = torch.zeros(batch_size, self.config.embedding_dim, dtype=torch.complex64, device=device)
        stack_memory = torch.zeros(batch_size, self.config.stack_size,
                                   2 * self.config.embedding_dim, device=device)
        pointer = torch.zeros(batch_size, self.config.stack_size, device=device)
        pointer[:, 0] = 1.0
        return ModelState(recurrent, stack_memory, pointer, None)

    def embed(self, token_ids):
        magnitude = F.softplus(self.embedding_magnitude(token_ids))
        phase = math.pi * torch.tanh(self.embedding_phase(token_ids))
        return torch.polar(magnitude, phase)

    def forward_step(self, token_ids, state: Optional[ModelState] = None):
        batch = token_ids.shape[0]
        if state is None:
            state = self.initial_state(batch, token_ids.device)

        z = 0.5 * self.embed(token_ids) + 0.5 * state.recurrent
        stack_memory, stack_pointer = state.stack_memory, state.stack_pointer
        recursive_history = []
        weighted_state = torch.zeros_like(z)
        remaining = torch.ones(batch, 1, device=z.device)
        expected_steps = torch.zeros(batch, 1, device=z.device)
        symbol_indices = None
        vq_losses, graph_losses, soft_assignments = [], [], []
        stack_depths = []

        for step in range(self.config.max_recursion_depth):
            history_tensor = torch.stack(recursive_history, dim=1) if recursive_history else None
            z_proc, halt_prob, stack_control = self.cell(z, history_tensor)

            if self.stack is not None:
                stack_read, stack_memory, stack_pointer = self.stack(
                    z_proc, stack_memory, stack_pointer, stack_control
                )
                z_combined = z_proc + stack_read
                depth = (stack_pointer * torch.arange(
                    self.config.stack_size, device=z.device, dtype=z.real.dtype
                )).sum(-1)
                stack_depths.append(depth)
            else:
                z_combined = z_proc

            z_symbol, symbol_indices, soft, codebook_loss, commitment_loss = self.quantizer(z_combined)
            graph_loss = self.graph.loss(state.previous_symbol, symbol_indices)
            vq_losses.append(codebook_loss + self.config.commitment_cost * commitment_loss)
            graph_losses.append(graph_loss)
            soft_assignments.append(soft)

            # Use a differentiable ACT remainder formulation. No thresholded
            # boolean mask disconnects halt probabilities from the loss.
            if step == self.config.max_recursion_depth - 1:
                halt_mass = remaining
            else:
                halt_mass = remaining * halt_prob
            weighted_state = weighted_state + halt_mass * z_combined
            expected_steps = expected_steps + remaining
            remaining = (remaining - halt_mass).clamp_min(0.0)

            recursive_history.append(z_proc)
            z = 0.7 * z_combined + 0.3 * z_symbol

        logits = self.decoder(complex_cat(weighted_state))
        next_state = ModelState(weighted_state, stack_memory, stack_pointer, symbol_indices)
        metrics = {
            "ponder": expected_steps.mean(),
            "vq_loss": torch.stack(vq_losses).mean(),
            "graph_loss": torch.stack(graph_losses).mean(),
            "soft_assignments": torch.stack(soft_assignments, dim=0),
            "stack_depth": torch.stack(stack_depths).mean() if stack_depths else logits.new_tensor(0.0),
            "symbol_indices": symbol_indices,
        }
        return logits, next_state, metrics


def split_text(text: str, val_fraction: float):
    chars = sorted(set(text))
    if len(chars) < 2:
        raise ValueError("Training text must contain at least two distinct characters.")
    mapping = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([mapping[c] for c in text], dtype=torch.long)
    split = max(2, min(len(ids) - 1, int(len(ids) * (1.0 - val_fraction))))
    return chars, mapping, ids[:split], ids[split:]


def sample_batch(data, batch_size, sequence_length, device):
    if len(data) <= sequence_length + 1:
        raise ValueError("Training split is too short for the configured sequence_length.")
    starts = torch.randint(0, len(data) - sequence_length - 1, (batch_size,))
    x = torch.stack([data[s:s + sequence_length] for s in starts]).to(device)
    y = torch.stack([data[s + 1:s + sequence_length + 1] for s in starts]).to(device)
    return x, y


def sequence_objective(model, x, y, cfg: Config):
    state = None
    prediction_losses, vq_losses, graph_losses, ponder_losses = [], [], [], []
    all_assignments = []
    for t in range(x.size(1)):
        logits, state, metrics = model.forward_step(x[:, t], state)
        prediction_losses.append(F.cross_entropy(logits, y[:, t]))
        vq_losses.append(metrics["vq_loss"])
        graph_losses.append(metrics["graph_loss"])
        ponder_losses.append(metrics["ponder"])
        # [recursion,batch,symbol] -> [batch,symbol], averaged over recursion
        all_assignments.append(metrics["soft_assignments"].mean(dim=0))

    prediction = torch.stack(prediction_losses).mean()
    vq = torch.stack(vq_losses).mean()
    graph = torch.stack(graph_losses).mean()
    ponder = torch.stack(ponder_losses).mean()

    assignments = torch.cat(all_assignments, dim=0)
    usage = assignments.mean(dim=0)
    usage = usage / usage.sum().clamp_min(cfg.eps)
    # Minimise negative entropy: this encourages non-collapsed usage.
    diversity = (usage * torch.log(usage.clamp_min(cfg.eps))).sum()
    total = (prediction + cfg.vq_weight * vq + cfg.graph_weight * graph
             + cfg.ponder_weight * ponder + cfg.diversity_weight * diversity)
    return total, {
        "prediction": prediction,
        "vq": vq,
        "graph": graph,
        "ponder": ponder,
        "diversity_loss": diversity,
        "usage_perplexity": torch.exp(-(usage * torch.log(usage.clamp_min(cfg.eps))).sum()),
        "active_symbols": (usage > (1.0 / cfg.n_symbols) * 0.25).sum(),
    }


@torch.no_grad()
def evaluate(model, data, cfg, device):
    model.eval()
    if len(data) < 2:
        return float("nan")
    # Deterministic contiguous validation scan, with fresh state.
    state = None
    losses = []
    for i in range(len(data) - 1):
        logits, state, _ = model.forward_step(data[i:i+1].to(device), state)
        losses.append(F.cross_entropy(logits, data[i+1:i+2].to(device)).item())
    return float(np.mean(losses))


def train_model(model, train_data, val_data, cfg, device, output_dir):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.learning_rate * 0.05
    )
    history = []
    best_val = float("inf")
    best_state = None
    start = time.time()

    print(f"Device: {device} | Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Fresh initialisation: no checkpoint is loaded.")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        x, y = sample_batch(train_data, cfg.batch_size, cfg.sequence_length, device)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = sequence_objective(model, x, y, cfg)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}: {loss.item()}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        row = {
            "epoch": epoch,
            "loss": float(loss.detach()),
            "prediction_loss": float(metrics["prediction"].detach()),
            "vq_loss": float(metrics["vq"].detach()),
            "graph_loss": float(metrics["graph"].detach()),
            "ponder": float(metrics["ponder"].detach()),
            "usage_perplexity": float(metrics["usage_perplexity"].detach()),
            "active_symbols": int(metrics["active_symbols"].detach()),
            "grad_norm": float(grad_norm),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        if epoch == 1 or epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            val_nll = evaluate(model, val_data, cfg, device)
            row["validation_nll"] = val_nll
            print(
                f"Epoch {epoch:04d}/{cfg.epochs} | total={row['loss']:.4f} "
                f"pred={row['prediction_loss']:.4f} val_nll={val_nll:.4f} "
                f"ponder={row['ponder']:.2f} usage_ppx={row['usage_perplexity']:.1f} "
                f"active={row['active_symbols']} grad={row['grad_norm']:.3f}"
            )
            if math.isfinite(val_nll) and val_nll < best_val:
                best_val = val_nll
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.time() - start
    report = {
        "config": asdict(cfg),
        "epochs_completed": cfg.epochs,
        "best_validation_nll": best_val,
        "elapsed_seconds": elapsed,
        "history": history,
        "scope_warning": (
            "This experiment measures character prediction and latent transition learning "
            "on a small corpus; it does not establish formal reasoning."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "vocabulary": None,
    }, output_dir / "sacrsn_v31_repaired.pt")
    return report


@torch.no_grad()
def collect_symbol_transitions(model, data, device):
    model.eval()
    state = None
    transitions = Counter()
    symbol_counts = Counter()
    previous = None
    for token in data:
        _, state, metrics = model.forward_step(token.view(1).to(device), state)
        current = int(metrics["symbol_indices"].item())
        symbol_counts[current] += 1
        if previous is not None:
            transitions[(previous, current)] += 1
        previous = current
    return symbol_counts, transitions


@torch.no_grad()
def calibrated_anomaly_report(model, train_data, val_data, chars, mapping, device, output_dir):
    """
    Scores next-character surprisal. Threshold is calibrated from validation
    surprisal quantiles; this is a language-model anomaly proxy, not an ethical
    or semantic violation detector.
    """
    def scores(data):
        state = None
        values = []
        for i in range(len(data) - 1):
            logits, state, _ = model.forward_step(data[i:i+1].to(device), state)
            target = data[i+1:i+2].to(device)
            values.append(float(F.cross_entropy(logits, target)))
        return values

    val_scores = scores(val_data)
    if not val_scores:
        return {"status": "not_available", "reason": "validation split too short"}
    threshold = float(np.quantile(val_scores, 0.95))
    report = {
        "method": "next-character negative log likelihood",
        "validation_score_count": len(val_scores),
        "validation_mean": float(np.mean(val_scores)),
        "validation_p95_threshold": threshold,
        "validation_max": float(np.max(val_scores)),
        "interpretation": "Scores above threshold are unusual relative to this validation split only.",
        "not_claimed": "This is not semantic, ethical, or general-purpose anomaly detection.",
    }
    (output_dir / "anomaly_calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def generate_text(model, start_text, chars, mapping, max_new_tokens, device, temperature=0.8):
    model.eval()
    state = None
    ids = [mapping.get(c, 0) for c in start_text]
    for token in ids:
        _, state, _ = model.forward_step(torch.tensor([token], device=device), state)
    current = torch.tensor([ids[-1]], device=device)
    output = list(start_text)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, state, _ = model.forward_step(current, state)
            probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
            current = torch.multinomial(probs, 1).squeeze(-1)
            output.append(chars[int(current.item())])
    return "".join(output)


def save_diagnostics(model, train_data, val_data, chars, mapping, output_dir, device):
    output_dir.mkdir(parents=True, exist_ok=True)
    counts, transitions = collect_symbol_transitions(model, train_data, device)
    rows = [
        {"from_symbol": a, "to_symbol": b, "count": n}
        for (a, b), n in transitions.most_common()
    ]
    (output_dir / "symbol_transitions.json").write_text(json.dumps({
        "symbol_counts": dict(counts),
        "transitions": rows,
        "warning": "Empirical latent transitions, not extracted logical rules.",
    }, indent=2), encoding="utf-8")

    probs = model.graph.probabilities().detach().cpu().numpy()
    np.savetxt(output_dir / "transition_matrix.csv", probs, delimiter=",", fmt="%.7f")
    generated = generate_text(model, chars[0], chars, mapping, 300, device)
    (output_dir / "generated_sample.txt").write_text(generated, encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matrix = probs
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest")
        ax.set_title("Learned latent-symbol transition probabilities")
        ax.set_xlabel("Next symbol")
        ax.set_ylabel("Previous symbol")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / "transition_matrix.png", dpi=140)
        plt.close(fig)
    except ImportError:
        print("matplotlib not installed; CSV transition matrix was saved instead.")


def main():
    parser = argparse.ArgumentParser(description="Train repaired SACRSN v31.1 from fresh initialisation.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--text-file", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cfg = Config(seed=args.seed)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    set_seed(cfg.seed)

    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    else:
        text = DEFAULT_TEXT
    chars, mapping, train_data, val_data = split_text(text, cfg.validation_fraction)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("runs") / f"fresh_{stamp}_seed{cfg.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SACRSNv31(len(chars), cfg).to(device)
    report = train_model(model, train_data, val_data, cfg, device, output_dir)
    save_diagnostics(model, train_data, val_data, chars, mapping, output_dir, device)
    anomaly = calibrated_anomaly_report(model, train_data, val_data, chars, mapping, device, output_dir)
    print("\nRun complete.")
    print(f"Best validation NLL: {report['best_validation_nll']:.4f}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Anomaly calibration: {anomaly}")


if __name__ == "__main__":
    main()
