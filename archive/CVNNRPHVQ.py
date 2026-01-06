import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
CONFIG = {
    "seq_len": 32,
    "embedding_dim": 64,
    "max_recursion_depth": 6,
    "p_keep": 0.9,
    "lookahead_steps": 3,
    "memory_span": 10,

    "n_symbols": 32,
    "n_concepts": 8,
    "commitment_cost": 0.25,
    "dynamic_threshold": 2.0,

    "epochs": 350,
    "learning_rate": 0.003,
    "grad_clip": 1.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "w_prediction": 1.0,
    "w_variance": 0.1,
    "w_depth_consistency": 0.1,
    "w_stability": 0.1,
    "w_depth_usage": 0.05,
    "w_symbolic": 0.1,
    "w_hierarchy": 0.1,

    "noise_std": 0.05,
    "eps": 1e-8
}

# ==========================================
# 2. Dataset
# ==========================================
TEXT_DATA = """True, without falsehood, certain and most true. 
That which is above is like to that which is below, 
and that which is below is like to that which is above.
The father of all perfection in the whole world is here.
Its force or power is entire if it be converted into earth."""

chars = sorted(set(TEXT_DATA))
vocab_size = len(chars)
char_to_ix = {c: i for i, c in enumerate(chars)}
ix_to_char = {i: c for i, c in enumerate(chars)}

data_tensor = torch.tensor(
    [char_to_ix[c] for c in TEXT_DATA],
    dtype=torch.long,
    device=CONFIG["device"]
)

# ==========================================
# 3. Complex Core
# ==========================================
class ComplexLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.r = nn.Linear(dim, dim, bias=False)
        self.i = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.r.weight)
        nn.init.orthogonal_(self.i.weight)

    def forward(self, z):
        return torch.complex(
            self.r(z.real) - self.i(z.imag),
            self.r(z.imag) + self.i(z.real)
        )

class RecursiveCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = ComplexLinear(dim)

    def forward(self, z):
        z = self.lin(z)
        mag = torch.sqrt(z.real**2 + z.imag**2 + CONFIG["eps"])
        z = z / (1.0 + mag)
        return torch.tanh(z.real) + 1j * torch.tanh(z.imag)

class ComplexAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim*2, dim)
        self.k = nn.Linear(dim*2, dim)
        self.v = nn.Linear(dim*2, dim*2)
        self.scale = dim ** -0.5

    def forward(self, history, current, confidence=None):
        if not history:
            return current

        Q = self.q(torch.cat([current.real, current.imag], -1)).unsqueeze(2)

        hr = torch.stack([h.real for h in history], 2)
        hi = torch.stack([h.imag for h in history], 2)
        H = torch.cat([hr, hi], -1)

        K = self.k(H)
        V = self.v(H)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if confidence is not None:
            scores = scores * confidence.unsqueeze(-1).unsqueeze(-1)

        w = F.softmax(scores, dim=-1)
        ctx = torch.matmul(w, V).squeeze(2)

        d = ctx.shape[-1] // 2
        return torch.complex(
            current.real + 0.1 * ctx[..., :d],
            current.imag + 0.1 * ctx[..., d:]
        )

# ==========================================
# 4. Hierarchical Dynamic VQ
# ==========================================
class DynamicHierarchicalVQ(nn.Module):
    def __init__(self, dim, n_sym, n_con):
        super().__init__()
        self.sym = nn.Parameter(torch.randn(n_sym, dim*2))
        self.con = nn.Parameter(torch.randn(n_con, dim*2))

    def quantize(self, z, book):
        d = (
            z.pow(2).sum(-1, keepdim=True)
            + book.pow(2).sum(-1)
            - 2 * z @ book.T
        )
        idx = d.argmin(-1)
        dist = d.min(-1).values

        if self.training and dist.mean() > CONFIG["dynamic_threshold"]:
            book.data[
                torch.randint(0, book.size(0), (1,)).item()
            ] = z.mean((0,1)).detach()

        zq = F.embedding(idx, book)
        loss = (
            F.mse_loss(zq, z.detach())
            + CONFIG["commitment_cost"] * F.mse_loss(zq.detach(), z)
        )

        zq = z + (zq - z).detach()
        return zq, loss, idx, dist

    def forward(self, z):
        zf = torch.cat([z.real, z.imag], -1)
        zs, ls, si, sd = self.quantize(zf, self.sym)
        zc, lc, ci, _ = self.quantize(zs, self.con)

        d = zs.shape[-1] // 2
        return (
            torch.complex(zs[..., :d], zs[..., d:]),
            F.one_hot(si, self.sym.size(0)).float(),
            F.one_hot(ci, self.con.size(0)).float(),
            ls, lc, si, ci, 1/(1+sd)
        )

# ==========================================
# 5. Model
# ==========================================
class CeilingSymbolicRNN(nn.Module):
    def __init__(self):
        super().__init__()
        d = CONFIG["embedding_dim"]
        self.mag = nn.Embedding(vocab_size, d)
        self.phase = nn.Parameter(torch.randn(vocab_size, d))
        self.cell = RecursiveCell(d)
        self.attn = ComplexAttention(d)
        self.gate = nn.Parameter(torch.zeros(d))
        self.dec = nn.Linear(d*2, vocab_size)
        self.vq = DynamicHierarchicalVQ(d, CONFIG["n_symbols"], CONFIG["n_concepts"])

    def embed(self, x):
        r = self.mag(x)
        t = self.phase[x]
        return torch.complex(r*torch.cos(t), r*torch.sin(t))

    def forward(self, x, h=None, mem=None, collect=False, training=False):
        z = self.embed(x)
        g = torch.sigmoid(self.gate).view(1,1,-1)
        z = z if h is None else (1-g)*h + g*z

        mem = mem or []
        depth, symL, conL = [], 0, 0
        conf = None

        for _ in range(CONFIG["max_recursion_depth"]):
            if training and torch.rand(1) > CONFIG["p_keep"]:
                if collect: depth.append(z)
                continue

            z = self.cell(z)
            z = self.attn(mem, z, conf)

            z, sp, cp, ls, lc, si, ci, conf = self.vq(z)
            z = 0.5*z + 0.5*z

            symL += ls
            conL += lc
            mem.append(z)
            mem[:] = mem[-CONFIG["memory_span"]:]

            if collect: depth.append(z)

        for _ in range(CONFIG["lookahead_steps"]):
            z = self.cell(z)

        out = self.dec(torch.cat([z.real, z.imag], -1))
        return out, z, depth, len(depth), mem, sp, cp, symL, conL, si, ci

# ==========================================
# 6. CORRECTED LOSSES
# ==========================================
def variance_loss(z):
    z = z.reshape(-1, z.size(-1))
    if z.size(0) <= 1:
        return torch.tensor(0.0, device=z.device)

    var = z.var(dim=0, unbiased=False)
    std = torch.sqrt(var + CONFIG["eps"])
    return torch.mean(F.relu(1.0 - std))

def depth_consistency_loss(states):
    if len(states) < 2:
        return torch.tensor(0.0, device=CONFIG["device"])
    return sum(
        ((states[i]-states[i-1]).real**2 +
         (states[i]-states[i-1]).imag**2).mean()
        for i in range(1,len(states))
    ) / len(states)

def stability_loss(a,b):
    return ((a-b).real**2 + (a-b).imag**2).mean()

# ==========================================
# 7. Training
# ==========================================
def train():
    m = CeilingSymbolicRNN().to(CONFIG["device"])
    opt = torch.optim.AdamW(m.parameters(), lr=CONFIG["learning_rate"])

    for e in range(CONFIG["epochs"]):
        h, mem, tot = None, [], 0
        for i in range(len(data_tensor)-CONFIG["seq_len"]-1):
            x = data_tensor[i:i+CONFIG["seq_len"]].unsqueeze(0)
            y = data_tensor[i+1:i+CONFIG["seq_len"]+1].unsqueeze(0)

            o, z, d, du, mem, _, _, ls, lc, _, _ = m(x, h, mem, True, True)
            mem = [s.detach() for s in mem]

            zn = None if h is None else h + CONFIG["noise_std"] * (
                torch.randn_like(h.real) + 1j*torch.randn_like(h.imag)
            )

            _, zp, *_ = m(x, zn, mem.copy(), False, False)

            lp = F.cross_entropy(o.view(-1, vocab_size), y.view(-1))
            loss = (
                lp +
                CONFIG["w_variance"] * variance_loss(torch.cat([z.real,z.imag],-1)) +
                CONFIG["w_depth_consistency"] * depth_consistency_loss(d) +
                CONFIG["w_stability"] * stability_loss(z,zp) +
                CONFIG["w_depth_usage"] * (1-du/CONFIG["max_recursion_depth"]) +
                CONFIG["w_symbolic"] * ls +
                CONFIG["w_hierarchy"] * lc
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), CONFIG["grad_clip"])
            opt.step()

            h = z.detach()
            tot += loss.item()

        if e % 50 == 0:
            print(f"Epoch {e:03d} | Loss {tot:.4f}")

    return m

# ==========================================
# 8. Run
# ==========================================
if __name__ == "__main__":
    model = train()
