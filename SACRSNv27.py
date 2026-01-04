# ============================================================
# SACRSN v26: THE NORMALISED OMNI-SCIENTIFIC EDITION
# Fixes: Autograd Safety, Persistent Stack, Normative Ethics
# Enhanced: Gradient Clipping, Attention, Batch Training
# ============================================================

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

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
    "seq_len": 32,
    "embedding_dim": 64,
    "n_symbols": 64,
    
    # Reasoning
    "max_recursion_depth": 8,
    "act_threshold": 0.9999,
    "ponder_penalty": 0.0001,
    
    # Memory
    "use_stack": True,
    "stack_size": 16,
    
    # Topology
    "commitment_cost": 0.25,
    "graph_bias_scale": 0.5,
    "symbol_consistency_weight": 0.01,
    "ethical_weight": 0.005,  # Reduced ethical weight
    
    # Training
    "epochs": 3000,
    "learning_rate": 1e-3,
    "grad_clip": 0.1,  # Lowered gradient clipping
    "eps": 1e-6,
    
    # Batch Training
    "batch_size": 8,
    "warmup_epochs": 100
}

# ==========================================
# 2. Data
# ==========================================
TEXT_DATA = """True, without falsehood, certain and most true. 
That which is above is like to that which is below, 
and that which is below is like to that which is above.
The father of all perfection in the whole world is here.
Its force or power is entire if it be converted into earth."""

chars = sorted(list(set(TEXT_DATA)))
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
data_tensor = torch.tensor([char_to_ix[c] for c in TEXT_DATA], dtype=torch.long).to(DEVICE)

# ==========================================
# 3. Complex Primitives
# ==========================================
class ComplexLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
    def forward(self, z):
        mag = torch.abs(z) + CONFIG["eps"]
        mean = mag.mean(dim=-1, keepdim=True)
        var = mag.var(dim=-1, keepdim=True)
        norm_mag = (mag - mean) / torch.sqrt(var + CONFIG["eps"])
        norm_mag = norm_mag * self.scale + self.shift
        phase = torch.angle(z)
        return torch.complex(norm_mag * torch.cos(phase), norm_mag * torch.sin(phase))

class ModReLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, z):
        norm = torch.abs(z) + CONFIG["eps"]
        scale = F.relu(norm + self.bias) / norm
        return z * scale

class ComplexLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc_real = nn.Linear(dim, dim, bias=False)
        self.fc_imag = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_uniform_(self.fc_real.weight)
        nn.init.xavier_uniform_(self.fc_imag.weight)
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
        # control: [Batch, 3]
        push, pop, noop = control[:, 0].view(-1,1), control[:, 1].view(-1,1), control[:, 2].view(-1,1)
        
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
        self.codebook = nn.Parameter(torch.randn(n_symbols, latent_dim*2))
        self.adjacency = nn.Parameter(torch.zeros(n_symbols, n_symbols))
    
    def forward(self, z, prev_symbol_idx=None):
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        d = torch.sum(z_flat**2, dim=-1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=-1) - \
            2 * torch.matmul(z_flat, self.codebook.t())
        
        if prev_symbol_idx is not None:
            # Handle batch index for adjacency
            # We take the mean adjacency effect for the batch (Simplified for batch training)
            # In a rigorous sequence model, each batch item has its own prev_sym.
            # Here we assume prev_symbol_idx is [Batch]
            if prev_symbol_idx.dim() > 0:
                # We can't easily bias the whole batch matrix efficiently with different priors per item without a loop or huge tensor expansion.
                # Simplification: Use the mode (most common symbol) or skip bias for batch training step.
                # For v26 accuracy, let's skip bias during batch training to avoid shape mismatch errors, 
                # but enable it during inference (batch_size=1).
                pass 
            else:
                 graph_prior = self.adjacency[prev_symbol_idx]
                 bias = CONFIG["graph_bias_scale"] * torch.sigmoid(graph_prior)
                 d = d - bias

        min_indices = torch.argmin(d, dim=-1)
        z_q = F.embedding(min_indices, self.codebook)
        
        loss_vq = F.mse_loss(z_q, z_flat.detach())
        loss_commit = F.mse_loss(z_q.detach(), z_flat)
        
        encodings = F.one_hot(min_indices, self.n_symbols).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        z_q = z_flat + (z_q - z_flat).detach()
        z_complex = torch.complex(z_q[..., :z.shape[-1]], z_q[..., z.shape[-1]:])
        
        return z_complex, loss_vq + loss_commit * CONFIG["commitment_cost"], min_indices, perplexity

# ==========================================
# 5. Enhanced Attention Module
# ==========================================
class ComplexAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = ComplexLinear(dim)
        self.k_proj = ComplexLinear(dim)
        self.v_proj = ComplexLinear(dim)
        self.scale = dim ** -0.5
    
    def forward(self, z, mask=None):
        q = self.q_proj(z)
        k = self.k_proj(z)
        v = self.v_proj(z)
        
        # Complex Dot Product Attention
        # (a+bi)(c-di) = (ac+bd) + i(bc-ad) -> We use the Real part for attention scores
        # Simplified: Just match magnitudes/phases implicitly via linear layers
        
        # To make this robust, we process Real/Imag parts as a concatenated vector for the softmax
        q_flat = torch.cat([q.real, q.imag], dim=-1)
        k_flat = torch.cat([k.real, k.imag], dim=-1)
        
        # Simple Dot Product on flat representation
        attn_scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply weights to V (Keep it complex)
        # We separate V into Real/Imag to multiply
        v_real = torch.matmul(attn_weights, v.real)
        v_imag = torch.matmul(attn_weights, v.imag)
        
        return torch.complex(v_real, v_imag)

# ==========================================
# 6. Enhanced Transformer Layer
# ==========================================
class MultiHeadComplexTransformer(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.attention = nn.ModuleList([ComplexAttention(dim) for _ in range(heads)])
        self.output_proj = ComplexLinear(dim)
        self.norm = ComplexLayerNorm(dim)
    
    def forward(self, z):
        # Multi-head average
        outputs = [head(z) for head in self.attention]
        # Stack and average complex tensors
        real_avg = torch.stack([o.real for o in outputs], dim=0).mean(dim=0)
        imag_avg = torch.stack([o.imag for o in outputs], dim=0).mean(dim=0)
        combined = torch.complex(real_avg, imag_avg)
        
        return self.norm(self.output_proj(combined))

# ==========================================
# 7. Core Processor & [FIXED] Ethical Layer
# ==========================================
class EthicalConstraint(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, prev_sym, curr_sym, adjacency):
        if prev_sym is None: return torch.tensor(0.0).to(adjacency.device)
        
        # [FIX] Normative Ethics
        # Handling Batch indices for lookup
        # adjacency: [N, N]
        # prev_sym: [Batch]
        # curr_sym: [Batch]
        
        if prev_sym.dim() == 0: # Single item
             row_logits = adjacency[prev_sym]
             return F.cross_entropy(row_logits.unsqueeze(0), curr_sym.unsqueeze(0))
        else: # Batch
             row_logits = adjacency[prev_sym] # [Batch, N]
             return F.cross_entropy(row_logits, curr_sym)

class AdaptiveRecursiveCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = ComplexLinear(dim)
        self.norm = ComplexLayerNorm(dim) 
        self.act = ModReLU(dim)
        self.halt_linear = nn.Linear(dim * 2, 1)
        self.stack_ctrl = nn.Linear(dim * 2, 3)
        self.attention = ComplexAttention(dim) # Added Self-Attention to cell
        nn.init.constant_(self.halt_linear.bias, -2.0) 
    
    def forward(self, z):
        z_proc = self.act(self.norm(self.linear(z)))
        # Self-Attention on the thought vector
        z_proc = self.attention(z_proc)
        
        z_flat = torch.cat([z_proc.real, z_proc.imag], dim=-1)
        halt_prob = torch.sigmoid(self.halt_linear(z_flat))
        stack_probs = F.softmax(self.stack_ctrl(z_flat), dim=-1)
        return z_proc, halt_prob, stack_probs

# ==========================================
# 8. Master Model (UberCRSN)
# ==========================================
class UberCRSN(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.emb_mag = nn.Embedding(vocab_size, dim)
        self.emb_phase = nn.Parameter(torch.randn(vocab_size, dim))
        self.cell = AdaptiveRecursiveCell(dim)
        self.vq_layer = GraphMemoryVQ(dim, CONFIG["n_symbols"])
        self.decoder = nn.Linear(dim*2, vocab_size)
        
        if CONFIG["use_stack"]:
            self.stack = DifferentiableStack(dim, CONFIG["stack_size"])
            
        self.ethics = EthicalConstraint()
        self.register_buffer("prev_sym_soft", torch.zeros(CONFIG["n_symbols"]))
        # Added Transformer for global context processing
        self.transformer = MultiHeadComplexTransformer(dim)

    def embed(self, idx):
        r = self.emb_mag(idx)
        t = self.emb_phase[idx]
        return torch.complex(r*torch.cos(t), r*torch.sin(t))

    def forward(self, input_ids, hidden=None, prev_sym=None):
        # input_ids: [Batch, 1]
        batch_size = input_ids.size(0)
        z = self.embed(input_ids).squeeze(1)
        
        if hidden is None:
            z_prev = torch.zeros_like(z)
            stack_mem = torch.zeros(batch_size, CONFIG["stack_size"], self.dim*2, device=z.device)
            stack_ptr = torch.zeros(batch_size, CONFIG["stack_size"], device=z.device)
            stack_ptr[:, 0] = 1.0
        else:
            z_prev, stack_mem, stack_ptr = hidden
            z = 0.5 * z + 0.5 * z_prev

        act_step = 0
        halting_probability = torch.zeros(batch_size, 1).to(z.device)
        remain = torch.ones(batch_size, 1).to(z.device)
        ponder_cost = 0
        stack_history = [] 
        
        z_weighted = torch.zeros_like(z) 
        current_sym = prev_sym
        vq_loss_total = 0
        perplexity_total = 0
        ethical_loss_total = 0
        
        # --- RECURSION LOOP ---
        for t in range(CONFIG["max_recursion_depth"]):
            act_step += 1
            z_proc, p_halt, stack_ctrl = self.cell(z)
            
            if CONFIG["use_stack"]:
                stack_read, stack_mem, stack_ptr = self.stack(z_proc, stack_mem, stack_ptr, stack_ctrl)
                z_combined = z_proc + stack_read
                depth = torch.sum(stack_ptr * torch.arange(CONFIG["stack_size"], device=z.device), dim=1)
                stack_history.append(depth)
            else:
                z_combined = z_proc
                stack_history.append(torch.zeros(1).to(z.device))

            z_sym, vq_loss, sym_idx, perplexity = self.vq_layer(z_combined, current_sym)
            
            eth_loss = self.ethics(current_sym, sym_idx, self.vq_layer.adjacency)
            ethical_loss_total += eth_loss
            current_sym = sym_idx
            
            z = 0.7 * z_combined + 0.3 * z_sym
            
            still_running = (halting_probability < CONFIG["act_threshold"]).float()
            p = p_halt * still_running
            if t == CONFIG["max_recursion_depth"] - 1: p = remain
            
            z_weighted = z_weighted + (p * z)
            halting_probability = halting_probability + p
            remain = remain - p
            ponder_cost += still_running.mean()
            vq_loss_total += vq_loss
            perplexity_total += perplexity

        features = torch.cat([z_weighted.real, z_weighted.imag], dim=-1)
        logits = self.decoder(features)
        
        next_hidden = (z_weighted, stack_mem, stack_ptr)
        
        # Flatten stack history for batch return (just take mean for logging)
        if len(stack_history) > 0:
            stack_hist_mean = torch.stack(stack_history).mean().item()
        else:
            stack_hist_mean = 0.0
            
        return logits, next_hidden, current_sym, ponder_cost, vq_loss_total, perplexity_total/act_step, ethical_loss_total, stack_hist_mean

# ==========================================
# 9. Training Engine with Batch Support
# ==========================================
def train():
    model = UberCRSN(vocab_size, CONFIG["embedding_dim"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
    
    # Enhanced scheduler with warmup
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=CONFIG["warmup_epochs"])
    if CONFIG["epochs"] > CONFIG["warmup_epochs"]:
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"] - CONFIG["warmup_epochs"], eta_min=1e-5)
        scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[scheduler, main_scheduler], milestones=[CONFIG["warmup_epochs"]])

    print(f"--- Training SACRSN v26 Enhanced ---")
    print(f"--- Fixes: AutoGrad | Stack Persist | Normative Ethics | Attention | Batch ---")
    
    try:
        prev_avg_loss = float('inf')
        patience = 0
        
        for epoch in range(CONFIG["epochs"]):
            hidden = None
            prev_sym = None
            total_loss = 0
            total_ponder = 0
            total_ppx = 0
            window = 16 
            entropy_weight = 0.01 * (1 - epoch / CONFIG["epochs"])
            
            # Batch processing
            batch_size = CONFIG["batch_size"]
            num_samples = len(data_tensor) - 1
            num_batches = (num_samples + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                
                # Get Batch [Batch_Size, 1]
                batch_x = data_tensor[start_idx:end_idx].view(-1, 1)
                batch_y = data_tensor[start_idx+1:end_idx+1].view(-1)
                
                # Forward Pass
                logits, hidden, sym_idx, ponder, vq_loss, ppx, eth_loss, _ = model(batch_x, hidden, prev_sym)
                
                # Detach hidden
                h_z, h_mem, h_ptr = hidden
                hidden = (h_z.detach(), h_mem.detach(), h_ptr.detach())
                prev_sym = sym_idx.detach()
                
                loss_pred = F.cross_entropy(logits, batch_y)
                loss_ponder = CONFIG["ponder_penalty"] * ponder
                
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                loss_entropy = -entropy_weight * (-(probs * log_probs).sum())
                
                adj_sig = torch.sigmoid(model.vq_layer.adjacency)
                row_entropy = -(adj_sig * torch.log(adj_sig + CONFIG["eps"])).sum(dim=-1).mean()
                loss_static = CONFIG["symbol_consistency_weight"] * row_entropy
                
                # Temporal Consistency
                curr_onehot = F.one_hot(sym_idx, CONFIG["n_symbols"]).float()
                # Batch Handling for OneHot
                if curr_onehot.dim() == 2: 
                     # Average over batch for buffer update (Simplification for v26)
                     curr_onehot_mean = curr_onehot.mean(dim=0)
                else:
                     curr_onehot_mean = curr_onehot
                     
                loss_temporal = CONFIG["symbol_consistency_weight"] * F.mse_loss(
                    curr_onehot_mean, 
                    model.prev_sym_soft.detach()
                )
                
                with torch.no_grad():
                    model.prev_sym_soft.copy_(
                        model.prev_sym_soft * 0.9 + curr_onehot_mean * 0.1
                    )
                
                loss_ethics = CONFIG["ethical_weight"] * eth_loss
                
                # Enhanced loss composition
                total_loss_batch = (loss_pred * 1.0) + \
                                  (loss_ponder * 0.01) + \
                                  (vq_loss * 0.05) + \
                                  (loss_entropy * 0.02) + \
                                  (loss_static * 0.01) + \
                                  (loss_temporal * 0.01) + \
                                  (loss_ethics * 0.005)
                
                opt.zero_grad()
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                opt.step()
                
                total_loss += total_loss_batch.item()
                total_ponder += ponder.item()
                total_ppx += ppx.item()
                
            scheduler.step()

            if epoch % 50 == 0:
                avg_loss = total_loss / num_batches
                avg_ponder = total_ponder / num_batches
                avg_ppx = total_ppx / num_batches
                lr = scheduler.get_last_lr()[0]
                print(f"Ep {epoch:04d} | Loss: {avg_loss:.4f} | Steps: {avg_ponder:.2f} | Usage(PPX): {avg_ppx:.1f} | LR: {lr:.6f}")
                
                if avg_loss < 0.01:
                    print("\n--- PERFECT CONVERGENCE ---")
                    return model
                
                # Convergence Monitor
                if epoch > 200 and avg_loss > prev_avg_loss:
                    patience += 1
                    if patience > 20: 
                        print("Early stopping due to no improvement")
                        return model
                prev_avg_loss = min(prev_avg_loss, avg_loss)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    
    return model

# ==========================================
# 10. Visualization Suite (X-RAY)
# ==========================================
def visualize_all(model):
    print("\n--- Generating Diagnostics & Images ---")
    model.eval()
    
    # 1. Semantic Mapping
    print("Mapping Symbols to Text...")
    symbol_to_char = defaultdict(list)
    hidden, prev_sym = None, None
    
    with torch.no_grad():
        # Single batch inference for viz
        for i in range(len(data_tensor) - 1):
            x = data_tensor[i].view(1, 1)
            _, hidden, prev_sym, _, _, _, _, _ = model(x, hidden, prev_sym)
            current_char = ix_to_char[data_tensor[i].item()]
            symbol_to_char[prev_sym.item()].append(current_char)

    node_labels = {}
    for sym_idx in range(CONFIG["n_symbols"]):
        char_list = symbol_to_char.get(sym_idx, [])
        if char_list:
            most_common = max(set(char_list), key=char_list.count)
            node_labels[sym_idx] = f"{most_common}\n({len(char_list)})"
        else:
            node_labels[sym_idx] = f"{sym_idx}"

    # 2. X-Ray Topology
    adj_probs = torch.sigmoid(model.vq_layer.adjacency).detach().cpu().numpy()
    G = nx.DiGraph()
    for i in range(CONFIG["n_symbols"]): G.add_node(i)

    edges, weights = [], []
    for i in range(CONFIG["n_symbols"]):
        for j in range(CONFIG["n_symbols"]):
            w = adj_probs[i, j]
            if w > 0.05: 
                G.add_edge(i, j, weight=w)
                edges.append((i, j))
                weights.append(w)
    
    plt.figure(figsize=(14, 14))
    try: pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
    except: pos = nx.circular_layout(G)
    
    node_colors = ['#a0cbe2' if i in symbol_to_char else '#ffe5e5' for i in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight="bold")
    
    for (u, v), w in zip(edges, weights):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=w * 2.0, alpha=max(0.1, w), edge_color='gray', arrowstyle='->', arrowsize=10)
    
    plt.title(f"1_semantic_topology (Active: {len(symbol_to_char)})")
    plt.savefig("1_semantic_topology.png", dpi=150)
    print("Saved 1_semantic_topology.png")
    plt.close()

    # 3. Inference Scan
    hidden, prev_sym = None, None
    x = torch.tensor([[char_to_ix["T"]]], device=DEVICE)
    
    stack_history = []
    act_history = []
    gen_text = "T"
    
    print("Running Inference Scan...")
    for _ in range(200):
        with torch.no_grad():
            logits, hidden, prev_sym, ponder, _, _, _, s_hist = model(x, hidden, prev_sym)
            
            # Stack depth
            stack_history.append(s_hist)
            act_history.append(1.0 + ponder.item())

            probs = F.softmax(logits, dim=-1)
            next_ix = torch.multinomial(probs, 1)
            char = ix_to_char[next_ix.item()]
            gen_text += char
            x = next_ix

    print(f"Generated: {gen_text}\n")

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
    plt.title("3_act_profile (Thinking Steps)")
    plt.savefig("3_act_profile.png")
    plt.close()

def extract_logic_rules(model, data_tensor):
    print("\n--- Extracting Explicit Logic Rules ---")
    model.eval()
    rule_book = defaultdict(list)
    hidden = None
    prev_sym = None
    
    with torch.no_grad():
        for i in range(len(data_tensor) - 1):
            x = data_tensor[i].view(1, 1)
            logits, hidden, sym_idx, ponder, _, _, _, _ = model(x, hidden, prev_sym)
            if prev_sym is not None:
                src = prev_sym.item()
                dst = sym_idx.item()
                rule_book[(src, dst)].append(ponder.item())
            prev_sym = sym_idx

    print(f"\n{'FROM':<6} | {'TO':<6} | {'COUNT':<6} | {'AVG STEPS':<10}")
    print("-" * 45)
    sorted_rules = sorted(rule_book.items(), key=lambda x: len(x[1]), reverse=True)
    for (src, dst), ponders in sorted_rules:
        if len(ponders) > 1:
            print(f"S_{src:<4} -> S_{dst:<4} | {len(ponders):<6} | {sum(ponders)/len(ponders):.2f}")

# ==========================================
# 11. Advanced Interaction (System 2 Features)
# ==========================================
def dream_mode(model):
    print("\n--- 🌙 Dream Mode (Pure Symbolic Walk) ---")
    adj = torch.sigmoid(model.vq_layer.adjacency).detach().cpu().numpy()
    
    model.eval()
    x = torch.tensor([[char_to_ix["T"]]], device=DEVICE)
    _, _, prev_sym, _, _, _, _, _ = model(x, None, None)
    curr_sym = prev_sym.item()
    
    output = "T"
    
    for _ in range(100):
        probs = adj[curr_sym]
        probs[probs < 0.2] = 0 
        
        if probs.sum() == 0: break
        probs = probs / probs.sum()
        next_sym = np.random.choice(len(probs), p=probs)
        
        # Direct Codebook Decode
        z_flat = model.vq_layer.codebook[next_sym].unsqueeze(0)
        logits = model.decoder(z_flat)
        char_idx = torch.argmax(logits).item()
        
        output += ix_to_char[char_idx]
        curr_sym = next_sym
        
    print(f"Dream Output: {output}\n")

def anomaly_detector(model):
    print("\n--- 🚨 Anomaly Detection Test ---")
    corrupt_text = "True without falsehood certain and most banana"
    print(f"Input: '{corrupt_text}'")
    
    input_tensor = torch.tensor([char_to_ix.get(c, 0) for c in corrupt_text], dtype=torch.long).to(DEVICE)
    hidden, prev_sym = None, None
    anomalies = []
    
    with torch.no_grad():
        for i in range(len(input_tensor) - 1):
            x = input_tensor[i].view(1, 1)
            _, hidden, prev_sym, _, _, _, eth_loss, _ = model(x, hidden, prev_sym)
            anomalies.append(eth_loss.item())

    plt.figure(figsize=(10, 4))
    plt.plot(list(corrupt_text)[1:], anomalies, marker='o', color='red')
    plt.title("Topological Violation Score (Anomaly Detection)")
    plt.grid(True, alpha=0.3)
    plt.savefig("5_anomaly_detection.png")
    print("Saved 5_anomaly_detection.png")
    plt.close()

# ==========================================
# 12. Main Execution
# ==========================================
if __name__ == "__main__":
    FILENAME = "crsn_omni_model.pth"
    
    trained_model = train()
    
    print(f"\n--- Saving Model to {FILENAME} ---")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': CONFIG,
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
