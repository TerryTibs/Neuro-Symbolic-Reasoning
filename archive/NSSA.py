# ============================================================
# Neuro-Symbolic Snake Agent
# Long-Term Symbol Memory + Symbolic Planning
# Fixed RNN Backprop Graph
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, defaultdict

# ============================================================
# 1. Environment
# ============================================================

class SnakeEnv:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = [(5, 5)]
        self.food = self._spawn_food()
        return self._get_state()

    def _spawn_food(self):
        while True:
            f = (random.randint(0, 9), random.randint(0, 9))
            if f not in self.snake:
                return f

    def _get_state(self):
        hx, hy = self.snake[0]
        fx, fy = self.food
        vec = np.array([
            hx / self.grid_size,
            hy / self.grid_size,
            fx / self.grid_size,
            fy / self.grid_size,
            (fx - hx) / self.grid_size,
            (fy - hy) / self.grid_size
        ], dtype=np.float32)

        raw = {
            "NEAR_WALL": hx in [0, self.grid_size - 1] or hy in [0, self.grid_size - 1],
            "FOOD_VISIBLE": True
        }
        return vec, raw

    def step(self, action):
        moves = [(1,0), (-1,0), (0,1), (0,-1)]
        dx, dy = moves[action]

        hx, hy = self.snake[0]
        nh = (hx + dx, hy + dy)

        if not (0 <= nh[0] < self.grid_size and 0 <= nh[1] < self.grid_size):
            return self._get_state(), -1.0, True
        if nh in self.snake:
            return self._get_state(), -1.0, True

        self.snake.insert(0, nh)
        reward = -0.01
        done = False

        if nh == self.food:
            reward = 1.0
            self.food = self._spawn_food()
        else:
            self.snake.pop()

        return self._get_state(), reward, done

# ============================================================
# 2. Symbol
# ============================================================

class Symbol:
    def __init__(self, name, dim, persistence=0.9):
        self.name = name
        self.prototype = torch.randn(dim) * 0.01
        self.activation = 0.0
        self.persistence = persistence

# ============================================================
# 3. Long-Term Symbol Memory
# ============================================================

class SymbolMemory:
    def __init__(self):
        self.working = deque(maxlen=50)
        self.long_term = defaultdict(float)

    def update(self, symbols):
        snapshot = {k: v.activation for k, v in symbols.items()}
        self.working.append(snapshot)

        # Consolidate slowly
        for k, v in snapshot.items():
            self.long_term[k] = 0.99 * self.long_term[k] + 0.01 * v

    def novelty(self):
        if len(self.working) < 2:
            return 0.0
        a = self.working[-1]
        b = self.working[-2]
        return sum(abs(a[k] - b.get(k, 0)) for k in a)

# ============================================================
# 4. Hierarchical Resonance
# ============================================================

class HierarchicalResonance(nn.Module):
    def __init__(self, input_dim, symbol_dim, lr=0.05):
        super().__init__()
        self.encoder = nn.Linear(input_dim, symbol_dim)
        self.lr = lr
        self.low = {"FOOD": Symbol("FOOD", symbol_dim),
                    "WALL": Symbol("WALL", symbol_dim)}
        self.mid = {"GOAL": Symbol("GOAL", symbol_dim),
                    "THREAT": Symbol("THREAT", symbol_dim)}
        self.high = {"SURVIVAL": Symbol("SURVIVAL", symbol_dim)}

    def update_symbol(self, s, z):
        sim = torch.cosine_similarity(z, s.prototype + 1e-6, dim=0)
        s.activation = s.persistence * s.activation + (1 - s.persistence) * sim.item()
        s.prototype += self.lr * s.activation * (z.detach() - s.prototype)

    def forward(self, vec, raw):
        z = torch.tanh(self.encoder(vec))

        for s in self.low.values():
            self.update_symbol(s, z)
        if raw["NEAR_WALL"]:
            self.low["WALL"].activation = 1.0

        self.mid["GOAL"].activation = self.low["FOOD"].activation
        self.mid["THREAT"].activation = self.low["WALL"].activation
        for s in self.mid.values():
            self.update_symbol(s, z)

        self.high["SURVIVAL"].activation = max(
            self.mid["GOAL"].activation,
            1.0 - self.mid["THREAT"].activation
        )
        for s in self.high.values():
            self.update_symbol(s, z)

        return z, {**self.low, **self.mid, **self.high}

# ============================================================
# 5. Cognitive Core
# ============================================================

class ComplexGatedRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.reset()

    def reset(self):
        self.h = torch.zeros(self.hidden_dim)

    def forward(self, x):
        # Detach hidden state to avoid backward-through-graph errors
        self.h = self.h.detach()
        self.h = torch.tanh(self.W(torch.cat([x, self.h])))
        return self.h

# ============================================================
# 6. Policy
# ============================================================

class Policy(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 4)

    def forward(self, h):
        return self.fc(h)

# ============================================================
# 7. Symbolic Planner
# ============================================================

class SymbolicPlanner:
    def __init__(self, horizon=3):
        self.horizon = horizon

    def plan(self, symbols):
        score = 0.0
        survival = symbols["SURVIVAL"].activation
        threat = symbols["THREAT"].activation
        for t in range(self.horizon):
            survival = max(survival - 0.1 * threat, 0)
            score += survival
        return score

# ============================================================
# 8. EthicalAI
# ============================================================

class EthicalAI:
    def filter(self, logits, symbols):
        adjusted = logits.clone()
        if symbols["THREAT"].activation > 0.8:
            adjusted[0] -= 2.5
        return torch.argmax(adjusted).item()

# ============================================================
# 9. Training Loop
# ============================================================

env = SnakeEnv()
resonance = HierarchicalResonance(6, 16)
cognition = ComplexGatedRNN(16, 32)
policy = Policy(32)
ethics = EthicalAI()
planner = SymbolicPlanner(horizon=4)
memory = SymbolMemory()

optimizer = optim.Adam(list(resonance.parameters()) + list(policy.parameters()), lr=1e-3)
beta_curiosity = 0.1
beta_plan = 0.2
episodes = 900

for ep in range(episodes):
    vec, raw = env.reset()
    cognition.reset()
    memory.working.clear()
    done = False
    total_reward = 0

    while not done:
        vec_t = torch.tensor(vec)

        z, symbols = resonance(vec_t, raw)
        memory.update(symbols)

        h = cognition(z)
        logits = policy(h)

        # Planning bonus
        plan_score = planner.plan(symbols)

        action = ethics.filter(logits, symbols)
        (vec_next, raw_next), ext_reward, done = env.step(action)

        curiosity = memory.novelty()
        reward = ext_reward + beta_curiosity * curiosity + beta_plan * plan_score

        logp = torch.log_softmax(logits, dim=0)[action]
        loss = -reward * logp

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        vec, raw = vec_next, raw_next
        total_reward += reward

    if ep % 20 == 0:
        print(
            f"Ep {ep:03d} | R {total_reward:.2f} | "
            f"SURV {symbols['SURVIVAL'].activation:.2f} | "
            f"LT_SURV {memory.long_term['SURVIVAL']:.2f}"
        )
