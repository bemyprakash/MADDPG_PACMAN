# 🟡 MADDPG Pac-Man — Multi-Agent Deep Deterministic Policy Gradient

A **fully custom PyTorch implementation** of the MADDPG algorithm applied to a cooperative/competitive 20×20 grid Pac-Man environment, built entirely from scratch — no external RL libraries. The system supports both **centralized-critic MADDPG** and **independent DDPG** modes, discrete action spaces via Gumbel-Softmax, a live replay buffer, and a GIF-rendering visualizer.

---

## Table of Contents

1. [Conceptual Background](#conceptual-background)
   - [Why MADDPG?](#why-maddpg)
   - [Centralized Training, Decentralized Execution (CTDE)](#centralized-training-decentralized-execution-ctde)
   - [Handling Discrete Actions with Gumbel-Softmax](#handling-discrete-actions-with-gumbel-softmax)
2. [Environment Design](#environment-design)
   - [Grid Layout](#grid-layout)
   - [Observation Space](#observation-space)
   - [Action Space](#action-space)
   - [Reward Structure](#reward-structure)
   - [Episode Termination Conditions](#episode-termination-conditions)
3. [Architecture Overview](#architecture-overview)
4. [File-by-File Breakdown](#file-by-file-breakdown)
5. [Setup & Installation](#setup--installation)
6. [Training](#training)
   - [All CLI Arguments](#all-cli-arguments)
   - [Training Loop Details](#training-loop-details)
   - [Saved Outputs](#saved-outputs)
7. [Visualization](#visualization)
8. [Running the Full Pipeline (Bash)](#running-the-full-pipeline-bash)
9. [Independent DDPG Baseline](#independent-ddpg-baseline)
10. [Key Design Decisions & Gotchas](#key-design-decisions--gotchas)

---

## Conceptual Background

### Why MADDPG?

In a **multi-agent setting**, standard single-agent RL (e.g., vanilla DDPG, DQN) breaks the i.i.d. (independent and identically distributed) assumption that underpins most convergence guarantees. The environment appears **non-stationary** from the perspective of any single agent, because the other agents' policies are also changing during training. This makes the Q-value targets unstable.

**MADDPG** (Multi-Agent Deep Deterministic Policy Gradient, Lowe et al. 2017) solves this by adopting the **Centralized Training, Decentralized Execution (CTDE)** paradigm:

- During **training**, each agent's critic is given access to the *joint* observations and *joint* actions of **all agents** — making the value function well-conditioned on the full system state.
- During **execution** (inference), each agent only uses its **own local observations** to act — making it fully deployable in a decentralized way.

### Centralized Training, Decentralized Execution (CTDE)

The key equations:

**Critic Update (TD-learning with Bellman backup):**
```
y_i = r_i + γ * Q_i_target(o'_1,...,o'_N, a'_1,...,a'_N)  where a'_j = μ_j_target(o'_j)
L(θ_i) = E[(Q_i(o_1,...,o_N, a_1,...,a_N) - y_i)²]
```

**Actor Update (maximize Q):**
```
∇_θ_i J = E[∇_θ_i μ_i(o_i) * ∇_a_i Q_i(o_1,...,o_N, a_1,...,a_N)|_{a_i = μ_i(o_i)}]
```

Each agent `i` has its own actor (`μ_i`) and its own centralized critic (`Q_i`), which takes all agents' observations and actions as input. The joint obs/action inputs to the critic are **concatenated vectors**.

**Target Networks** for both actor and critic are maintained and updated via **Polyak (soft) averaging**:
```
θ_target ← τ * θ + (1 - τ) * θ_target
```
where `τ = 0.01` by default. This prevents the training targets from changing too rapidly.

### Handling Discrete Actions with Gumbel-Softmax

DDPG was originally designed for **continuous action spaces** (relying on the chain rule: `∂Q/∂a * ∂a/∂θ`). Pac-Man has a **discrete action space** (5 moves). To bridge this gap, this implementation uses the **Gumbel-Softmax estimator** (Jang et al. 2017, Maddison et al. 2017):

1. The actor outputs **logits** (raw un-normalized scores) for each action.
2. During training, Gumbel noise is sampled (`-log(-log(U))` where `U ~ Uniform(0,1)`) and added to the logits.
3. A **softmax** with temperature `T=1.0` is applied, yielding differentiable, soft action probabilities.
4. The straight-through trick (`hard=True`) produces a **one-hot** forward pass but uses the soft distribution for backpropagation — preserving differentiability.
5. The resulting one-hot/soft action vectors flow through the critic, allowing gradient computation (`∂Q/∂a` → `∂a/∂θ`).

During **evaluation**, the actor simply takes the **argmax** over logits (pure deterministic greedy), with no Gumbel noise.

---

## Environment Design

### Grid Layout

| Property | Value |
|---|---|
| Grid size | 20×20 (default, configurable) |
| Pac-Men agents | 2 (cooperative team) |
| Ghost agents | 2 (cooperative team) |
| Wall structure | Fixed border + ~40 random inner walls (seeded with `np.random.seed(42)` for reproducibility) |
| Food pellets | All remaining free cells (fully packed at start) |
| Power pellets | 2 placed at random empty cells |
| Pac-Man team lives | 3 (shared pool) |

The grid uses integer codes:
- `0` — Empty cell
- `1` — Wall (impassable)
- `2` — Food pellet
- `3` — Power pellet

> **Note on wall seeding:** Walls are always placed using `np.random.seed(42)`, then the seed is reset (`np.random.seed()`). This means the static wall layout is **fixed across all episodes**, but food/agent spawns remain random.

### Observation Space

Each agent receives its own **local observation vector** of size:
```
obs_dim = 6 * grid_size² + 1
```
For a 20×20 grid: `6 * 400 + 1 = 2401` floats.

The observation is constructed from **6 binary spatial channels**, each flattened from a `(grid_size, grid_size)` grid:

| Channel | Contents |
|---|---|
| Channel 0 | Wall positions (`grid == 1`) |
| Channel 1 | Food pellet positions (`grid == 2`) |
| Channel 2 | Power pellet positions (`grid == 3`) |
| Channel 3 | All Pac-Men positions (alive only) |
| Channel 4 | All Ghost positions |
| Channel 5 | **Self** position (this agent only) |

Plus **1 scalar** at the end: `1.0` if power mode is currently active, `0.0` otherwise.

All channels are shared across agents (global map info), but Channel 5 (self) differs per agent. This design gives every agent full global visibility of the grid.

### Action Space

Each agent has **5 discrete actions**:

| Index | Action |
|---|---|
| 0 | Move Up (row - 1) |
| 1 | Move Down (row + 1) |
| 2 | Move Left (col - 1) |
| 3 | Move Right (col + 1) |
| 4 | Stay (no movement) |

Wall collision is handled by `_move()`: if the target cell is a wall or out-of-bounds, the position remains unchanged.

### Reward Structure

| Event | Reward |
|---|---|
| Pac-Man eats food | `+10` |
| Pac-Man eats power pellet | `+30` |
| Pac-Man eats a ghost (power mode active) | `+20` (Pac-Man), `-20` (Ghost) |
| Ghost catches Pac-Man (normal mode) | `-20` (Pac-Man), `+20` (Ghost) |
| Any agent moves successfully | `+0.5` |
| Any agent hits a wall / stays still | `-2.0` |
| Ghost moves closer to nearest Pac-Man | `+0.1` |

The **positive movement reward** (`+0.5`) and **wall/still penalty** (`-2.0`) are dense shaping signals designed to prevent agents from getting stuck in place — a common failure mode without such heuristics.

The **ghost proximity reward** (`+0.1`) is computed by comparing the Manhattan distance to the nearest alive Pac-Man before and after each ghost move. This shapes ghosts to actively hunt.

### Episode Termination Conditions

An episode ends (`done = True` for all agents) when **any** of the following occurs:
1. `step_count >= max_steps` (default: 200)
2. All Pac-Men have been killed (team lives depleted to 0)
3. No food or power pellets remain on the grid (Pac-Men cleared the board)

After ghost catches (normal mode), a Pac-Man respawns at a random free cell if the team still has lives remaining. After power-mode catches, the ghost respawns at a random free cell.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        MADDPG System                            │
│                                                                 │
│   ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐  │
│   │  Agent 0   │   │  Agent 1   │   │  Agent 2   │   │  Agent 3   │  │
│   │ (Pac-Man)  │   │ (Pac-Man)  │   │  (Ghost)   │   │  (Ghost)   │  │
│   │            │   │            │   │            │   │            │  │
│   │  Actor     │   │  Actor     │   │  Actor     │   │  Actor     │  │
│   │  [obs_i]   │   │  [obs_i]   │   │  [obs_i]   │   │  [obs_i]   │  │
│   │  ↓ action_i│   │  ↓ action_i│   │  ↓ action_i│   │  ↓ action_i│  │
│   │            │   │            │   │            │   │            │  │
│   │  Critic    │   │  Critic    │   │  Critic    │   │  Critic    │  │
│   │  [ALL obs] │   │  [ALL obs] │   │  [ALL obs] │   │  [ALL obs] │  │
│   │  [ALL acts]│   │  [ALL acts]│   │  [ALL acts]│   │  [ALL acts]│  │
│   └────────────┘   └────────────┘   └────────────┘   └────────────┘  │
│                                                                 │
│                  ┌──────────────────────┐                       │
│                  │   Replay Buffer       │                       │
│                  │  capacity: 20,000    │                       │
│                  │  per-agent storage   │                       │
│                  └──────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

Each agent has:
- **Actor**: `MLPActor(obs_dim → 256 → 256 → action_dim)` — takes *own* obs, outputs Gumbel-Softmax action.
- **Critic**: `MLPCritic(joint_obs_dim + joint_act_dim → 256 → 256 → 1)` — takes *all* agents' obs + acts, outputs Q-value.
- **Target Actor + Target Critic**: frozen copies updated via soft update.

---

## File-by-File Breakdown

### `env.py` — Custom Pac-Man Environment

**Class:** `PacManEnv`

| Method | Description |
|---|---|
| `__init__(grid_size, n_pacmen, n_ghosts, max_steps, power_duration)` | Initializes all dimensions, calls `reset()`. |
| `reset()` | Resets step count, power timer, team lives. Builds walls (seeded), places 2 power pellets, spawns agents, fills remaining cells with food. Returns initial observations. |
| `_get_obs()` | Builds 6-channel spatial observation for each agent and concatenates into flat vectors. Appends power-mode status scalar. |
| `_move(pos, action)` | Attempts to move position by action delta. Returns original position if wall or out-of-bounds collision. |
| `step(actions)` | Advances environment by one step: moves Pac-Men (collecting food/pellets), moves Ghosts (with proximity reward), resolves collisions (power/normal mode), decrements power timer, checks termination. Returns `(obs, rewards, dones, {})`. |
| `get_eval_state()` | Returns a clean dict snapshot of the environment state for the visualizer: `{grid, pacmen_pos, pacmen_alive, ghosts_pos, power_mode}`. |

---

### `networks.py` — Neural Network Definitions

**Function:** `gumbel_softmax(logits, temperature=1.0, hard=False)`
- Adds Gumbel noise to logits: `(logits + Gumbel) / temperature`
- Applies softmax to produce continuous, differentiable action probabilities.
- If `hard=True`: produces a one-hot vector in the forward pass using the **straight-through estimator** — gradients backpropagate through the soft version.

**Class:** `MLPActor(obs_dim, action_dim, hidden_dim=256)`
- 3-layer MLP: `obs_dim → 256 → 256 → action_dim`
- Activation: `ReLU` on hidden layers, raw logits as output.
- In training (`deterministic=False`): calls `gumbel_softmax()`.
- In evaluation (`deterministic=True`): returns a one-hot argmax vector.

**Class:** `MLPCritic(joint_obs_dim, joint_act_dim, hidden_dim=256)`
- Input dimension: `joint_obs_dim + joint_act_dim` (all agents' observations and actions concatenated).
- 3-layer MLP: `input_dim → 256 → 256 → 1`
- Outputs a single scalar Q-value.

---

### `maddpg.py` — MADDPG Algorithm

**Function:** `soft_update(target, source, tau)`
- Performs Polyak averaging: `θ_target ← τ*θ + (1-τ)*θ_target`

**Class:** `SingleAgent`
- Bundles: `actor`, `critic`, `target_actor`, `target_critic`, `actor_optimizer`, `critic_optimizer`.
- Target networks are initialized as **deep copies** of the originals.
- Both actor and critic use `Adam` optimizer with `lr=1e-3` (default).

**Class:** `MADDPG`
- `__init__`: Creates `n_agents` `SingleAgent` instances. In MADDPG mode, each critic receives `sum(obs_dims)` and `sum(action_dims)` as joint inputs. In independent mode, it only sees its own obs/act.
- `get_actions(obs_list, explore)`: Runs each agent's actor, returns list of numpy one-hot action arrays. `explore=True` uses Gumbel-Softmax; `explore=False` uses deterministic argmax.
- `update(sample, agent_idx)`:
  1. **Critic update**: Computes Bellman target using target networks (`y = r + γ * Q_target`), then minimizes MSE loss. Gradient norm clipping at `0.5`.
  2. **Actor update**: Computes current policy actions; evaluates joint Q-value (other agents' actions taken from replay buffer and detached). Maximizes Q by minimizing `-Q.mean()`. Gradient norm clipping at `0.5`.
  3. **Soft update**: Updates target actor and target critic via Polyak averaging.

---

### `replay_buffer.py` — Experience Replay

**Class:** `MultiAgentReplayBuffer(max_size, n_agents, obs_dims, action_dims)`

Stores transitions for all agents in separate pre-allocated NumPy arrays:
- `obs_buffers[i]` — shape `(max_size, obs_dim_i)`
- `action_buffers[i]` — shape `(max_size, action_dim_i)` (one-hot vectors)
- `reward_buffers[i]` — shape `(max_size, 1)`
- `next_obs_buffers[i]` — shape `(max_size, obs_dim_i)`
- `done_buffers[i]` — shape `(max_size, 1)`

Uses a **circular pointer** (`ptr`) that wraps at `max_size` — oldest transitions are overwritten when full.

| Method | Description |
|---|---|
| `add(obs, actions, rewards, next_obs, dones)` | Writes one transition for all agents at the current pointer; advances pointer. |
| `sample(batch_size)` | Samples `batch_size` random indices; returns lists of batched arrays, one per agent. |
| `__len__()` | Returns current buffer fill count. |

---

### `train.py` — Training Script

**Flow:**
1. Parses CLI arguments.
2. Instantiates `PacManEnv`, `MADDPG`, and `MultiAgentReplayBuffer`.
3. Runs `args.episodes` training episodes:
   - Each step: selects actions (Gumbel-Softmax) + **10% ε-greedy random exploration** override.
   - Stores transitions in the replay buffer.
   - Every **10 steps**, updates each agent's networks (if buffer has ≥ `batch_size` samples).
4. Every **100 episodes**: logs average reward and Pac-Man win rate to console.
5. Every **1000 episodes** (and at the end): saves model weights to `models/` and plots to `plots/`.

**Exploration strategy:** Hybrid — Gumbel-Softmax stochasticity (soft) + 10% ε-greedy uniform random override. When ε-greedy is triggered, the one-hot action stored in the replay buffer is overridden to match the actual (random) action taken.

---

### `visualize.py` — GIF Renderer

**Flow:**
1. Loads saved actor weights from `models/agent_{i}_actor.pth`.
2. Runs a full deterministic episode (with 10% random wander), capturing `get_eval_state()` snapshots every step.
3. Renders each frame using `matplotlib` on a dark background with scatter plots.
4. Saves a timestamped GIF to `videos/demo_{timestamp}.gif` using Pillow writer.

**Visual legend:**

| Symbol | Represents |
|---|---|
| 🟡 Yellow circle | Pac-Man |
| 🔴 Red triangle | Ghost (normal mode) |
| 🔵 Cyan triangle | Ghost (power mode — vulnerable!) |
| ⬜ White dot | Food pellet |
| 🟢 Green/white dot | Power pellet |
| 🟦 Dark teal square | Static wall |

The title bar shows current step and `POWER MODE: ACTIVE` / `NORMAL` status, color-coded green/white.

---

### `run.sh` — Full Pipeline Script (Linux/macOS/WSL)

Runs training then visualization sequentially. Supports the same configurable flags via bash argument parsing.

```bash
./run.sh [--episodes N] [--batch_size N] [--grid_size N] [--power_duration N] [--independent]
```

If training fails (non-zero exit code), the script halts before attempting visualization.

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional but recommended for faster training)

### Install Dependencies

```bash
pip install torch numpy matplotlib pillow
```

> `pillow` is required by `matplotlib`'s `animation.save()` with the Pillow GIF writer.

### Clone & Run

```bash
git clone <repo-url>
cd MADDPG
pip install torch numpy matplotlib pillow
python train.py
```

---

## Training

### Quick Start

```bash
# Default: 10,000 episodes, 20x20 grid, MADDPG
python train.py
```

### All CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--episodes` | `10000` | Total number of training episodes |
| `--batch_size` | `256` | Mini-batch size sampled from replay buffer per update |
| `--buffer_size` | `20000` | Maximum replay buffer capacity (circular) |
| `--lr` | `0.001` | Learning rate for all Adam optimizers (actor + critic) |
| `--gamma` | `0.95` | Discount factor γ for Bellman backup |
| `--tau` | `0.01` | Soft update rate for target networks |
| `--grid_size` | `20` | Grid dimension (NxN) |
| `--max_steps` | `200` | Maximum steps per episode before forced termination |
| `--power_duration` | `10` | Number of steps Power Mode remains active after pellet eaten |
| `--independent` | Flag (off) | Switch to Independent DDPG (no centralized critic) |

### Example Configurations

```bash
# Longer training with larger batch for better sample efficiency
python train.py --episodes 20000 --batch_size 1024

# Smaller, faster-to-train grid
python train.py --grid_size 10 --max_steps 150 --episodes 5000

# Power mode that lasts longer (easier for Pac-Men)
python train.py --power_duration 25

# Run Independent DDPG baseline
python train.py --independent

# Combined custom run
python train.py --grid_size 15 --episodes 30000 --batch_size 512 --gamma 0.99 --tau 0.005
```

### Training Loop Details

- Networks are updated **every 10 environment steps** (not after every step) to reduce training instability from highly correlated consecutive samples.
- All 4 agents are updated **sequentially** per update cycle (each samples independently from the shared buffer).
- Gradient clipping at **norm 0.5** is applied to both actor and critic to prevent explosive gradients in the early training phase when Q-value estimates are far from ground truth.
- The **10% pure random exploration** rate (`ε = 0.1`) is fixed throughout training — there is no annealing schedule. This helps maintain exploration even in later episodes.

### Saved Outputs

| Path | Contents |
|---|---|
| `models/agent_{i}_actor.pth` | Saved actor weights for agent `i` (i=0..3) |
| `models/agent_{i}_critic.pth` | Saved critic weights for agent `i` |
| `plots/reward_curve.png` | Total episodic reward + 100-episode rolling average |
| `plots/win_rate.png` | Pac-Man rolling win rate (100-episode window) |

Models and plots are saved every **1000 episodes** and at the final episode.

**Win condition heuristic**: Pac-Man team "wins" an episode if their cumulative reward exceeds the Ghost team's cumulative reward (`team_pacman_reward > team_ghost_reward`).

---

## Visualization

Requires trained model weights in the `models/` directory.

```bash
python visualize.py
```

To match the training environment configuration:
```bash
python visualize.py --grid_size 20 --power_duration 10 --max_steps 200
```

Output: `videos/demo_YYYYMMDD_HHMMSS.gif`

The episode is run **deterministically** (no Gumbel noise, `explore=False`), with a 10% random wander mixed in to prevent agents from looping in place during evaluation.

---

## Running the Full Pipeline (Bash)

For Linux, macOS, or Windows WSL:

```bash
chmod +x run.sh
./run.sh --episodes 10000 --grid_size 20
```

For Independent DDPG baseline pipeline:
```bash
./run.sh --episodes 10000 --independent
```

The script:
1. Prints a config summary header
2. Runs `train.py` with the given arguments
3. If training succeeds, automatically runs `visualize.py`
4. Outputs path to the generated GIF

---

## Independent DDPG Baseline

When `--independent` is passed, the system switches from **centralized critics** (MADDPG) to **decentralized critics** (Independent DDPG):

| Property | MADDPG | Independent DDPG |
|---|---|---|
| Critic input (obs) | All agents' observations concatenated | Own observation only |
| Critic input (act) | All agents' actions concatenated | Own action only |
| Joint obs dim | `n_agents × obs_dim` | `obs_dim` |
| Joint act dim | `n_agents × action_dim` | `action_dim` |

All other aspects (actor architecture, Gumbel-Softmax, replay buffer, update cycle) remain identical. The `independent_ddpg` flag propagates through `MADDPG.__init__()` → `SingleAgent` construction → `MADDPG.update()`.

This baseline is useful for directly quantifying the benefit of centralized training on this specific multi-agent task.

---

## Key Design Decisions & Gotchas

### Fixed Wall Seeding
Walls are seeded with `np.random.seed(42)` and then reset. The static layout is **identical across all episodes**. Agents do not need to re-learn wall positions. Food and agent spawns are still random each episode.

### Full Global Observability
All 6 channels are built from the **full global grid** — every agent can see the entire board. Channel 5 (`c_self`) is the only per-agent distinction. This is a simplification that makes learning easier but is not a realistic partial-observability setup.

### Dead Pac-Man Handling
When a Pac-Man runs out of team lives, all Pac-Men are marked `alive=False` and their positions are set to `[-1, -1]`. The `_get_obs()` and `get_eval_state()` code handles dead Pac-Men by skipping their position when building the `c_pacmen` and `c_self` channels.

### One-Hot Action Storage
Actions are stored in the replay buffer as **one-hot float32 vectors** (not raw integers). This is required because the critic receives these vectors as continuous inputs. The ε-greedy override ensures the stored one-hot matches the action actually taken in the environment step.

### No Learning Rate Annealing / Entropy Bonus
The current implementation uses a fixed learning rate and no entropy regularization. Adding LR scheduling (e.g., cosine decay) or entropy bonuses could improve convergence — especially for the ghost agents, which have a harder learning signal.

### Network Update Frequency
Updates happen every **10 environment steps** (controlled by `steps % 10 == 0` in `train.py`). This reduces correlation between successive updates. Decoupling environment interaction from learning updates is a standard practice in off-policy RL.

### Gradient Clipping
Both actor and critic gradients are clipped at `max_norm = 0.5`. This is important early in training when Q-value estimates are highly inaccurate, causing large, noisy gradients that could destabilize the actor.

---

## Project Structure

```
MADDPG/
├── env.py              # Custom 20x20 Pac-Man grid environment
├── networks.py         # MLPActor, MLPCritic, Gumbel-Softmax
├── maddpg.py           # MADDPG / Independent DDPG algorithm
├── replay_buffer.py    # Multi-agent circular replay buffer
├── train.py            # Training loop + CLI + plotting
├── visualize.py        # GIF renderer from saved model weights
├── run.sh              # Full pipeline bash script
├── models/             # Saved actor/critic weights (generated)
├── plots/              # Reward & win-rate curves (generated)
└── videos/             # Timestamped demo GIFs (generated)
```
