# Multi-Agent Deep Deterministic Policy Gradient (MADDPG) for Pac-Man

This repository contains a full PyTorch implementation of MADDPG applied to a cooperative/competitive 20x20 Grid Pac-Man environment with distinct roles, custom rewards, and Power Pellet mechanics.

## Features
- **Custom Environment:** 20x20 Grid holding 2 cooperative Pac-men and 2 cooperative Ghosts. 
- **Power Pellets:** Eating one turns both Pac-men into hunters for `power_duration` steps.
- **MADDPG:** Implemented from scratch using a continuous approximation representation (Gumbel-Softmax estimator) allowing DDPG policies on discrete actions.
- **Experiment Toggles:** Capable of running Independent DDPG instead of MADDPG for simple baseline comparisons.

## Requirements
```bash
pip install torch numpy matplotlib
```

## How to Train
Run `train.py` to begin training over the default 10k episodes.
Models will periodically save to the `models/` folder and plot learning curves to `plots/`.

```bash
python train.py
```

### Advanced Config
```bash
# Run baseline Independent DDPG
python train.py --independent

# Play around with environment dimensions and hyperparams
python train.py --grid_size 15 --power_duration 15 --episodes 20000 --batch_size 1024
```

## How to Visualize
Once training has generated `models/` checkpoints, you can parse a deterministic demonstration video using:

```bash
python visualize.py
```
This generates a smooth evaluation grid step-through and outputs it directly as `videos/demo.gif`.

### Understanding the Visualization
- **Yellow Dots:** Pac-Man
- **Red Dots:** Ghosts (Normal Mode)
- **Blue Dots:** Ghosts (Vulnerable / Power-Mode Active)
- **White Dots:** Food
- **Green Dots:** Power Pellets
- **Dark Gray Blocks:** Static Walls
