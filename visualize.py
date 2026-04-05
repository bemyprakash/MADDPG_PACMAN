import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from env import PacManEnv
from maddpg import MADDPG

device = torch.device('cpu')

def load_models(n_agents, obs_dims, action_dims):
    maddpg = MADDPG(obs_dims, action_dims, n_agents)
    for i, agent in enumerate(maddpg.agents):
        path = f"models/agent_{i}_actor.pth"
        if os.path.exists(path):
            agent.actor.load_state_dict(torch.load(path, map_location=device))
        else:
            print(f"Warning: {path} not found. Using random weights.")
    return maddpg

def run_episode(env, maddpg, max_steps=200):
    obs = env.reset()
    frames = []
    
    for _ in range(max_steps):
        frames.append(env.get_eval_state())
        
        # Get deterministic actions
        with torch.no_grad():
            actions = maddpg.get_actions(obs, explore=False)
            
        discrete_actions = []
        for i in range(env.n_agents):
            if np.random.rand() < 0.1: # 10% wander logic guarantees they don't get stuck!
                discrete_actions.append(np.random.randint(env.action_space_size))
            else:
                discrete_actions.append(np.argmax(actions[i]))
        obs, rewards, dones, info = env.step(discrete_actions)
        
        if all(dones):
            break
            
    frames.append(env.get_eval_state())
    return frames

def render_video(frames, args):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    
    def update(i):
        ax.clear()
        state = frames[i]
        grid = state['grid']
        power_mode = state['power_mode']
        
        ax.set_xlim(-0.5, args.grid_size - 0.5)
        ax.set_ylim((args.grid_size - 0.5, -0.5)) # Invert Y axis
        
        ax.set_xticks(np.arange(-0.5, args.grid_size, 1))
        ax.set_yticks(np.arange(-0.5, args.grid_size, 1))
        ax.grid(color='#222222', linestyle='-', linewidth=1)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        walls_x, walls_y = [], []
        food_x, food_y = [], []
        power_x, power_y = [], []
        
        for x in range(args.grid_size):
            for y in range(args.grid_size):
                if grid[x, y] == 1:
                    walls_x.append(y)
                    walls_y.append(x)
                elif grid[x, y] == 2:
                    food_x.append(y)
                    food_y.append(x)
                elif grid[x, y] == 3:
                    power_x.append(y)
                    power_y.append(x)
                    
        if walls_x:
            ax.scatter(walls_x, walls_y, s=350, marker='s', edgecolors='#0ff', facecolors='#002233', linewidths=2)
            
        if food_x:
            ax.scatter(food_x, food_y, s=30, color='#fff', zorder=2)
            
        if power_x:
            ax.scatter(power_x, power_y, s=400, color='#0f0', alpha=0.3, zorder=1)
            ax.scatter(power_x, power_y, s=150, color='#fff', zorder=2)
            
        ghost_x, ghost_y = [], []
        for gx, gy in state['ghosts_pos']:
            if gx >= 0 and gy >= 0:
                ghost_x.append(gy)
                ghost_y.append(gx)
                
        if ghost_x:
            ghost_color = '#0ff' if power_mode else '#f00'
            ax.scatter(ghost_x, ghost_y, s=500, marker='^', color=ghost_color, alpha=0.9, zorder=3)
            ax.scatter(ghost_x, ghost_y, s=100, marker='o', color='white', zorder=4)
            
        pac_x, pac_y = [], []
        for j, (px, py) in enumerate(state['pacmen_pos']):
            if state['pacmen_alive'][j] and px >= 0 and py >= 0:
                pac_x.append(py)
                pac_y.append(px)
                
        if pac_x:
            ax.scatter(pac_x, pac_y, s=500, color='#ff0', zorder=5)
            ax.scatter(pac_x, pac_y, s=50, color='black', marker='>', zorder=6)
            
        status_text = "POWER MODE: ACTIVE" if power_mode else "NORMAL"
        color_text = '#0f0' if power_mode else '#fff'
        ax.set_title(f"MADDPG PAC-MAN | Step {i} | {status_text}", color=color_text, fontsize=14, weight='bold', pad=15)
        
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=200)
    
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('videos', exist_ok=True)
    out_path = f'videos/demo_{timestamp}.gif'
    print(f"Saving beautiful animation to {out_path} ...")
    ani.save(out_path, writer='pillow')
    print("Done!")

def main():
    import argparse
    parser = argparse.ArgumentParser("Visualize MADDPG Pac-Man")
    parser.add_argument("--grid_size", type=int, default=20, help="grid size")
    parser.add_argument("--max_steps", type=int, default=200, help="max steps per episode")
    parser.add_argument("--power_duration", type=int, default=10, help="power mode duration")
    args = parser.parse_args()
    
    env = PacManEnv(grid_size=args.grid_size, max_steps=args.max_steps, power_duration=args.power_duration)
    
    obs_dims = [env.obs_dim] * env.n_agents
    action_dims = [env.action_space_size] * env.n_agents
    
    maddpg = load_models(env.n_agents, obs_dims, action_dims)
    frames = run_episode(env, maddpg, args.max_steps)
    print(f"Episode finished in {len(frames)} steps. Rendering...")
    render_video(frames, args)

if __name__ == '__main__':
    main()
