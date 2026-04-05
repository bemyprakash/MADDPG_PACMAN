import numpy as np
import torch
import os
import argparse
import matplotlib.pyplot as plt
from env import PacManEnv
from maddpg import MADDPG
from replay_buffer import MultiAgentReplayBuffer

def parse_args():
    parser = argparse.ArgumentParser("MADDPG for Pac-Man")
    parser.add_argument("--episodes", type=int, default=10000, help="number of episodes")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--buffer_size", type=int, default=20000, help="replay buffer capacity")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="soft update rate")
    parser.add_argument("--grid_size", type=int, default=20, help="grid size")
    parser.add_argument("--max_steps", type=int, default=200, help="max steps per episode")
    parser.add_argument("--power_duration", type=int, default=10, help="power mode duration")
    parser.add_argument("--independent", action="store_true", help="use independent DDPG toggle")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Init Env
    env = PacManEnv(grid_size=args.grid_size, max_steps=args.max_steps, power_duration=args.power_duration)
    
    # Define dims
    obs_dims = [env.obs_dim] * env.n_agents
    action_dims = [env.action_space_size] * env.n_agents
    
    # Init MADDPG
    maddpg = MADDPG(obs_dims, action_dims, n_agents=env.n_agents, 
                    gamma=args.gamma, tau=args.tau, independent_ddpg=args.independent)
    
    # Init Replay Buffer
    buffer = MultiAgentReplayBuffer(args.buffer_size, env.n_agents, obs_dims, action_dims)
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    reward_history = []
    pacman_wins = []
    
    print(f"Starting Training for {args.episodes} episodes...")
    
    for ep in range(args.episodes):
        obs = env.reset()
        episode_reward = np.zeros(env.n_agents)
        steps = 0
        
        while steps < args.max_steps:
            # Action selection
            with torch.no_grad():
                actions = maddpg.get_actions(obs, explore=True)
                
            # Discretize actions for env step and inject epsilon-greedy exploration
            discrete_actions = []
            for i in range(env.n_agents):
                if np.random.rand() < 0.1: # 10% pure random exploration
                    rand_act = np.random.randint(env.action_space_size)
                    discrete_actions.append(rand_act)
                    # Override the one-hot action buffer to match actual step taken
                    one_hot = np.zeros_like(actions[i])
                    one_hot[rand_act] = 1.0
                    actions[i] = one_hot
                else:
                    discrete_actions.append(np.argmax(actions[i]))
            
            # Step env
            next_obs, rewards, dones, info = env.step(discrete_actions)
            
            # Store in buffer
            buffer.add(obs, actions, rewards, next_obs, dones)
            
            obs = next_obs
            episode_reward += np.array(rewards)
            steps += 1
            
            # Update Networks
            if len(buffer) > args.batch_size and (steps % 10 == 0):
                for a_i in range(env.n_agents):
                    sample = buffer.sample(args.batch_size)
                    maddpg.update(sample, a_i)
                    
            if all(dones):
                break
                
        # Logging
        total_ep_reward = np.sum(episode_reward)
        reward_history.append(total_ep_reward)
        team_pacman_reward = np.sum(episode_reward[:env.n_pacmen])
        team_ghost_reward = np.sum(episode_reward[env.n_pacmen:])
        
        # Did Pac-Men "win"? Simple heuristic: pacmen scored positive, or survived to end.
        pacman_won = team_pacman_reward > team_ghost_reward
        pacman_wins.append(1 if pacman_won else 0)
        
        if (ep + 1) % 100 == 0:
            avg_rew = np.mean(reward_history[-100:])
            win_rate = np.mean(pacman_wins[-100:])
            print(f"Episode: {ep+1}/{args.episodes} | Avg Reward: {avg_rew:.2f} | P-Man Win%: {win_rate*100:.1f}%")
            
        if (ep + 1) % 1000 == 0 or ep + 1 == args.episodes:
            # Save weights
            for i, agent in enumerate(maddpg.agents):
                torch.save(agent.actor.state_dict(), f"models/agent_{i}_actor.pth")
                torch.save(agent.critic.state_dict(), f"models/agent_{i}_critic.pth")
            
            # Plot
            plt.figure(figsize=(10, 5))
            plt.plot(reward_history, alpha=0.6, label='Total Reward')
            plt.plot(np.convolve(reward_history, np.ones(100)/100, mode='valid'), label='100-ep Avg')
            plt.title('Total Episodic Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()
            plt.savefig("plots/reward_curve.png")
            plt.close()
            
            plt.figure(figsize=(10, 5))
            plt.plot(np.convolve(pacman_wins, np.ones(100)/100, mode='valid'))
            plt.title('Pac-Man Rolling Win Rate (100-ep)')
            plt.xlabel('Episode')
            plt.ylabel('Win Rate')
            plt.savefig("plots/win_rate.png")
            plt.close()

if __name__ == '__main__':
    main()
