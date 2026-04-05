import torch
import torch.nn.functional as F
from networks import MLPActor, MLPCritic
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class SingleAgent:
    def __init__(self, obj_dim, action_dim, joint_obs_dim, joint_act_dim, lr=1e-3):
        self.actor = MLPActor(obj_dim, action_dim).to(device)
        self.critic = MLPCritic(joint_obs_dim, joint_act_dim).to(device)
        
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

class MADDPG:
    def __init__(self, obs_dims, action_dims, n_agents, gamma=0.95, tau=0.01, independent_ddpg=False):
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.independent_ddpg = independent_ddpg
        
        self.agents = []
        for i in range(n_agents):
            if independent_ddpg:
                joint_obs_dim = obs_dims[i]
                joint_act_dim = action_dims[i]
            else:
                joint_obs_dim = sum(obs_dims)
                joint_act_dim = sum(action_dims)
                
            agent = SingleAgent(obs_dims[i], action_dims[i], joint_obs_dim, joint_act_dim)
            self.agents.append(agent)
            
    def get_actions(self, obs_list, explore=True):
        actions = []
        for i, agent in enumerate(self.agents):
            obs = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(device)
            if explore:
                # Add gumbel noise for exploration in training
                action = agent.actor(obs, temperature=1.0, hard=True, deterministic=False)
            else:
                # Deterministic argmax for evaluation
                action = agent.actor(obs, deterministic=True)
            actions.append(action.squeeze(0).cpu().detach().numpy())
        return actions

    def update(self, sample, agent_idx):
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = sample
        
        # Convert lists of batches to tensors
        obs = [torch.FloatTensor(o).to(device) for o in obs_batch]
        action = [torch.FloatTensor(a).to(device) for a in action_batch]
        reward = [torch.FloatTensor(r).to(device) for r in reward_batch]
        next_obs = [torch.FloatTensor(no).to(device) for no in next_obs_batch]
        done = [torch.FloatTensor(d).to(device) for d in done_batch]
        
        agent = self.agents[agent_idx]
        
        # 1. Update Critic
        agent.critic_optimizer.zero_grad()
        
        with torch.no_grad():
            if self.independent_ddpg:
                c_next_o = next_obs[agent_idx]
                c_next_a = agent.target_actor(next_obs[agent_idx], temperature=1.0, hard=True, deterministic=False)
            else:
                target_actions = []
                for i, a in enumerate(self.agents):
                    t_act = a.target_actor(next_obs[i], temperature=1.0, hard=True, deterministic=False)
                    target_actions.append(t_act)
                
                c_next_o = torch.cat(next_obs, dim=1)
                c_next_a = torch.cat(target_actions, dim=1)
                
            target_q = agent.target_critic(c_next_o, c_next_a)
            # Bellman backup
            y = reward[agent_idx] + self.gamma * target_q * (1 - done[agent_idx])
            
        if self.independent_ddpg:
            c_o = obs[agent_idx]
            c_a = action[agent_idx]
        else:
            c_o = torch.cat(obs, dim=1)
            c_a = torch.cat(action, dim=1)
            
        q = agent.critic(c_o, c_a)
        critic_loss = F.mse_loss(q, y)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()
        
        # 2. Update Actor
        agent.actor_optimizer.zero_grad()
        
        # We need gradients flowing from critic to actor.
        # So we use current actor to get actions for the central critic
        current_action = agent.actor(obs[agent_idx], temperature=1.0, hard=True, deterministic=False)
                
        if self.independent_ddpg:
            c_o_actor = obs[agent_idx]
            c_a_actor = current_action
        else:
            all_actions = []
            for i in range(self.n_agents):
                if i == agent_idx:
                    all_actions.append(current_action)
                else:
                    # Treat other agents' actions as fixed constants from replay buffer
                    all_actions.append(action[i].detach())
            c_o_actor = torch.cat(obs, dim=1)
            c_a_actor = torch.cat(all_actions, dim=1)
            
        actor_loss = -agent.critic(c_o_actor, c_a_actor).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
        agent.actor_optimizer.step()
        
        # 3. Soft updates
        soft_update(agent.target_actor, agent.actor, self.tau)
        soft_update(agent.target_critic, agent.critic, self.tau)
        
        return critic_loss.item(), actor_loss.item()
