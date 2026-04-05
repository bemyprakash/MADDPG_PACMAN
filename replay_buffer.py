import numpy as np

class MultiAgentReplayBuffer:
    def __init__(self, max_size, n_agents, obs_dims, action_dims):
        self.max_size = int(max_size)
        self.n_agents = n_agents
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.ptr = 0
        self.size = 0
        
        # Storing data for each agent
        self.obs_buffers = [np.zeros((self.max_size, obs_dims[i]), dtype=np.float32) for i in range(n_agents)]
        self.action_buffers = [np.zeros((self.max_size, action_dims[i]), dtype=np.float32) for i in range(n_agents)]
        self.reward_buffers = [np.zeros((self.max_size, 1), dtype=np.float32) for i in range(n_agents)]
        self.next_obs_buffers = [np.zeros((self.max_size, obs_dims[i]), dtype=np.float32) for i in range(n_agents)]
        self.done_buffers = [np.zeros((self.max_size, 1), dtype=np.float32) for i in range(n_agents)]
        
    def add(self, obs, actions, rewards, next_obs, dones):
        for i in range(self.n_agents):
            self.obs_buffers[i][self.ptr] = obs[i]
            # actions[i] might be integer or one-hot. We assume they are one-hot/probabilities.
            self.action_buffers[i][self.ptr] = actions[i]
            self.reward_buffers[i][self.ptr] = rewards[i]
            self.next_obs_buffers[i][self.ptr] = next_obs[i]
            self.done_buffers[i][self.ptr] = dones[i]
            
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        obs_batch = [self.obs_buffers[i][idxs] for i in range(self.n_agents)]
        action_batch = [self.action_buffers[i][idxs] for i in range(self.n_agents)]
        reward_batch = [self.reward_buffers[i][idxs] for i in range(self.n_agents)]
        next_obs_batch = [self.next_obs_buffers[i][idxs] for i in range(self.n_agents)]
        done_batch = [self.done_buffers[i][idxs] for i in range(self.n_agents)]
        
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def __len__(self):
        return self.size
