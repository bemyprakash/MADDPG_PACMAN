import torch
import torch.nn as nn
import torch.nn.functional as F

def gumbel_softmax(logits, temperature=1.0, hard=False):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    """
    gumbels = -torch.empty_like(logits).exponential_().log()
    y = (logits + gumbels) / temperature
    y = F.softmax(y, dim=-1)
    
    if hard:
        # Straight-through estimator
        index = y.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        y = (y_hard - y).detach() + y
    return y

class MLPActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(MLPActor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs, temperature=1.0, hard=False, deterministic=False):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        
        if deterministic:
            # For evaluation: return one-hot argmax
            index = logits.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            return y_hard
        else:
            # For training: use Gumbel-Softmax
            return gumbel_softmax(logits, temperature=temperature, hard=hard)


class MLPCritic(nn.Module):
    def __init__(self, joint_obs_dim, joint_act_dim, hidden_dim=256):
        super(MLPCritic, self).__init__()
        # Input to critic is concatenated observations and actions from all considered agents
        input_dim = joint_obs_dim + joint_act_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, joint_obs, joint_act):
        x = torch.cat([joint_obs, joint_act], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value
