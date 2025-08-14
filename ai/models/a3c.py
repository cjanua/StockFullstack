# ai/models/a3c.py
import torch.nn as nn
from torch.distributions import Categorical


class A3CTradingAgent(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()

        # Shared feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
            nn.Softmax(dim=-1)
        )

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, state):
        features = self.feature_extractor(state)
        policy = self.actor(features)
        value = self.critic(features)
        return policy, value

    def choose_action(self, state):
        policy, value = self.forward(state)
        dist = Categorical(policy)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value
