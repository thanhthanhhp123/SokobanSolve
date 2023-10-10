import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd


class OnPolicy(nn.Module):
    def __init__(self):
        super(OnPolicy, self).__init__()
    
    def forward(self, x):
        raise NotImplementedError
    
    def act(self, x, deterministic=False):
        logit, value = self.forward(x)
        probs = F.softmax(logit, dim=1)

        if deterministic:
            action = probs.max(1)[1]
        else:
            action = probs.multinomial(num_samples=1)
        return action
    
    def evaluate_actions(self, x, action):
        logit, value = self.forward(x)
        probs = F.softmax(logit, dim=1)
        log_probs = F.log_softmax(logit, dim=1)

        action_log_probs = log_probs.gather(1, action)

        entropy = -(probs * log_probs).sum(1).mean()
        return logit, action_log_probs, value, entropy

class ActorCritic(OnPolicy):
    def __init__(self, in_shape, num_actions):
        super(ActorCritic, self).__init__()

        self.in_shape = in_shape
        self.in_channels = in_shape[0]
        self.out_channels = 16

        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 256),
        )

        self.critic = nn.Linear(256, 1)
        self.actor = nn.Linear(256, num_actions)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return self.actor(x), self.critic(x)
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.in_shape))).view(1, -1).size(1)
    # def feature_size(self):
    #     convoutput1 = self.calculate_conv_out(self.in_shape[1:3], self.out_channels, 3)       
    #     convoutput2 = self.calculate_conv_out(convoutput1[1:3], self.out_channels, 3, 2)
    #     features = int(np.prod(convoutput2))
    #     return features
    
    def calculate_conv_out(self, img_dim, out_channels, kernel_size, stride=1, padding=0):        
        output_width = (img_dim[0] - kernel_size + 2*padding) // stride + 1
        output_height = (img_dim[1] - kernel_size + 2*padding) // stride + 1
        return [out_channels, output_width, output_height]
    

class RolloutStorage(object):
    def __init__(self, num_steps, num_envs, state_shape):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.states = torch.zeros(num_steps + 1, num_envs, *state_shape)
        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.masks = torch.ones(num_steps + 1, num_envs, 1)
        self.actions = torch.zeros(num_steps, num_envs, 1).long()
        self.use_cuda = False
    
    def cuda(self):
        self.use_cuda = True
        self.states = self.states.cuda()
        self.rewards = self.rewards.cuda()
        self.masks = self.masks.cuda()
        self.actions = self.actions.cuda()
    
    def insert(self, step, state, action, reward, mask = None):
        self.states[step + 1].copy_(state)
        self.actions[step].copy_(action)
        self.rewards[step].copy_(reward.unsqueeze(1))
        # self.masks[step + 1].copy_(mask)
    
    def after_update(self):
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])
    
    def compute_returns(self, next_value, gamma):
        returns = torch.zeros(self.num_steps + 1, self.num_envs, 1)
        if self.use_cuda:
            returns = returns.cuda()
        returns[-1] = next_value
        for step in reversed(range(self.num_steps)):
            returns[step] = returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]
        return returns[:-1]