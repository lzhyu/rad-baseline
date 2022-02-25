from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym.spaces import Box
import gym
import torch
from torch import nn
import numpy as np

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class SACEncoder(BaseFeaturesExtractor):
    """Convolutional encoder of pixels observations."""
    def __init__(self,  observation_space: gym.spaces.Box, \
     features_dim=50, obs_shape=(9,84,84), num_layers=4, num_filters=32):
        super().__init__(observation_space, features_dim)

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        # self.features_dim = features_dim
        self.num_layers = num_layers
        # try 2 5x5s with strides 2x2. with samep adding, it should reduce 84 to 21, so with valid, it should be even smaller than 21.
        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        print(observation_space)
        with torch.no_grad():
            n_flatten = self.forward_conv(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
        self.fc = nn.Linear(n_flatten, features_dim)
        self.ln = nn.LayerNorm(features_dim)

        #self.apply(weight_init)


    def forward_conv(self, obs):
        if obs.max() > 1.:
            obs = obs / 255.

        conv = torch.relu(self.convs[0](obs))

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()
        h_fc = self.fc(h)
        out = self.ln(h_fc)
        
        return out