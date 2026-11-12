import math
from itertools import pairwise

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def mlp(in_dim, out_dim, hidden, layer_norm=False):
    """MLP with ReLU hidden layers and optional LayerNorm before each ReLU"""
    sizes = [in_dim, *hidden, out_dim]
    layers = []
    for index, (n_in, n_out) in enumerate(pairwise(sizes)):
        layers.append(nn.Linear(n_in, n_out))
        if index < len(sizes) - 2:
            if layer_norm:
                layers.append(nn.LayerNorm(n_out))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class DeterministicPolicy(nn.Module):
    """BC and DAgger policy, tanh on the continuous outputs and logits on the binary ones"""

    def __init__(self, obs_dim, action_dim, hidden, n_continuous=None):
        super().__init__()
        self.obs_dim, self.action_dim, self.hidden = obs_dim, action_dim, tuple(hidden)
        self.n_continuous = action_dim if n_continuous is None else n_continuous
        self.model = mlp(obs_dim, action_dim, hidden)

    def forward(self, obs):
        out = self.model(obs)
        return torch.cat([torch.tanh(out[..., : self.n_continuous]), out[..., self.n_continuous :]], dim=-1)


class GaussianPolicy(nn.Module):
    """SAC actor, a Gaussian squashed by tanh"""

    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, obs_dim, action_dim, hidden):
        super().__init__()
        self.obs_dim, self.action_dim, self.hidden = obs_dim, action_dim, tuple(hidden)
        self.model = mlp(obs_dim, 2 * action_dim, hidden)

    def _mean_log_std(self, obs):
        mean, log_std = self.model(obs).chunk(2, dim=-1)
        return mean, log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)

    def sample(self, obs):
        """Reparameterized action and its log probability, one value per row"""
        mean, log_std = self._mean_log_std(obs)
        std = log_std.exp()
        pre_tanh = mean + std * torch.randn_like(mean)
        action = torch.tanh(pre_tanh)
        log_prob = (-0.5 * ((pre_tanh - mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        # Change of variables for tanh, written in the numerically stable form
        log_prob = log_prob - (2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))).sum(-1)
        return action, log_prob

    def deterministic(self, obs):
        mean, _ = self._mean_log_std(obs)
        return torch.tanh(mean)


class EnsembleLinear(nn.Module):
    """n independent linear layers applied in one batched matmul, [n, B, in] to [n, B, out]"""

    def __init__(self, n, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, in_dim, out_dim))
        self.bias = nn.Parameter(torch.empty(n, 1, out_dim))
        # Same initialization as nn.Linear for each member
        bound = 1.0 / math.sqrt(in_dim)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return torch.baddbmm(self.bias, x, self.weight)


class EnsembleLayerNorm(nn.Module):
    """LayerNorm with a separate affine transform per ensemble member"""

    def __init__(self, n, dim):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(n, 1, dim))
        self.bias = nn.Parameter(torch.zeros(n, 1, dim))

    def forward(self, x):
        return F.layer_norm(x, (self.dim,)) * self.weight + self.bias


class EnsembleCritic(nn.Module):
    """n Q networks evaluated in parallel, returns [n, batch]"""

    def __init__(self, obs_dim, action_dim, hidden, n_critics, layer_norm=True):
        super().__init__()
        self.n = n_critics
        layers, size = [], obs_dim + action_dim
        for width in hidden:
            layers.append(EnsembleLinear(n_critics, size, width))
            if layer_norm:
                layers.append(EnsembleLayerNorm(n_critics, width))
            layers.append(nn.ReLU())
            size = width
        layers.append(EnsembleLinear(n_critics, size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.model(x.unsqueeze(0).expand(self.n, *x.shape)).squeeze(-1)


@torch.no_grad()
def policy_action(policy, normalizer, obs):
    """Deterministic action of any trained policy for one raw feature vector"""
    x = normalizer(torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0))
    if isinstance(policy, GaussianPolicy):
        action = policy.deterministic(x)
    else:
        action = policy(x)
    return action[0].cpu().numpy()
