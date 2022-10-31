import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import pdb

class GradientOptimizerMPC(object):

    def __init__(self, config, act_space, planning_horizon, optimisation_iters, learning_rate, world_model):
        super().__init__()
        self.device = config.device
        self.action_size = int(act_space.shape[0])
        self.min_action = torch.tensor(act_space.low, device=config.device)
        self.max_action = torch.tensor(act_space.high, device=config.device)
        self.planning_horizon = planning_horizon
        self.optimisation_iters = optimisation_iters
        self.world_model = world_model
        self.lr = learning_rate

    def reset(self):
        pass

    def policy(self, state, sample: bool):

        with torch.no_grad():
            state = state.copy()
            for k, v in state.items():
                B, Z = v.size(0), v.size(1)
                state[k] = v.unsqueeze(dim=1).reshape(-1, Z)
        actions = torch.zeros(self.planning_horizon, B, self.action_size).to(self.device)
        actions.requires_grad = True
        optimizer = optim.Adam([actions], lr=self.lr)
        self.world_model.requires_grad_(False)
        for _ in range(self.optimisation_iters):
            current_state = state
            actions.requires_grad = False
            actions.clamp_(self.min_action, self.max_action)
            actions.requires_grad = True
            returns = []
            for i in range(len(actions)):
                current_state = self.world_model.rssm.img_step(current_state, actions[i], sample)
                feat = self.world_model.rssm.get_feat(current_state)
                returns.append(self.world_model.heads["reward"](feat).mode())
            returns = torch.stack(returns).view(self.planning_horizon, -1).sum()
            cost = -1*returns
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()
        actions = actions.detach()
        return actions[0]

# python3 dreamerv2_torch/train.py --logdir logs/cartpole_cem/1 --configs dmc_vision --task dmc_cartpole_swingup --task_behavior grad