from math import inf
import nntplib
from typing import Dict
from dreamerv2_torch.policy.policy import Policy
from dreamerv2_torch.world_model import WorldModel
import numpy as np
import torch
from copy import deepcopy
from torch import Tensor, jit
# https://github.com/Kaixhin/PlaNet/blob/master/planner.py
from .cem_gd_optimizer import CEMGDOptimizer, TrajectoryOptimizer

# Model-predictive control planner with cross-entropy method and learned transition model
class GradientMPC(Policy):
  
    def __init__(self, config, act_space, world_model, planning_horizon):
        super().__init__()

        self.traj_optimizer = TrajectoryOptimizer(
            device=config.device,
            optimizer=CEMGDOptimizer(
                num_iterations=1, # This is only CEM iterations
                elite_ratio=1,
                population_size=1,
                alpha=0.1,
                num_top=1,
                resample_amount=1,
                device=config.device,
                lower_bound=np.tile(act_space.low, (planning_horizon, 1)).tolist(),
                upper_bound=np.tile(act_space.high, (planning_horizon, 1)).tolist()),
            action_lb=act_space.low,
            action_ub=act_space.high,
            planning_horizon=planning_horizon,
            resample=False,
        )
        self.world_model = world_model
    
    def policy(self, state: Dict[str, Tensor], sample: bool):
        def cost_fun(trajectories, grad):
            nonlocal state
            current_state = state.copy()
            for k, v in current_state.items():
                B, Z = v.size(0), v.size(1)
                assert B == 1
                current_state[k] = v.unsqueeze(dim=1).expand(B, trajectories.size(0), Z).reshape(-1, Z)
            returns = []
            actions = trajectories.transpose(0, 1)
            for action in actions:
                current_state = self.world_model.rssm.img_step(current_state, action, False)
                feat = self.world_model.rssm.get_feat(current_state)
                returns.append(self.world_model.heads["reward"](feat).mode())
            # Calculate expected returns (technically sum of rewards over planning horizon)
            values = torch.stack(returns).view(trajectories.size(1), -1).sum(dim=0)
            return values
        return self.traj_optimizer.optimize(None, cost_fun, use_opt=True, use_cem=False)[0].unsqueeze(0)

    def reset(self):
        self.traj_optimizer.reset()

# python3 dreamerv2_torch/train.py --logdir logs/cartpole_gd/1337 --configs dmc_vision --task dmc_cartpole_swingup --task_behavior gd --seed 1337 --steps 2e5