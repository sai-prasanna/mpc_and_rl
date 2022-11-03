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
class CrossEntropyMethodMPC(Policy):
  
    def __init__(self, config, act_space, world_model, num_iterations, planning_horizon, elite_ratio, population_size, alpha, num_top, resample_amount):
        super().__init__()

        self.traj_optimizer = TrajectoryOptimizer(
            device=config.device,
            optimizer=CEMGDOptimizer(
                num_iterations=num_iterations, 
                elite_ratio=elite_ratio,
                population_size=population_size, 
                alpha=alpha,
                num_top=num_top,
                resample_amount=resample_amount,
                device=config.device, 
                lower_bound=np.tile(act_space.low, (planning_horizon, 1)).tolist(),
                upper_bound=np.tile(act_space.high, (planning_horizon, 1)).tolist()),
            action_lb=act_space.low,
            action_ub=act_space.high,
            planning_horizon=planning_horizon,
            keep_last_solution=False,
            resample=False
        )
        self.world_model = world_model
    
    def policy(self, state: Dict[str, Tensor], sample: bool):
        def reward_fun(trajectories, sample):
            nonlocal state
            current_state = state.copy()
            for k, v in current_state.items():
                B, Z = v.size(0), v.size(1)
                assert B == 1
                current_state[k] = v.unsqueeze(dim=1).expand(B, trajectories.size(0), Z).reshape(-1, Z)
            returns = []
            actions = trajectories.transpose(0, 1)
            with torch.no_grad():
                for action in actions:
                    current_state = self.world_model.rssm.img_step(current_state, action, sample)
                    feat = self.world_model.rssm.get_feat(current_state)
                    returns.append(self.world_model.heads["reward"](feat).mode())
            # Calculate expected returns (technically sum of rewards over planning horizon)
                values = torch.stack(returns).view(trajectories.size(1), -1).sum(dim=0)
                return values  
        return self.traj_optimizer.optimize(reward_fun, None, use_opt=False, use_cem=True)[0].unsqueeze(0)

    def reset(self):
        self.traj_optimizer.reset()

# python3 dreamerv2_torch/train.py --logdir logs/cartpole_cem/1337 --configs dmc_vision --task dmc_cartpole_swingup --task_behavior cem --seed 1337 --steps 2e5