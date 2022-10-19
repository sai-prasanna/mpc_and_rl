from math import inf
import nntplib
from dreamerv2_torch.policy.policy import Policy
from dreamerv2_torch.world_model import WorldModel
import torch
from torch import jit
# https://github.com/Kaixhin/PlaNet/blob/master/planner.py


# Model-predictive control planner with cross-entropy method and learned transition model
class CrossEntropyMethodMPC(Policy):
  
    def __init__(self, config, act_space, planning_horizon, optimisation_iters, candidates, top_candidates, world_model):
        super().__init__()
        self.action_size = int(act_space.shape[0])
        self.min_action = torch.tensor(act_space.low, device=config.device)
        self.max_action = torch.tensor(act_space.high, device=config.device)
        self.planning_horizon = planning_horizon
        self.optimisation_iters = optimisation_iters
        self.candidates, self.top_candidates = candidates, top_candidates
        self.world_model = world_model

    def policy(self, state, sample: bool):
        with torch.no_grad():
            state = state.copy()
            for k, v in state.items():
                B, Z = v.size(0), v.size(1)
                state[k] = v.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
            # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
            action_mean, action_std_dev = torch.zeros(self.planning_horizon, B, 1, self.action_size, device=state['deter'].device), torch.ones(self.planning_horizon, B, 1, self.action_size, device=state['deter'].device)
            for _ in range(self.optimisation_iters):
                    # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
                    actions = (action_mean + action_std_dev * torch.randn(self.planning_horizon, B, self.candidates, self.action_size, device=action_mean.device)).view(self.planning_horizon, B * self.candidates, self.action_size)  # Sample actions (time x (batch x candidates) x actions)
                    actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range
                    # Sample next states
                    returns = []
                    for action in actions:
                        state = self.world_model.rssm.img_step(state, action, sample)
                        feat = self.world_model.rssm.get_feat(state)
                        returns.append(self.world_model.heads["reward"](feat).mode())
                    # Calculate expected returns (technically sum of rewards over planning horizon)
                    returns = torch.stack(returns).view(self.planning_horizon, -1).sum(dim=0)
                    # Re-fit belief to the K best action sequences
                    _, topk = returns.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False)
                    topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)  # Fix indices for unrolled actions
                    best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates, self.action_size)
                    # Update belief with new means and standard deviations
                    action_mean, action_std_dev = best_actions.mean(dim=2, keepdim=True), best_actions.std(dim=2, unbiased=False, keepdim=True)
            # Return first action mean µ_t
            top_action = action_mean[0].squeeze(dim=1)
            action_std_dev = action_std_dev[0].squeeze(dim=1)
            if sample:
                top_action = (top_action + action_std_dev * torch.randn(action_std_dev.shape, device=action_std_dev.device)).clamp(min=self.min_action, max=self.max_action)
            return top_action


# # Model-predictive control planner with cross-entropy method and learned transition model
# class CrossEntropyMethodMPC(jit.ScriptModule, Policy):
#   __constants__ = ['action_size', 'planning_horizon', 'optimisation_iters', 'candidates', 'top_candidates', 'min_action', 'max_action']

#   def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model, reward_model, min_action=-inf, max_action=inf):
#     super().__init__()
#     self.transition_model, self.reward_model = transition_model, reward_model
#     self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
#     self.planning_horizon = planning_horizon
#     self.optimisation_iters = optimisation_iters
#     self.candidates, self.top_candidates = candidates, top_candidates

#   @jit.script_method
#   def forward(self, state):
#     B, Z = state.size(0), state.size(1)
#     state = state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
#     # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
#     action_mean, action_std_dev = torch.zeros(self.planning_horizon, B, 1, self.action_size, device=state.device), torch.ones(self.planning_horizon, B, 1, self.action_size, device=state.device)
#     for _ in range(self.optimisation_iters):
#       # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
#       actions = (action_mean + action_std_dev * torch.randn(self.planning_horizon, B, self.candidates, self.action_size, device=action_mean.device)).view(self.planning_horizon, B * self.candidates, self.action_size)  # Sample actions (time x (batch x candidates) x actions)
#       actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range
#       # Sample next states
#       beliefs, states, _, _ = self.transition_model(state, actions)
#       # Calculate expected returns (technically sum of rewards over planning horizon)
#       returns = self.reward_model(states.view(-1, Z)).view(self.planning_horizon, -1).sum(dim=0)
#       # Re-fit belief to the K best action sequences
#       _, topk = returns.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False)
#       topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)  # Fix indices for unrolled actions
#       best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates, self.action_size)
#       # Update belief with new means and standard deviations
#       action_mean, action_std_dev = best_actions.mean(dim=2, keepdim=True), best_actions.std(dim=2, unbiased=False, keepdim=True)
#     # Return first action mean µ_t
#     return action_mean[0].squeeze(dim=1)

