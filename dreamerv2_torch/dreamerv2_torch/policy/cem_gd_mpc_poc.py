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
from .adam_projected import Adam


# Model-predictive control planner with cross-entropy method and learned transition model
class CrossEntropyGDMPC(Policy):
  
    def __init__(self, config, act_space, planning_horizon, num_iterations, population_size, elite_ratio, world_model, num_top=3, use_opt=True, alpha=0.1, keep_last_solution=True, resample_amount=10, resample=False, return_mean_elites=False):
        super().__init__()

        assert num_top <= population_size
        assert num_top <= resample_amount and resample_amount <= population_size
        self.config = config
        self.action_size = int(act_space.shape[0])
        self.lower_bound = torch.tensor(act_space.low, device=config.device, dtype=torch.float32)
        self.upper_bound = torch.tensor(act_space.high, device=config.device, dtype=torch.float32)
        
        self.initial_solution = (
            ((torch.tensor(self.lower_bound) + torch.tensor(self.upper_bound)) / 2)
            .float()
            .to(config.device)
        )
        
        self.initial_solution = self.initial_solution.repeat((planning_horizon, 1))
        self.previous_solution = self.initial_solution.clone()
        self.initial_var = ((self.upper_bound - self.lower_bound) ** 2) / 16
        
        self.planning_horizon = planning_horizon
        self.num_iterations = num_iterations
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.elite_num = np.ceil(self.population_size * self.elite_ratio).astype(
            np.int32
        )
        self.world_model = world_model
        self.use_opt = use_opt
        self.num_top = num_top
        self.alpha = alpha
        self.keep_last_solution = keep_last_solution
        self.replan_freq = 1
        self.init = True
        self.resample_amount = resample_amount
        self.resample = resample
        self.return_mean_elites = return_mean_elites

    def policy(self, state: Dict[str, Tensor], sample: bool):
        if self.resample or self.init or not self.use_opt:
            best_solution = self.optimize(x0=self.previous_solution, state=state, is_resampling=True)
            self.init = False
        else:
            best_solution = self.optimize(x0=self.previous_solution, state=state, is_resampling=False)


        if self.keep_last_solution:
            self.previous_solution = best_solution.roll(-self.replan_freq, dims=0)
            self.previous_solution[-self.replan_freq :] = self.initial_solution[0]
        return best_solution[0].unsqueeze(0)

    def reset(self):
        self.previous_solution = self.initial_solution.clone()
        self.init = True

    def optimize(self, x0, state, is_resampling):

        def reward_fun(trajectories, sample):
            nonlocal state
            current_state = state.copy()
            for k, v in current_state.items():
                B, Z = v.size(0), v.size(1)
                assert B == 1
                current_state[k] = v.unsqueeze(dim=1).expand(B, trajectories.size(0), Z).reshape(-1, Z)
            returns = []
            actions = trajectories.transpose(0, 1)
            for action in actions:
                current_state = self.world_model.rssm.img_step(current_state, action, sample)
                feat = self.world_model.rssm.get_feat(current_state)
                returns.append(self.world_model.heads["reward"](feat).mode())
            # Calculate expected returns (technically sum of rewards over planning horizon)
            values = torch.stack(returns).view(self.planning_horizon, -1).sum(dim=0)
            return values

        top_trajectories = self.get_top_trajectories(x0, reward_fun, is_resampling)
        if self.use_opt:
            top_trajectories = self.optimize_trajectory_batch(reward_fun, top_trajectories)
        trajectory_batch = torch.stack(top_trajectories)
        with torch.no_grad():
            batch_rew = reward_fun(trajectory_batch, sample=False)
        i = torch.argmax(batch_rew)
        best_trajectory = trajectory_batch[i]

        if torch.any(best_trajectory.isnan()) or torch.any(best_trajectory.isinf()):
            best_trajectory = x0
        best_trajectory = torch.squeeze(best_trajectory, dim=0)
        return best_trajectory
    
    def optimize_trajectory_batch(self, obj_fun, action_sequences_list,
        start_lr=0.01, factor_shrink=1.5, max_tries=7, max_iterations=15):
        for action_sequences in action_sequences_list:
            action_sequences.requires_grad = True

        n = len(action_sequences_list)

        optimizer = Adam(
            [{
                'params': act_seq, 
                'factor': 1,
                'action_bounds': (self.lower_bound, self.upper_bound)
                } for act_seq in action_sequences_list],
            lr=start_lr
        )
        optimizer.zero_grad()

        saved_parameters = [None for i in range(n)]
        saved_opt_states = [None for i in range(n)]
        current_iteration = np.array([0 for i in range(n)])
        done = np.array([False for i in range(n)])

        action_sequences_batch = torch.stack(action_sequences_list)
        objective_all = obj_fun(action_sequences_batch, sample=False)

        current_objective = [objective_all[i] for i in range(n)]

        for i in range(n):
            action_sequences = action_sequences_list[i]
            saved_parameters[i] = action_sequences.detach().clone()
            saved_opt_states[i] = deepcopy(optimizer.state[action_sequences])
            objective_all[i].backward(retain_graph=(i != n - 1))

        while not np.all(done):
            optimizer.step()

            # Compute objectives of all trajectories after stepping
            action_sequences_batch = torch.stack(action_sequences_list)
            objective_all = obj_fun(action_sequences_batch, True)

            backwards_pass = []

            for i in range(n):
                if done[i]:
                    continue
                action_sequences = action_sequences_list[i]
                if objective_all[i] > current_objective[i]:
                    # If after the step, the cost is higher, then undo
                    action_sequences.data = saved_parameters[i].data.clone()
                    optimizer.state[action_sequences] = deepcopy(saved_opt_states[i])
                    optimizer.param_groups[i]['factor'] *= factor_shrink

                    if optimizer.param_groups[i]['factor'] > factor_shrink**max_tries:
                        # line search failed, mark action sequence as done
                        action_sequences.grad = None
                        done[i] = True
                else:
                    # successfully completed step.
                    # Save current state, and compute gradients
                    saved_parameters[i] = action_sequences.detach().clone()
                    saved_opt_states[i] = deepcopy(optimizer.state[action_sequences])
                    current_objective[i] = objective_all[i]
                    optimizer.param_groups[i]['factor'] = 1
                    action_sequences.grad = None
                    backwards_pass.append(i)
                                        
                    current_iteration[i] += 1
                    if current_iteration[i] > max_iterations:
                        action_sequences.grad = None
                        done[i] = True
                
            to_compute = [objective_all[i] for i in backwards_pass]
            grads = [(torch.empty_like(objective_all[i])*0 + 1).to(self.config.device) for i in backwards_pass]
            torch.autograd.backward(to_compute, grads)
        
        return [traj.detach() for traj in action_sequences_list]

        
    

    def get_top_trajectories(self, x0, reward_fun, is_resampling: bool):

        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
    
        mu = x0.clone()
        var = self.initial_var.clone()

        top_rewards = np.array([-1e10 for i in range(self.num_top)])
        top_sols = [torch.empty_like(mu) for i in range(self.num_top)]

        amount = self.population_size if is_resampling else self.resample_amount
        n_iter = self.num_iterations
        if self.use_opt:
            n_iter = 5
        elite_num = self.elite_num
        population = torch.zeros((amount,) + self.previous_solution.shape).to(
            device=self.config.device
        )
        for i in range(n_iter):
            lb_dist = mu - self.lower_bound
            ub_dist = self.upper_bound - mu
            mv = torch.min(torch.square(lb_dist / 2), torch.square(ub_dist / 2))
            constrained_var = torch.min(mv, var)

            population = truncated_normal_(population)
            population = population * torch.sqrt(constrained_var) + mu
            with torch.no_grad():
                values = reward_fun(population, sample=True)

            values[values.isnan()] = -1e-10
            elite_amount = elite_num if is_resampling else self.resample_amount
            best_values, elite_idx = values.topk(elite_amount)
            best_values = best_values.cpu().numpy()
            elite = population[elite_idx]

            new_mu = torch.mean(elite, dim=0)
            new_var = torch.var(elite, unbiased=False, dim=0)
            mu = self.alpha * mu + (1 - self.alpha) * new_mu
            var = self.alpha * var + (1 - self.alpha) * new_var

            for i in range(self.num_top):
                # keep track of self.num_top trajectories
                s = best_values[i] > top_rewards
                if np.any(s):
                    mask = np.ma.masked_array(top_rewards, mask=~s)
                    highest_idx = np.argmax(mask)
                    top_rewards[highest_idx] = best_values[i]
                    top_sols[highest_idx] = population[elite_idx[i]].detach().clone()

        if self.use_opt or self.return_mean_elites:
            return top_sols
        else:
            return [mu]

# inplace truncated normal function for pytorch.
# credit to https://github.com/Xingyu-Lin/mbpo_pytorch/blob/main/model.py#L64
def truncated_normal_(tensor: torch.Tensor, mean: float = 0, std: float = 1):
    """Samples from a truncated normal distribution in-place.
    Args:
        tensor (tensor): the tensor in which sampled values will be stored.
        mean (float): the desired mean (default = 0).
        std (float): the desired standard deviation (default = 1).
    Returns:
        (tensor): the tensor with the stored values. Note that this modifies the input tensor
            in place, so this is just a pointer to the same object.
    """
    torch.nn.init.normal_(tensor, mean=mean, std=std)
    while True:
        cond = torch.logical_or(tensor < mean - 2 * std, tensor > mean + 2 * std)
        if not torch.sum(cond):
            break
        tensor = torch.where(
            cond,
            torch.nn.init.normal_(
                torch.ones(tensor.shape, device=tensor.device), mean=mean, std=std
            ),
            tensor,
        )
    return tensor