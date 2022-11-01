from typing import Callable, Dict
from dreamerv2_torch.policy.policy import TrainablePolicy
import torch
import torch.nn as nn
import torch.distributions as td
from torch import Tensor

from dreamerv2_torch import common
from dreamerv2_torch import expl
from dreamerv2_torch.world_model import WorldModel
from dreamerv2_torch.policy import ActorCritic, CrossEntropyMethodMPC, GradientOptimizerMPC, CrossEntropyGDMPC
import expl


class Agent(nn.Module):
    def __init__(self, config, obs_space, act_space, step: common.Counter):
        super().__init__()
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.step = step
        self.register_buffer("tfstep", torch.ones(()) * int(self.step))
        self.wm = WorldModel(config, obs_space, self.tfstep)
        if config.task_behavior == 'actor_critic':
            self._task_behavior = ActorCritic(config, self.act_space, self.tfstep)
        elif config.task_behavior == 'cem':
            self._task_behavior = CrossEntropyMethodMPC(config, self.act_space, 15, 10, 1000, 100, self.wm)
        elif config.task_behavior == 'grad':
            self._task_behavior = GradientOptimizerMPC(config, self.act_space, 15, 10, 0.02, self.wm)
        elif config.task_behavior == 'cem_gd':
            self._task_behavior = CrossEntropyGDMPC(config, self.act_space,planning_horizon=45, num_iterations=5, elite_ratio=0.1, world_model=self.wm, population_size=10, alpha=0.1, num_top=1, resample_amount=10)
        if config.expl_behavior == "greedy":
            self._expl_behavior = self._task_behavior
        else:
            self._expl_behavior = getattr(expl, config.expl_behavior)(
                self.config,
                self.act_space,
                self.wm,
                self.tfstep,
                lambda seq: self.wm.heads["reward"](seq["feat"]).mode(),
            )

    def policy(self, obs, state=None, mode="train"):
        obs = self.wm.preprocess(obs)
        self.tfstep.copy_(torch.tensor([int(self.step)])[0])
        if state is None:
            latent = self.wm.rssm.initial(len(obs["reward"]))
            action = torch.zeros((len(obs["reward"]),) + self.act_space.shape).to(
                obs["reward"].device
            )
            state = latent, action
            self._task_behavior.reset()
        latent, action = state
        embed = self.wm.encoder(obs)
        sample = (mode == "train") or not self.config.eval_state_mean
        latent, _ = self.wm.rssm.obs_step(
            latent, action, embed, obs["is_first"], sample
        )
        policy_state = latent.copy()
        policy_state['feat'] = self.wm.rssm.get_feat(latent)
        if mode == "eval":
            action = self._task_behavior.policy(policy_state, sample=False)
            noise = self.config.eval_noise
        elif mode == "explore":
            action = self._expl_behavior.policy(policy_state, sample=True)
            noise = self.config.expl_noise
        elif mode == "train":
            sample = self._task_behavior.policy(policy_state, sample=True)
            noise = self.config.expl_noise
        action = common.action_noise(action, noise, self.act_space)
        outputs = {"action": action}
        state = (latent, action)
        return outputs, state

    def _train(self, data, state=None):
        metrics = {}
        state, outputs, mets = self.wm._train(data, state)
        metrics.update(mets)
        start = outputs["post"]
        reward = lambda seq: self.wm.heads["reward"](seq["feat"]).mode()
        if isinstance(self._task_behavior, TrainablePolicy):
            metrics.update(
                self._task_behavior.train_batch(self.wm, start, data["is_terminal"], reward)
            )
        if self.config.expl_behavior != "greedy":
            mets = self._expl_behavior.train_batch(start, outputs, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        return state, metrics

    def report(self, data):
        report = {}
        data = self.wm.preprocess(data)
        for key in self.wm.heads["decoder"].cnn_keys:
            name = key.replace("/", "_")
            report[f"openl_{name}"] = self.wm.video_pred(data, key)
        return report

    def initialize_lazy_modules(self, data):
        _, _, outputs, _ = self.wm.loss(data, None)
        start = outputs["post"]
        reward = lambda seq: self.wm.heads["reward"](seq["feat"]).mode()
        
        if self.config.expl_behavior != "greedy":
            self._expl_behavior.lazy_initialize(start, outputs, data)
        self.zero_grad()
        self.wm.initialize_optimizer()
        if isinstance(self._task_behavior, TrainablePolicy):
            self._task_behavior.loss(self.wm, start, data["is_terminal"], reward)
            self._task_behavior.initialize_optimizer()
