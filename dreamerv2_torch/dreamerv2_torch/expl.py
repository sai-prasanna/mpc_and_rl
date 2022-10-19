import torch.distributions as td

import agent
import common
import torch
from torch import nn


class Random(nn.Module):
    def __init__(self, config, act_space, wm, tfstep, reward):
        self.config = config
        self.act_space = act_space

    def actor(self, feat):
        shape = feat.shape[:-1] + self.act_space.shape
        if self.config.actor.dist == "onehot":
            return common.OneHotDist(torch.zeros(shape))
        else:
            dist = td.Uniform(-torch.ones(shape), torch.ones(shape))
            return td.Independent(dist, 1)

    def train(self, start, context, data):
        return None, {}


class Plan2Explore(nn.Module):
    def __init__(self, config, act_space, wm, tfstep, reward):
        super().__init__()
        self.config = config
        self.reward = reward
        self.wm = wm
        self.ac = agent.ActorCritic(config, act_space, tfstep)
        self.actor = self.ac.actor
        stoch_size = config.rssm.stoch
        self._use_amp = config.precision == 16
        if config.rssm.discrete:
            stoch_size *= config.rssm.discrete
        size = {
            "embed": 32 * config.encoder.cnn_depth,
            "stoch": stoch_size,
            "deter": config.rssm.deter,
            "feat": config.rssm.stoch + config.rssm.deter,
        }[self.config.disag_target]
        self._networks = nn.ModuleList(
            [common.MLP(size, **config.expl_head) for _ in range(config.disag_models)]
        )
        self.extr_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)
        self.intr_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)

    def lazy_initialize(self, start, context, data):
        self.loss(start, context, data)
        self.ac.loss(self.wm, start, data["is_terminal"], self._intr_reward)
        self.initialize_optimizer()
        self.ac.initialize_optimizer()

    def initialize_optimizer(self):
        self.opt = common.Optimizer(
            "expl", self._networks.parameters(), **self.config.expl_opt
        )

    def loss(self, start, context, data):
        data = self.wm.preprocess(data)
        stoch = start["stoch"]
        if self.config.rssm.discrete:
            stoch = torch.reshape(
                stoch, stoch.shape[:-2] + (stoch.shape[-2] * stoch.shape[-1],)
            )
        target = {
            "embed": context["embed"],
            "stoch": stoch,
            "deter": start["deter"],
            "feat": context["feat"],
        }[self.config.disag_target]
        inputs = context["feat"]
        if self.config.disag_action_cond:
            action = data["action"].to(inputs.dtype)
            inputs = torch.concat([inputs, action], -1)
        loss = self._ensemble_loss(inputs, target)
        return loss

    def _train(self, start, context, data):
        metrics = {}
        with common.RequiresGrad(self._networks):
            with torch.cuda.amp.autocast(self._use_amp):
                loss = self.loss(start, context, data)
            metrics.update(self.opt(loss))
        metrics.update(
            self.ac._train(self.wm, start, data["is_terminal"], self._intr_reward)
        )
        return None, metrics

    def _intr_reward(self, seq):
        inputs = seq["feat"]
        if self.config.disag_action_cond:
            action = seq["action"].to(inputs.dtype)
            inputs = torch.concat([inputs, action], -1)
        preds = [head(inputs).mode() for head in self._networks]
        disag = torch.stack(preds, dim=0).std(0).mean(-1)
        if self.config.disag_log:
            disag = torch.log(disag)
        reward = self.config.expl_intr_scale * self.intr_rewnorm(disag)[0]
        if self.config.expl_extr_scale:
            reward += (
                self.config.expl_extr_scale * self.extr_rewnorm(self.reward(seq))[0]
            )
        return reward.unsqueeze(-1)

    def _ensemble_loss(self, inputs, targets):
        if self.config.disag_offset:
            targets = targets[:, self.config.disag_offset :]
            inputs = inputs[:, : -self.config.disag_offset]
        targets = targets.detach()
        inputs = inputs.detach()
        preds = [head(inputs) for head in self._networks]
        loss = -sum([pred.log_prob(targets).mean() for pred in preds])
        return loss


class ModelLoss(nn.Module):
    def __init__(self, config, act_space, wm, tfstep, reward):
        self.config = config
        self._use_amp = config.precision == 16
        self.reward = reward
        self.wm = wm
        self.ac = agent.ActorCritic(config, act_space, tfstep)
        self.actor = self.ac.actor
        self.head = common.MLP(1, **self.config.expl_head)

    def initialize_optimizer(self):
        self.opt = common.Optimizer("expl", self.head, **self.config.expl_opt)

    def train(self, start, context, data):
        metrics = {}
        with common.RequiresGrad(self.head):
            with torch.cuda.amp.autocast(self._use_amp):
                loss = self.loss(context)
            metrics.update(self.opt(loss))
        metrics.update(
            self.ac.train(self.wm, start, data["is_terminal"], self._intr_reward)
        )
        return None, metrics

    def loss(self, context):
        target = context[self.config.expl_model_loss].to(torch.float32)
        return -self.head(context["feat"]).log_prob(target).mean()

    def _intr_reward(self, seq):
        reward = self.config.expl_intr_scale * self.head(seq["feat"]).mode()
        if self.config.expl_extr_scale:
            reward += self.config.expl_extr_scale * self.reward(seq)
        return reward
