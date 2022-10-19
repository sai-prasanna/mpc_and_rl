from typing import Callable, Dict
import torch
import torch.nn as nn
import torch.distributions as td
from torch import Tensor

from dreamerv2_torch import common
from dreamerv2_torch.world_model import WorldModel
from dreamerv2_torch.policy.policy import TrainablePolicy

class ActorCritic(nn.Module, TrainablePolicy):
    def __init__(self, config, act_space, tfstep):
        super().__init__()
        self.config = config
        self._use_amp = True if config.precision == 16 else False
        self.act_space = act_space
        self.tfstep = tfstep
        discrete = hasattr(act_space, "n")
        if self.config.actor.dist == "auto":
            self.config = self.config.update(
                {"actor.dist": "onehot" if discrete else "trunc_normal"}
            )
        if self.config.actor_grad == "auto":
            self.config = self.config.update(
                {"actor_grad": "reinforce" if discrete else "dynamics"}
            )
        self.actor = common.MLP(int(act_space.shape[0]), **self.config.actor)
        self.critic = common.MLP(1, **self.config.critic)
        if self.config.slow_target:
            self._target_critic = common.MLP(1, **self.config.critic)
            self._updates = nn.Parameter(
                torch.zeros((), dtype=torch.int64), requires_grad=False
            )
        else:
            self._target_critic = self.critic
        self.rewnorm = common.StreamNorm(**self.config.reward_norm)

    def initialize_optimizer(self):
        self.actor_opt = common.Optimizer(
            "actor",
            self.actor.parameters(),
            use_amp=self._use_amp,
            **self.config.actor_opt,
        )
        self.critic_opt = common.Optimizer(
            "critic",
            self.critic.parameters(),
            use_amp=self._use_amp,
            **self.config.critic_opt,
        )

    def train_batch(self, world_model: WorldModel, start, is_terminal, reward_fn):
        actor_loss, critic_loss, metrics = self.loss(
            world_model, start, is_terminal, reward_fn
        )
        with common.RequiresGrad(self.actor):
            metrics.update(self.actor_opt(actor_loss, self.actor.parameters()))
        with common.RequiresGrad(self.critic):
            metrics.update(self.critic_opt(critic_loss, self.critic.parameters()))
        self.update_slow_target()  # Variables exist after first forward pass.
        return metrics

    def loss(self, world_model, start, is_terminal, reward_fn):
        metrics = {}
        hor = self.config.imag_horizon
        # The weights are is_terminal flags for the imagination start states.
        # Technically, they should multiply the losses from the second trajectory
        # step onwards, which is the first imagined step. However, we are not
        # training the action that led into the first step anyway, so we can use
        # them to scale the whole sequence.
        with common.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                seq = world_model.imagine(self.actor, start, is_terminal, hor)
                reward = reward_fn(seq)
                seq["reward"], mets1 = self.rewnorm(reward)
                mets1 = {f"reward_{k}": v for k, v in mets1.items()}
                target, mets2 = self.target(seq)
                actor_loss, mets3 = self.actor_loss(seq, target)
        with common.RequiresGrad(self.critic):
            with torch.cuda.amp.autocast(self._use_amp):
                critic_loss, mets4 = self.critic_loss(seq, target)
        metrics.update(**mets1, **mets2, **mets3, **mets4)
        return actor_loss, critic_loss, metrics

    def actor_loss(self, seq, target):
        # Actions:      0   [a1]  [a2]   a3
        #                  ^  |  ^  |  ^  |
        #                 /   v /   v /   v
        # States:     [z0]->[z1]-> z2 -> z3
        # Targets:     t0   [t1]  [t2]
        # Baselines:  [v0]  [v1]   v2    v3
        # Entropies:        [e1]  [e2]
        # Weights:    [ 1]  [w1]   w2    w3
        # Loss:              l1    l2
        metrics = {}
        # Two states are lost at the end of the trajectory, one for the boostrap
        # value prediction and one because the corresponding action does not lead
        # anywhere anymore. One target is lost at the start of the trajectory
        # because the initial state comes from the replay buffer.
        policy = self.actor(seq["feat"][:-2].detach())
        if self.config.actor_grad == "dynamics":
            objective = target[1:]
        elif self.config.actor_grad == "reinforce":
            baseline = self._target_critic(seq["feat"][:-2]).mode()
            advantage = (target[1:] - baseline).detach()
            action = seq["action"][1:-1].detach()
            objective = policy.log_prob(action) * advantage
        elif self.config.actor_grad == "both":
            baseline = self._target_critic(seq["feat"][:-2]).mode()
            advantage = (target[1:] - baseline).detach()
            objective = policy.log_prob(seq["action"][1:-1]) * advantage
            mix = common.schedule(self.config.actor_grad_mix, self.tfstep)
            objective = mix * target[1:] + (1 - mix) * objective
            metrics["actor_grad_mix"] = mix
        else:
            raise NotImplementedError(self.config.actor_grad)
        ent = policy.entropy().unsqueeze(-1)
        ent_scale = common.schedule(self.config.actor_ent, self.tfstep)
        objective = objective + ent_scale * ent
        weight = seq["weight"].detach()
        actor_loss = -(weight[:-2] * objective).mean()
        metrics["actor_ent"] = ent.mean().detach().cpu()
        metrics["actor_ent_scale"] = ent_scale
        return actor_loss, metrics

    def critic_loss(self, seq, target):
        # States:     [z0]  [z1]  [z2]   z3
        # Rewards:    [r0]  [r1]  [r2]   r3
        # Values:     [v0]  [v1]  [v2]   v3
        # Weights:    [ 1]  [w1]  [w2]   w3
        # Targets:    [t0]  [t1]  [t2]
        # Loss:        l0    l1    l2
        dist = self.critic(seq["feat"][:-1].detach())
        target = target.detach()
        weight = seq["weight"].detach()
        critic_loss = -(dist.log_prob(target).unsqueeze(-1) * weight[:-1]).mean()
        metrics = {"critic": dist.mode().mean().detach().cpu()}
        return critic_loss, metrics

    def target(self, seq):
        # States:     [z0]  [z1]  [z2]  [z3]
        # Rewards:    [r0]  [r1]  [r2]   r3
        # Values:     [v0]  [v1]  [v2]  [v3]
        # Discount:   [d0]  [d1]  [d2]   d3
        # Targets:     t0    t1    t2
        reward = seq["reward"]
        disc = seq["discount"]
        value = self._target_critic(seq["feat"]).mode()
        # Skipping last time step because it is used for bootstrapping.
        target = common.lambda_return(
            reward[:-1],
            value[:-1],
            disc[:-1],
            bootstrap=value[-1],
            lambda_=self.config.discount_lambda,
            axis=0,
        )
        metrics = {}
        metrics["critic_slow"] = value.mean().detach().cpu()
        metrics["critic_target"] = target.mean().detach().cpu()
        return target, metrics

    def update_slow_target(self):
        if self.config.slow_target:
            if self._updates % self.config.slow_target_update == 0:
                mix = (
                    1.0
                    if self._updates == 0
                    else float(self.config.slow_target_fraction)
                )
                for s, d in zip(
                    self.critic.parameters(), self._target_critic.parameters()
                ):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
    
    def policy(self, state: Dict[str, Tensor], sample: bool):
        action_dist = self.actor(state['feat'])
        return action_dist.sample() if sample else action_dist.mode()

    def reset(self):
        pass