from typing import Callable, Dict

import torch
import torch.distributions as td
import torch.nn as nn
from torch import Tensor

from dreamerv2_torch import common


class WorldModel(common.Module):
    def __init__(self, config, obs_space, tfstep):
        super().__init__()
        self._use_amp = True if config.precision == 16 else False
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        self.config = config
        self.tfstep = tfstep

        self.rssm = common.EnsembleRSSM(**config.rssm)
        self.encoder = common.Encoder(shapes, **config.encoder)
        self.heads = nn.ModuleDict()
        self.heads["decoder"] = common.Decoder(shapes=shapes, **config.decoder)
        self.heads["reward"] = common.MLP(1, **config.reward_head)

        if config.pred_discount:
            self.heads["discount"] = common.MLP(1, **config.discount_head)
        for name in config.grad_heads:
            assert name in self.heads, name

    def initialize_optimizer(self):
        self.model_opt = common.Optimizer(
            "model",
            list(self.encoder.parameters())
            + list(self.rssm.parameters())
            + list(self.heads.parameters()),
            **self.config.model_opt,
            use_amp=self._use_amp,
        )

    def _train(self, data, state=None):
        with common.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                model_loss, state, outputs, metrics = self.loss(data, state)
            metrics.update(self.model_opt(model_loss))
        return state, outputs, metrics

    def loss(self, data, state=None):
        data = self.preprocess(data)
        embed = self.encoder(data)
        post, prior = self.rssm.observe(embed, data["action"], data["is_first"], state)
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
        assert len(kl_loss.shape) == 0
        likes = {}
        losses = {"kl": kl_loss}
        feat = self.rssm.get_feat(post)
        for name, head in self.heads.items():
            grad_head = name in self.config.grad_heads
            inp = feat if grad_head else feat.detach()
            out = head(inp)
            dists = out if isinstance(out, dict) else {name: out}
            for key, dist in dists.items():
                like = dist.log_prob(data[key])
                likes[key] = like
                losses[key] = -like.mean()
        model_loss = sum(
            self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items()
        )
        detach_dict = lambda x: {k: v.detach() for k, v in x.items()}
        outs = dict(
            embed=embed,
            feat=feat,
            post=detach_dict(post),
            prior=prior,
            likes=likes,
            kl=kl_value,
        )
        metrics = {
            f"{name}_loss": value.detach().cpu() for name, value in losses.items()
        }
        metrics["model_kl"] = kl_value.mean().detach().cpu()
        metrics["prior_ent"] = self.rssm.get_dist(prior).entropy().mean().detach().cpu()
        metrics["post_ent"] = self.rssm.get_dist(post).entropy().mean().detach().cpu()
        last_state = {k: v[:, -1].detach() for k, v in post.items()}
        return model_loss, last_state, outs, metrics

    def imagine(
        self,
        policy: Callable[[Tensor], "td.Distribution"],
        start_state: Dict[str, Tensor],
        is_terminal: Tensor,
        horizon: int,
    ) -> Tensor:
        """Given a batch of states, rolls out more state of length horizon."""
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start_state.items()}
        start["feat"] = self.rssm.get_feat(start)
        start["action"] = torch.zeros_like(policy(start["feat"]).mode())
        seq = {k: [v] for k, v in start.items()}
        for _ in range(horizon):
            action = policy(seq["feat"][-1].detach()).sample()
            state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
            feature = self.rssm.get_feat(state)
            for key, value in {**state, "action": action, "feat": feature}.items():
                seq[key].append(value)

        seq = {k: torch.stack(v, 0) for k, v in seq.items()}

        if "discount" in self.heads:
            disc = self.heads["discount"](seq["feat"]).mean()
            if is_terminal is not None:
                # Override discount prediction for the first step with the true
                # discount factor from the replay buffer.
                true_first = 1.0 - flatten(is_terminal).astype(disc.dtype)
                true_first *= self.config.discount
                disc = torch.concat([true_first[None], disc[1:]], 0)
        else:
            disc = self.config.discount * torch.ones(
                seq["feat"].shape[:-1], device=seq["feat"].device
            )
        seq["discount"] = disc.unsqueeze(-1)
        # Shift discount factors because they imply whether the following state
        # will be valid, not whether the current state is valid.
        seq["weight"] = torch.cumprod(
            torch.cat([torch.ones_like(disc[:1]), disc[:-1]], 0), 0
        ).unsqueeze(-1)
        return seq

    def preprocess(self, obs):
        obs = obs.copy()
        dtype = torch.float32
        obs = {
            k: torch.tensor(v).to(next(self.parameters()).device)
            for k, v in obs.items()
        }
        for key, value in obs.items():
            if key.startswith("log_"):
                continue
            if value.dtype == torch.int32:
                value = value.to(dtype)
            if value.dtype == torch.uint8:
                value = value.to(dtype) / 255.0 - 0.5
            obs[key] = value
        obs["reward"] = {
            "identity": lambda x: x,
            "sign": torch.sign,
            "tanh": torch.tanh,
        }[self.config.clip_rewards](obs["reward"]).unsqueeze(-1)
        obs["discount"] = 1.0 - obs["is_terminal"].float().unsqueeze(-1)  # .to(dtype)
        obs["discount"] *= self.config.discount
        return obs

    def video_pred(self, data, key):
        decoder = self.heads["decoder"]
        truth = data[key][:6] + 0.5
        embed = self.encoder(data)
        states, _ = self.rssm.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = decoder(self.rssm.get_feat(states))[key].mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.rssm.imagine(data["action"][:6, 5:], init)
        openl = decoder(self.rssm.get_feat(prior))[key].mode()
        model = torch.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        video = torch.concat([truth, model, error], 2)
        B, T, H, W, C = video.shape
        return video.permute(1, 2, 0, 3, 4).reshape(T, H, B * W, C).cpu()
