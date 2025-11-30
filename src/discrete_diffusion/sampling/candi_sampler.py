"""Hybrid sampler for CANDI."""

from __future__ import annotations

import torch

from ..forward_process.utils import sample_categorical
from .base import Sampler


class CANDI_Sampler(Sampler):
    """Base inference method that implements helper functions needed for CANDI"""

    def __init__(self, config, forward_process=None, use_approx=True):
        self.config = config
        self.forward_process = forward_process
        self.use_approx = use_approx
        self.num_steps = config.sampling.steps


    def _continuous_step(
        self,
        model,
        x: torch.Tensor,
        time_t: torch.Tensor,
        time_s: torch.Tensor,
        sigma_s: torch.Tensor,
        sigma_t: torch.Tensor,
        reveal_mask: torch.Tensor = None,
        is_embed=False,
    ) -> torch.Tensor:

        dt_cont_vec = (
            torch.ones(x.shape[0], device=x.device) * (sigma_s - sigma_t).item()
        )
        time_t_vec = torch.ones(x.shape[0], device=x.device) * time_t.item()
        sigma_t_vec = torch.ones(x.shape[0], device=x.device) * sigma_t.item()
        if reveal_mask is None:
            reveal_mask = torch.zeros(x.shape[:-1], device=x.device)
        cond_denoised = model.forward(
            xt=x,
            discrete_noise=time_t_vec,
            reveal_mask=reveal_mask,
            continuous_noise=sigma_t_vec,
            is_embed=is_embed,
        ).double()
        denoised = cond_denoised.exp()
        if self.is_embed:
            x0_hat = self.backbone.get_embedding(denoised)
        else:
            x0_hat = denoised
        d = (x - x0_hat) / (sigma_t_vec[:, None, None] ** 2)
        x_cont = x - dt_cont_vec[:, None, None] * d
        return x_cont, denoised

    def _continuous_step_approx(
        self,
        model,
        x: torch.Tensor,
        time_t: torch.Tensor,
        time_s: torch.Tensor,
        sigma_s: torch.Tensor,
        sigma_t: torch.Tensor,
        embedding_cache: torch.Tensor,
        reveal_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        dt = sigma_s - sigma_t
        time_t_vec = torch.ones(x.shape[0], device=x.device) * time_t.item()
        sigma_t_vec = torch.ones(x.shape[0], device=x.device) * sigma_t.item()
        if reveal_mask is None:
            reveal_mask = torch.zeros(x.shape[:-1], device=x.device)
        cond_denoised = model.forward(
            xt=x,
            discrete_noise=time_t_vec,
            reveal_mask=reveal_mask,
            continuous_noise=sigma_t_vec[:, None, None],
            embedding=embedding_cache,
        ).double()

        denoised = cond_denoised.exp()
        x0_hat = sample_categorical(denoised)
        embedding_hat = model.backbone.get_embedding(x0_hat)
        d = (embedding_cache - embedding_hat) / (sigma_t**2)
        new_embedding_cache = embedding_cache - dt * d
        return new_embedding_cache, x0_hat

    def _discrete_step_approx(
        self, x0_hat, xt, t, dt, prev_clean_mask, noise_removal_step=False
    ):

        if noise_removal_step:
            s = 0
        else:
            s = t - dt

        # unmasking correspends to 1-alpha(s) / 1-alpha(t)
        # this is just s / t under a log linear schedule
        unmask = (
            torch.rand(prev_clean_mask.shape, device=prev_clean_mask.device)
            < (t - s) / t
        )
        xt[~prev_clean_mask] = x0_hat[~prev_clean_mask]
        new_clean_mask = prev_clean_mask | unmask
        return xt, new_clean_mask

    def _discrete_step(
        self, x_sigma, p_x0, t, dt, prev_clean_mask, noise_removal_step=False
    ):
        if noise_removal_step:
            s = 0.0
        else:
            s = t - dt

        t_vec = torch.ones(x_sigma.shape[0], device=x_sigma.device) * t.item()
        s_vec = torch.ones(x_sigma.shape[0], device=x_sigma.device) * s.item()
        mask_probs = torch.ones(
            (x_sigma.shape[0], x_sigma.shape[1], 1), device=x_sigma.device
        ) * (s)
        unmasked_probs = p_x0 * (t_vec - s_vec)[:, None, None]

        q_xs = torch.cat([unmasked_probs, mask_probs], dim=-1)
        _x = sample_categorical(q_xs)
        new_clean_mask = (prev_clean_mask.bool() | (_x != self.mask_index)).float()

        # For tokens that got sampled to real values (not mask), use those
        old_x_tokens = x_sigma.argmax(dim=-1)

        # For tokens that got sampled to mask, keep old tokens but mark as not clean
        sampled_real_tokens = torch.where(_x != self.mask_index, _x, old_x_tokens)
        # Apply copy logic: keep old tokens where prev_clean_mask is True
        updated_tokens = torch.where(
            prev_clean_mask.bool(), old_x_tokens, sampled_real_tokens
        )
        updated_x = (
            torch.nn.functional.one_hot(updated_tokens, num_classes=x_sigma.shape[-1])
            .float()
            .to(x_sigma.device)
        )

        updated_x = (
            updated_x * new_clean_mask.unsqueeze(-1)
            + (1 - new_clean_mask).unsqueeze(-1) * x_sigma
        )

        return updated_x, new_clean_mask

    def generate_samples_nocache(
        self, model, *, num_samples, num_steps, eps, inject_bos
    ):
        if num_steps is None:
            num_steps = self.config.sampling.steps

        x = self.prior_sample(num_samples, model.num_tokens)
        clean_mask = torch.zeros((num_samples, model.num_tokens), device=x.device)
        timesteps = torch.linspace(0.999, eps, num_steps + 1, device=model.device)
        continuous_noise = model.noise.sigma_t(timesteps)
        dt = (1 - eps) / (num_steps)

        self.max_sigma = continuous_noise.max().item()

        self.prev_px0 = None
        for i in range(num_steps):
            t = timesteps[i]
            s = timesteps[i + 1]

            sigma_s = continuous_noise[i]
            sigma_t = continuous_noise[i + 1]

            x_cont, p_x0 = self._continuous_step(
                model=model, 
                x=x,
                time_t=t,
                time_s=s,
                sigma_s=sigma_s,
                sigma_t=sigma_t,
                reveal_mask=clean_mask,
            )

            x, clean_mask = self._discrete_step(
                x_cont, p_x0, t, dt, prev_clean_mask=clean_mask
            )
        final_tokens = x.argmax(dim=-1)
        return final_tokens

    # optimized function with cached embeddings
    @torch.no_grad()
    def generate_samples_cache(self, model, *, num_samples, num_steps, eps, inject_bos):
        if num_steps is None:
            num_steps = self.config.sampling.steps

        x = model.prior_sample(num_samples, model.num_tokens)
        embedding_cache = model.backbone.get_embedding(x)
        timesteps = torch.linspace(0.999, eps, num_steps + 1, device=model.device)
        continuous_noise = model.noise.sigma_t(timesteps)
        clean_mask = torch.zeros(
            (num_samples, model.num_tokens), device=x.device, dtype=torch.bool
        )

        timesteps = torch.linspace(0.999, eps, num_steps + 1, device=model.device)
        dt = (1 - eps) / (num_steps)

        self.max_sigma = continuous_noise.max().item()
        x = x.argmax(dim=-1)
        for i in range(num_steps):
            t = timesteps[i]
            s = timesteps[i + 1]

            sigma_s = continuous_noise[i]
            sigma_t = continuous_noise[i + 1]
            embedding_cache, x0_hat = self._continuous_step_approx(
                model=model, 
                x=x,
                time_t=t,
                time_s=s,
                sigma_s=sigma_s,
                sigma_t=sigma_t,
                reveal_mask=clean_mask.float(),
                embedding_cache=embedding_cache,
            )
            x, clean_mask = self._discrete_step_approx(
                x0_hat, x, t, dt, prev_clean_mask=clean_mask, noise_removal_step=False
            )
        return x
    
    def generate(
            self, model, *, num_samples, num_steps, eps, inject_bos
        ):
        print(num_steps)
        if num_steps is None:
            num_steps = self.num_steps
        if self.use_approx:
            return self.generate_samples_cache(
                model, num_samples=num_samples, num_steps=num_steps, eps=eps, inject_bos=inject_bos
            )
        else:
            return self.generate_samples_nocache(
                model, num_samples=num_samples, num_steps=num_steps, eps=eps, inject_bos=inject_bos
            )
    
__all__ = ["CANDI_Sampler"]