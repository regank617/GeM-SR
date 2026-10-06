from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# SR3 schedule (same as train_sr3 / GaussianDiffusion defaults) for UNet time embedding scale
def _import_make_beta_schedule():
    try:
        from model.sr3_modules.diffusion import make_beta_schedule
        return make_beta_schedule
    except ImportError:
        _sr3 = Path(__file__).resolve().parent.parent / "sr3" / "repo"
        if str(_sr3) not in sys.path:
            sys.path.insert(0, str(_sr3))
        from model.sr3_modules.diffusion import make_beta_schedule
        return make_beta_schedule


class FlowMatching(nn.Module):
    """
    Rectified Flow (Flow Matching) formulation for 1-step to few-step generative modeling.

    The SR3 UNet's ``PositionalEncoding`` multiplies the **scalar noise level** by
    frequency terms (WaveGrad-style); SR3 training never passes integer timesteps — it passes
    ``continuous_sqrt_alpha_cumprod`` in ``(0, 1]`` (see ``GaussianDiffusion.p_losses``).

    We map flow time ``t in [0, 1]`` (noise at 0 → data at 1) into that **same numeric
    range** as ``sqrt(alphas_cumprod)`` so sinusoidal channels are active like in SR3,
    instead of feeding raw ``t`` which can sit in a narrower effective band than the DDPM
    training distribution.
    """

    def __init__(self, denoise_fn, n_timestep=2000, linear_start=1e-6, linear_end=1e-2):
        super().__init__()
        self.denoise_fn = denoise_fn
        make_beta_schedule = _import_make_beta_schedule()
        betas = make_beta_schedule(
            schedule="linear",
            n_timestep=n_timestep,
            linear_start=linear_start,
            linear_end=linear_end,
        )
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1.0 - betas
        alphas_cumprod = torch.tensor(np.cumprod(alphas, axis=0), dtype=torch.float32)
        sqrt_ac = torch.sqrt(alphas_cumprod)
        self.register_buffer("_sqrt_alphas_cumprod", sqrt_ac)

    def _flow_t_to_sr3_noise_level(self, t):
        """Map flow time t∈[0,1] to SR3-style continuous sqrt-alpha (high = clean, low = noisy)."""
        if t.dim() == 1:
            t = t.view(-1, 1)
        sa = self._sqrt_alphas_cumprod
        lo, hi = sa.min(), sa.max()
        return t * hi + (1.0 - t) * lo

    def forward(self, x_in):
        """
        Training Step: Learn the straight-line vector field.
        """
        x_1 = x_in["HR"]  # Target Data (t=1)
        x_lr = x_in["SR"]  # Upsampled Low-Res Condition
        b, c, h, w = x_1.shape

        # 1. Sample continuous time t ~ U(0, 1)
        t = torch.rand((b,), device=x_1.device)
        t_expand = t.view(b, 1, 1, 1)

        # 2. Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(x_1)

        # 3. Interpolate along the straight line
        x_t = t_expand * x_1 + (1.0 - t_expand) * x_0

        # 4. The target velocity is simply Data - Noise
        target_velocity = x_1 - x_0

        # 5. UNet conditioning on same scale as SR3 DDPM (sqrt alphas), not raw t
        noise_level = self._flow_t_to_sr3_noise_level(t)
        pred_velocity = self.denoise_fn(
            torch.cat([x_lr, x_t], dim=1),
            noise_level,
        )

        # 6. Loss is the MSE between predicted and actual velocity
        return F.mse_loss(pred_velocity, target_velocity)

    @torch.no_grad()
    def super_resolution(self, x_lr, num_steps=4, continous=False):
        """
        Inference Step: Euler Integration to generate images in ``num_steps``.
        """
        b, c, h, w = x_lr.shape
        x_t = torch.randn_like(x_lr)  # Start at pure noise (t=0)

        dt = 1.0 / num_steps
        ret_img = x_t if continous else None

        for i in range(num_steps):
            # Flow time at step start: i / num_steps (noise-like → data-like as i increases)
            t = torch.full((b, 1), i / num_steps, device=x_lr.device, dtype=x_lr.dtype)
            noise_level = self._flow_t_to_sr3_noise_level(t)

            pred_velocity = self.denoise_fn(
                torch.cat([x_lr, x_t], dim=1),
                noise_level,
            )

            # Euler step: x_{t+1} = x_t + v * dt
            x_t = x_t + pred_velocity * dt

            if continous:
                ret_img = torch.cat([ret_img, x_t], dim=0)

        x_t = x_t.clamp(-1.0, 1.0)
        return ret_img if continous else x_t
