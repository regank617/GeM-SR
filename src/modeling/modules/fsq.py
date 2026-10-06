import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict

# FSQ for 2D Inputs

class FSQ(nn.Module):
    """
    Finite Scalar Quantization (FSQ) Layer.

    FSQ maps continuous latent vectors to a discrete grid defined by 'levels' 
    per dimension. Unlike traditional VQ-VAE, FSQ does not require a learned 
    codebook or an EM-style update; it uses a fixed grid and rounding logic 
    with a Straight-Through Estimator (STE).

    Attributes:
        levels (List[int]): Number of discrete points for each dimension.
        dim (int): Dimensionality of the latent vector (len(levels)).
        vocab_size (int): Total number of unique discrete codes (product of levels).
        zero_idx (int): Flat index of the symmetric "center" bin per dimension
            (odd: geometric center; even: ``(L-1)//2`` per axis). Used for null / stats.

    Odd levels (default): symmetric integer grid after rounding (classic FSQ).

    Even levels: set ``allow_even_levels=True``. Uses uniform binning
    ``k = clamp(round((tanh(z)+1)/2 * (L-1)), 0, L-1)`` per dimension (no
    half-integer rounding artifact). Dequantization matches ``indices_to_codes``.
    """
    def __init__(self, levels: List[int], expected_dim: int = None, allow_even_levels: bool = False):
        super().__init__()

        self.levels = list(levels)
        self.dim = len(self.levels)
        self.allow_even_levels = bool(allow_even_levels)
        has_even = any(l % 2 == 0 for l in self.levels)
        if has_even and not self.allow_even_levels:
            even_levels = [l for l in self.levels if l % 2 == 0]
            raise ValueError(
                "FSQ has even level counts; pass allow_even_levels=True for ablations "
                f"(even levels: {even_levels}). Odd-only default keeps a symmetric half-integer grid."
            )

        # Validate against expected dimensions
        if expected_dim is not None and self.dim != expected_dim:
            raise ValueError(
                f"FSQ configuration mismatch: 'levels' implies {self.dim} dims, "
                f"but expected {expected_dim}."
            )

        self.register_buffer(
            "basis",
            torch.tensor([1] + [np.prod(self.levels[:i]) for i in range(1, len(self.levels))], dtype=torch.int64),
        )
        # (L-1)/2 used in indices_to_codes for all L>=2; forward uses per-dim odd vs even paths.
        self.register_buffer("half_width", torch.tensor([(l - 1) / 2.0 for l in self.levels], dtype=torch.float32))
        self.vocab_size = int(np.prod(self.levels))
        z_center = torch.tensor([(l - 1) // 2 for l in self.levels], dtype=torch.int64)
        self.register_buffer("zero_idx", torch.sum(z_center * self.basis))

    def forward(self, z: torch.Tensor):
        """
        Quantizes a continuous tensor into discrete codes.

        Args:
            z (torch.Tensor): Continuous latent tensor [B, D, H, W].

        Returns:
            z_q (torch.Tensor): Quantized continuous tensor (gradients attached).
            indices (torch.Tensor): LongTensor of flat indices [B, H, W].
            commit_loss (torch.Tensor): MSE between continuous and rounded values.
        """
        with torch.no_grad():
            percent_saturated = (z.abs() > 2.0).float().mean().item() * 100
            percent_dead = (z.abs() < 0.1).float().mean().item() * 100
        z = torch.tanh(z)
        B, D, H, W = z.shape
        z_pos_ch = []
        z_q_ch = []
        commit_terms: List[torch.Tensor] = []
        for i in range(self.dim):
            Li = int(self.levels[i])
            zi = z[:, i, :, :]
            if Li % 2 == 1:
                hw = (Li - 1) / 2.0
                z_scaled = zi * hw
                z_rounded = torch.round(z_scaled)
                zp = (z_rounded + hw).long().clamp(0, Li - 1)
                z_q_scaled = z_scaled + (z_rounded - z_scaled).detach()
                z_q_i = z_q_scaled / hw
                commit_terms.append(F.mse_loss(z_scaled, z_rounded.detach()))
            else:
                if Li <= 1:
                    zp = torch.zeros((B, H, W), dtype=torch.long, device=z.device)
                    z_q_i = torch.zeros_like(zi)
                    commit_terms.append(torch.tensor(0.0, device=z.device, dtype=z.dtype))
                else:
                    v = torch.clamp(torch.round((zi + 1.0) * (Li - 1) / 2.0), 0, Li - 1)
                    z_recon = v * (2.0 / (Li - 1)) - 1.0
                    z_q_i = zi + (z_recon - zi).detach()
                    zp = v.long()
                    commit_terms.append(F.mse_loss(zi, z_recon.detach()))
            z_pos_ch.append(zp)
            z_q_ch.append(z_q_i)
        z_pos = torch.stack(z_pos_ch, dim=1)
        z_q = torch.stack(z_q_ch, dim=1)
        indices = torch.sum(z_pos.long() * self.basis.view(1, -1, 1, 1).to(z.device), dim=1)

        z_scaled_legacy = z * self.half_width.view(1, -1, 1, 1).to(z.device)
        z_rounded_legacy = torch.round(z_scaled_legacy)
        with torch.no_grad():
            quantization_error = F.mse_loss(z_scaled_legacy, z_rounded_legacy).item()

        commit_loss = torch.stack(commit_terms).mean()

        info_dict = {
            "percent_saturated": percent_saturated,
            "percent_dead": percent_dead,
            "quantization_error": quantization_error,
            "commit_loss": commit_loss,
        }
        return z_q, indices, info_dict

    def indices_to_codes(self, indices: torch.Tensor):
        """
        De-quantizes flat indices back into continuous coordinate vectors.

        Args:
            indices (torch.Tensor): Flat discrete indices [B, H, W].

        Returns:
            torch.Tensor: Quantized coordinates in latent space [B, D, H, W].
        """
        codes_list = []
        residue = indices
        for i in reversed(range(self.dim)):
            b = self.basis[i]
            l = self.levels[i]
            val = torch.div(residue, b, rounding_mode='floor')
            val = val % l
            codes_list.append(val)
            residue = residue - (val * b)
        codes_list = list(reversed(codes_list))
        #z_pos = torch.stack(codes_list, dim=1).float()
        #z_pos = torch.stack(codes_list, dim=-1).float()
        z_pos = torch.stack(codes_list, dim=-1).float()
        
        # FIX: Normalize back to [-1.0, 1.0] domain
        hw = self.half_width.to(z_pos.device)
        return (z_pos - hw) / hw
        #return z_pos - self.half_width.to(z_pos.device) #.view(1, -1, 1, 1)
    

# FSQ for 3D inputs
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

class FSQ3D(nn.Module):
    """
    3D Finite Scalar Quantization (FSQ) Layer.
    Expects input tensors of shape [B, C, D, H, W].
    """
    def __init__(self, levels: List[int], expected_dim: int = None):
        super().__init__()

        # Ensure all levels are odd
        if any(l % 2 == 0 for l in levels):
            even_levels = [l for l in levels if l % 2 == 0]
            raise ValueError(
                f"FSQ requires all levels to be odd to ensure a centered zero-point. "
                f"Found even levels: {even_levels}"
            )

        self.levels = levels
        self.dim = len(levels)

        # Validate against expected dimensions
        if expected_dim is not None and self.dim != expected_dim:
            raise ValueError(
                f"FSQ configuration mismatch: 'levels' implies {self.dim} dims, "
                f"but expected {expected_dim}."
            )
        
        self.register_buffer('basis', torch.tensor([1] + [np.prod(levels[:i]) for i in range(1, len(levels))], dtype=torch.int64))
        self.register_buffer('half_width', torch.tensor([(l - 1) / 2 for l in levels]))
        self.vocab_size = np.prod(levels)
        z_pos = self.half_width
        self.register_buffer('zero_idx', torch.sum(z_pos.long() * self.basis))

    def forward(self, z: torch.Tensor):
        """
        Quantizes a continuous 3D tensor into discrete codes.

        Args:
            z (torch.Tensor): Continuous latent tensor [B, C, D, H, W].

        Returns:
            z_q (torch.Tensor): Quantized continuous tensor (gradients attached).
            indices (torch.Tensor): LongTensor of flat indices [B, D, H, W].
            info_dict (dict): Dictionary of quantization statistics.
        """
        with torch.no_grad():
            percent_saturated = (z.abs() > 2.0).float().mean().item() * 100 
            percent_dead = (z.abs() < 0.1).float().mean().item() * 100
            
        z = torch.tanh(z)
        
        # CHANGED TO 5D VIEW: [1, Channels, 1, 1, 1]
        view_shape = (1, -1, 1, 1, 1)
        
        z_scaled = z * self.half_width.view(*view_shape)
        z_rounded = torch.round(z_scaled)

        with torch.no_grad():
            quantization_error = F.mse_loss(z_scaled, z_rounded).item()

        # STE (Straight-Through Estimator)
        z_q_scaled = z_scaled + (z_rounded - z_scaled).detach()
        z_q = z_q_scaled / self.half_width.view(*view_shape)

        z_pos = z_rounded + self.half_width.view(*view_shape)
        
        # Multiply by basis and sum over the channel dimension (dim=1)
        # Results in indices of shape [B, D, H, W]
        indices = torch.sum(z_pos.long() * self.basis.view(*view_shape), dim=1)
        commit_loss = F.mse_loss(z_scaled, z_rounded.detach())

        info_dict = {
            'percent_saturated': percent_saturated,
            'percent_dead': percent_dead,
            'quantization_error': quantization_error,
            'commit_loss': commit_loss
        }
        return z_q, indices, info_dict 

    def indices_to_codes(self, indices: torch.Tensor):
        """
        De-quantizes flat indices back into continuous coordinate vectors.

        Args:
            indices (torch.Tensor): Flat discrete indices [B, D, H, W].

        Returns:
            torch.Tensor: Quantized coordinates in latent space [B, C, D, H, W].
        """
        codes_list = []
        residue = indices
        for i in reversed(range(self.dim)):
            b = self.basis[i]
            l = self.levels[i]
            val = torch.div(residue, b, rounding_mode='floor')
            val = val % l
            codes_list.append(val)
            residue = residue - (val * b)
            
        codes_list = list(reversed(codes_list))
        
        # FIX: Stack on dim=1 to output [B, C, D, H, W] instead of [B, D, H, W, C]
        z_pos = torch.stack(codes_list, dim=1).float()
        
        # Normalize back to [-1.0, 1.0] domain using 5D view
        hw = self.half_width.view(1, -1, 1, 1, 1).to(z_pos.device)
        return (z_pos - hw) / hw