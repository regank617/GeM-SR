import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # When imported as part of the package.
    from .canopy_plots import createCHM
except ImportError:
    # Fallback for running as a standalone script (python -m hhdc.forward_model).
    from hhdc.canopy_plots import createCHM


class LidarForwardImagingModel(nn.Module):
    def __init__(
        self,
        input_res_m=(2.0, 2.0),
        output_res_m=(3.0, 6.0),
        footprint_diameter_m=10.0,
        b=0.1,
        eta=0.5,
        ref_altitude=500.0,
        ref_photon_count=20.0,
    ):
        """
        Args:
            input_res_m (tuple): Physical size of input pixels (dy, dx) in meters.
            output_res_m (tuple): Physical size of output pixels (dy, dx) in meters.
            footprint_diameter_m (float): The 1/e^2 beam diameter in meters.
            b (float): Background noise.
            eta (float): Readout noise.
            ref_altitude (float): Reference altitude (km).
            ref_photon_count (float): Target photon count.
        """
        super().__init__()
        self.b = b
        self.eta = eta
        self.ref_altitude = ref_altitude
        self.ref_photon_count = ref_photon_count

        self.input_res_m = input_res_m
        self.output_res_m = output_res_m

        # 1. Calculate Area Scale Factor
        in_area = input_res_m[0] * input_res_m[1]
        out_area = output_res_m[0] * output_res_m[1]
        self.area_scale_factor = out_area / in_area

        # 2. Calculate Sigma in Input Pixels
        # Def: 1/e^2 diameter is 4 * sigma
        sigma_m = footprint_diameter_m / 4.0

        avg_input_res = (input_res_m[0] + input_res_m[1]) / 2.0
        sigma_px = sigma_m / avg_input_res

        # 3. Create Base Kernel
        # We use 6*sigma for the kernel size to capture >99% of energy
        # (Since diameter is 4*sigma, this means kernel is 1.5x the footprint size)
        kernel_size = int(math.ceil(6 * sigma_px))
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.register_buffer("kernel", self._create_gaussian_kernel(kernel_size, sigma_px))

        print(f"Model Initialized: In {input_res_m}m -> Out {output_res_m}m")
        print(f"Footprint (1/e^2): {footprint_diameter_m}m (Sigma: {sigma_m:.2f}m / {sigma_px:.2f} px)")

    def _create_gaussian_kernel(self, size, sigma):
        coords = torch.arange(size).float() - (size - 1) / 2
        x_grid, y_grid = torch.meshgrid(coords, coords, indexing='ij')
        kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)

    def forward(self, X_h, altitude=500.0):
        if X_h.ndim == 3:
            X_h = X_h.unsqueeze(0)

        batch_size, num_bins, h_in, w_in = X_h.shape

        # --- 1. Dynamic Output Size ---
        fov_h_m = h_in * self.input_res_m[0]
        fov_w_m = w_in * self.input_res_m[1]

        out_h = int(fov_h_m / self.output_res_m[0])
        out_w = int(fov_w_m / self.output_res_m[1])
        output_size = (out_h, out_w)

        # --- 2. Physics Normalization ---
        energy_per_tube = X_h.sum(dim=1, keepdim=True)
        global_mean_energy = energy_per_tube.mean(dim=(2, 3), keepdim=True)
        X_norm = X_h / (global_mean_energy + 1e-8)

        dist_scale = (self.ref_altitude / altitude) ** 2
        target_intensity = (self.ref_photon_count / self.area_scale_factor) * dist_scale
        X_scaled = X_norm * target_intensity

        # --- 3. Spatial Blurring ---
        # Expand single-channel kernel to match height bins
        current_kernel = self.kernel.repeat(num_bins, 1, 1, 1)

        padding = current_kernel.shape[-1] // 2
        X_blurred = F.conv2d(X_scaled, current_kernel, padding=padding, groups=num_bins)

        # --- 4. Downsampling ---
        X_binned = F.interpolate(X_blurred, size=output_size, mode='area')
        X_integrated = X_binned * self.area_scale_factor

        # --- 5. Noise ---
        lambda_val = torch.relu(X_integrated) + self.b
        X_l = torch.poisson(lambda_val)
        gaussian_noise = torch.randn_like(X_l) * self.eta
        Y_l = X_l + gaussian_noise

        if Y_l.shape[0] == 1:
            Y_l = Y_l.squeeze(0)

        return Y_l
    # In forward_model.py, add this method:

    def forward_noiseless(self, X_h, altitude=500.0):
        """
        Runs the forward physics WITHOUT auto-normalizing the input intensity.
        Assumes X_h is ALREADY in the correct units (e.g. photon counts or proportional energy).
        """
        batch_size, num_bins, h_in, w_in = X_h.shape

        # --- Skip Normalization ---
        # We trust that X_h has the correct magnitude.
        # We might still apply range scaling if X_h is 0-1 but represents 0-50 photons.
        
        # Optional: If Refiner output is 0-1, scale it to max photon count here
        # X_scaled = X_h * self.ref_photon_count 
        # OR if Refiner learns to output photons directly:
        X_scaled = X_h 

        # --- 3. Spatial Blurring (Same as before) ---
        current_kernel = self.kernel.repeat(num_bins, 1, 1, 1)
        padding = current_kernel.shape[-1] // 2
        X_blurred = F.conv2d(X_scaled, current_kernel, padding=padding, groups=num_bins)

        # --- 4. Downsampling (Same as before) ---
        fov_h_m = h_in * self.input_res_m[0]
        fov_w_m = w_in * self.input_res_m[1]
        out_h = int(fov_h_m / self.output_res_m[0])
        out_w = int(fov_w_m / self.output_res_m[1])
        
        X_binned = F.interpolate(X_blurred, size=(out_h, out_w), mode='area')
        X_integrated = X_binned * self.area_scale_factor

        # Return expected photons (Lambda), no Poisson sampling
        return torch.relu(X_integrated) + self.b
    