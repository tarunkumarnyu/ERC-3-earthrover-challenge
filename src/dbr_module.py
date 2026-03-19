"""
Depth Barrier Regularization (DBR) Module

Implements the DBR training loss from the paper:
"Depth Barrier Regularization for Safe Vision Based Navigation"

This module provides:
    - DepthPolarReducer: Converts depth maps to polar clearance vectors
    - BarrierLoss: Penalizes waypoints in unsafe directions
    - DBRModule: Complete DBR pipeline for training

Usage:
    from dbr_module import DBRModule

    dbr = DBRModule(num_bins=32, margin=0.5, temperature=20.0)
    loss, clearance = dbr(predicted_waypoints, depth_map)

Author: Vivek Mattam
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class DepthPolarReducer(nn.Module):
    """
    Converts depth maps to polar clearance vectors.

    Takes a (B, H, W) depth map and produces (B, num_bins) clearance values,
    where each bin represents the minimum depth in that angular direction.

    Uses soft-min (LogSumExp) for differentiability.
    """

    def __init__(self, num_bins=32, temperature=20.0, crop_bottom_ratio=0.6,
                 fov_horizontal=90.0, image_width=640, image_height=480,
                 fx=None, fy=None, cx=None, cy=None):
        """
        Args:
            num_bins: Number of angular bins (default 32)
            temperature: Soft-min temperature κ (default 20.0, higher=sharper)
            crop_bottom_ratio: Keep bottom X% of image (default 0.6 = bottom 60%)
            fov_horizontal: Horizontal field of view in degrees
            image_width: Image width in pixels
            image_height: Image height in pixels
            fx, fy, cx, cy: Camera intrinsics (auto-computed if None)
        """
        super().__init__()

        self.num_bins = num_bins
        self.temperature = temperature
        self.crop_bottom_ratio = crop_bottom_ratio
        self.fov_horizontal = math.radians(fov_horizontal)

        # Camera intrinsics
        if fx is None:
            fx = image_width / (2.0 * math.tan(self.fov_horizontal / 2.0))
        if cx is None:
            cx = image_width / 2.0

        self.register_buffer('fx', torch.tensor(fx, dtype=torch.float32))
        self.register_buffer('cx', torch.tensor(cx, dtype=torch.float32))

        # Precompute bin edges and centers
        bin_edges = torch.linspace(-self.fov_horizontal / 2, self.fov_horizontal / 2, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_width = bin_edges[1] - bin_edges[0]

        self.register_buffer('bin_edges', bin_edges)
        self.register_buffer('bin_centers', bin_centers)
        self.register_buffer('bin_width', torch.tensor(bin_width))

        # Precompute per-pixel yaw angles for the given image size
        crop_start = int(image_height * (1.0 - crop_bottom_ratio))
        crop_height = image_height - crop_start

        u = torch.arange(image_width, dtype=torch.float32)
        pixel_yaw = torch.atan((u - cx) / fx)  # (W,)

        # Compute bin membership weights (triangular kernel)
        # Shape: (num_bins, W)
        # For each bin, weight = 1 - |pixel_yaw - bin_center| / bin_width
        # Clamped to [0, 1]
        pixel_yaw_expanded = pixel_yaw.unsqueeze(0)  # (1, W)
        bin_centers_expanded = bin_centers.unsqueeze(1)  # (num_bins, 1)

        weights = 1.0 - torch.abs(pixel_yaw_expanded - bin_centers_expanded) / bin_width
        weights = torch.clamp(weights, min=0.0)  # (num_bins, W)

        # Expand to (num_bins, crop_height, W) - same weight for all rows
        weights = weights.unsqueeze(1).expand(-1, crop_height, -1)

        self.register_buffer('bin_weights', weights)  # (num_bins, H_crop, W)
        self.crop_start = crop_start

    def forward(self, depth_map, depth_mask=None):
        """
        Convert depth map to polar clearance.

        Args:
            depth_map: (B, H, W) depth in meters
            depth_mask: (B, H, W) optional mask (1=valid, 0=invalid)

        Returns:
            clearance: (B, num_bins) clearance per direction in meters
        """
        B, H, W = depth_map.shape

        # Crop bottom portion
        depth_cropped = depth_map[:, self.crop_start:, :]  # (B, H_crop, W)

        if depth_mask is not None:
            mask_cropped = depth_mask[:, self.crop_start:, :]
        else:
            mask_cropped = (depth_cropped > 0).float()

        # Apply mask: set invalid pixels to large depth (won't affect min)
        depth_masked = depth_cropped * mask_cropped + (1 - mask_cropped) * 100.0

        # Soft-min using LogSumExp trick
        # r_b = -(1/κ) * log(Σ exp(-κ * D) * w_b)
        #
        # For numerical stability:
        # r_b = -(1/κ) * log(Σ exp(-κ * D + log(w_b + eps)))

        kappa = self.temperature
        clearance = torch.zeros(B, self.num_bins, device=depth_map.device)

        for b in range(self.num_bins):
            # Get weights for this bin: (H_crop, W)
            w = self.bin_weights[b]  # (H_crop, W)

            # Skip bins with no contributing pixels
            if w.sum() < 1e-6:
                clearance[:, b] = 100.0
                continue

            # Compute weighted soft-min
            # exp(-κ * D) * w, summed over spatial dimensions
            log_w = torch.log(w + 1e-10)  # (H_crop, W)

            # (B, H_crop, W)
            exponent = -kappa * depth_masked + log_w.unsqueeze(0)

            # LogSumExp over spatial dims for numerical stability
            max_exp = exponent.view(B, -1).max(dim=1, keepdim=True)[0]  # (B, 1)
            max_exp = max_exp.unsqueeze(-1)  # (B, 1, 1)

            sum_exp = (torch.exp(exponent - max_exp) * mask_cropped).sum(dim=[1, 2])  # (B,)
            log_sum = max_exp.squeeze() + torch.log(sum_exp + 1e-10)  # (B,)

            clearance[:, b] = -log_sum / kappa

        return clearance


class BarrierLoss(nn.Module):
    """
    Barrier loss that penalizes waypoints in unsafe directions.

    L_DBR = (1/T) * Σ softplus(τ - d_min(t))

    Where:
        τ = safety margin (meters)
        d_min(t) = clearance in waypoint t's direction
    """

    def __init__(self, margin=0.5):
        """
        Args:
            margin: Safety margin τ in meters (default 0.5)
        """
        super().__init__()
        self.margin = margin

    def forward(self, waypoints, clearance, bin_centers):
        """
        Compute barrier loss.

        Args:
            waypoints: (B, T, 2) predicted waypoints in ego frame
            clearance: (B, num_bins) clearance per direction
            bin_centers: (num_bins,) center angles of each bin

        Returns:
            loss: scalar barrier loss
        """
        B, T, _ = waypoints.shape
        num_bins = clearance.shape[1]

        # Compute yaw angle for each waypoint
        # φ_t = arctan2(y_t, x_t)
        wp_yaw = torch.atan2(waypoints[:, :, 1], waypoints[:, :, 0])  # (B, T)

        # Interpolate clearance at each waypoint's direction using soft attention
        # Weight = triangular kernel: 1 - |wp_yaw - bin_center| / bin_width
        bin_width = bin_centers[1] - bin_centers[0] if num_bins > 1 else 1.0

        # (B, T, 1) vs (1, 1, num_bins) -> (B, T, num_bins)
        wp_yaw_expanded = wp_yaw.unsqueeze(-1)  # (B, T, 1)
        centers_expanded = bin_centers.unsqueeze(0).unsqueeze(0)  # (1, 1, num_bins)

        # Attention weights (triangular kernel)
        attn = 1.0 - torch.abs(wp_yaw_expanded - centers_expanded) / bin_width
        attn = torch.clamp(attn, min=0.0)  # (B, T, num_bins)

        # Normalize weights
        attn_sum = attn.sum(dim=-1, keepdim=True) + 1e-10
        attn = attn / attn_sum  # (B, T, num_bins)

        # Interpolate clearance at waypoint directions
        # (B, T, num_bins) * (B, 1, num_bins) -> sum -> (B, T)
        clearance_expanded = clearance.unsqueeze(1)  # (B, 1, num_bins)
        d_min = (attn * clearance_expanded).sum(dim=-1)  # (B, T)

        # Barrier loss: softplus(τ - d_min)
        violation = self.margin - d_min  # Positive when unsafe
        loss = F.softplus(violation)  # (B, T)

        # Average over waypoints and batch
        return loss.mean()


class DBRModule(nn.Module):
    """
    Complete DBR module combining depth reduction and barrier loss.

    Usage during training:
        dbr = DBRModule()
        loss, clearance = dbr(predicted_waypoints, depth_map)
        total_loss = citywalker_loss + loss_weight * loss
    """

    def __init__(self, num_bins=32, temperature=20.0, crop_bottom_ratio=0.6,
                 fov_horizontal=90.0, margin=0.5, loss_weight=1.0,
                 image_width=640, image_height=480):
        """
        Args:
            num_bins: Angular bins for polar clearance
            temperature: Soft-min temperature (κ=20)
            crop_bottom_ratio: Bottom portion of image to use
            fov_horizontal: Camera horizontal FOV in degrees
            margin: Safety margin τ in meters
            loss_weight: λ_bar weight for DBR loss
            image_width: Input image width
            image_height: Input image height
        """
        super().__init__()

        self.loss_weight = loss_weight

        self.reducer = DepthPolarReducer(
            num_bins=num_bins,
            temperature=temperature,
            crop_bottom_ratio=crop_bottom_ratio,
            fov_horizontal=fov_horizontal,
            image_width=image_width,
            image_height=image_height
        )

        self.barrier = BarrierLoss(margin=margin)

    def forward(self, waypoints, depth_map, depth_mask=None):
        """
        Compute DBR loss.

        Args:
            waypoints: (B, T, 2) predicted waypoints in meters
            depth_map: (B, H, W) depth in meters
            depth_mask: (B, H, W) optional validity mask

        Returns:
            loss: weighted barrier loss (scalar)
            clearance: (B, num_bins) clearance per direction
        """
        # Step 1: Convert depth to polar clearance
        clearance = self.reducer(depth_map, depth_mask)

        # Step 2: Compute barrier loss
        loss = self.barrier(waypoints, clearance, self.reducer.bin_centers)

        # Apply weight
        weighted_loss = self.loss_weight * loss

        return weighted_loss, clearance


# Quick test if run directly
if __name__ == "__main__":
    print("Testing DBR Module...")
    print("=" * 60)

    # Create module
    dbr = DBRModule(num_bins=32, margin=0.5, temperature=20.0)

    # Create dummy data
    B, T = 2, 5  # batch=2, waypoints=5
    H, W = 480, 640

    dummy_waypoints = torch.randn(B, T, 2) * 2.0  # Random waypoints
    dummy_depth = torch.rand(B, H, W) * 10.0 + 0.5  # Random depths 0.5-10.5m

    print(f"\n[1] Input shapes:")
    print(f"  Waypoints: {dummy_waypoints.shape}")
    print(f"  Depth map: {dummy_depth.shape}")

    # Compute DBR loss
    print(f"\n[2] Computing DBR loss...")
    loss, clearance = dbr(dummy_waypoints, dummy_depth)

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Clearance shape: {clearance.shape}")
    print(f"  Clearance range: [{clearance.min():.2f}, {clearance.max():.2f}] meters")

    # Test with unsafe waypoints (pointing at close obstacles)
    print(f"\n[3] Testing with unsafe waypoints...")
    # Create depth map with obstacle directly ahead
    depth_with_obstacle = torch.ones(B, H, W) * 5.0  # 5m everywhere
    depth_with_obstacle[:, 300:, 280:360] = 0.3  # Obstacle ahead at 0.3m

    safe_waypoints = torch.tensor([[[1.0, 2.0]] * T] * B)  # Point left (safe)
    unsafe_waypoints = torch.tensor([[[1.0, 0.0]] * T] * B)  # Point ahead (unsafe)

    loss_safe, _ = dbr(safe_waypoints, depth_with_obstacle)
    loss_unsafe, _ = dbr(unsafe_waypoints, depth_with_obstacle)

    print(f"  Safe waypoints loss: {loss_safe.item():.4f}")
    print(f"  Unsafe waypoints loss: {loss_unsafe.item():.4f}")
    print(f"  Unsafe > Safe: {loss_unsafe.item() > loss_safe.item()}")

    # Test gradient flow
    print(f"\n[4] Testing gradient flow...")
    waypoints_grad = torch.randn(B, T, 2, requires_grad=True)
    loss_grad, _ = dbr(waypoints_grad, dummy_depth)
    loss_grad.backward()
    print(f"  Gradient shape: {waypoints_grad.grad.shape}")
    print(f"  Gradient norm: {waypoints_grad.grad.norm():.4f}")
    print(f"  Gradients flow: {waypoints_grad.grad.norm() > 0}")

    print("\nDBR Module test passed!")
