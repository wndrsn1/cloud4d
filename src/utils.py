"""
Utility functions for cloud volume prediction.

Contains helper functions for:
- Timestep embeddings and modulation (from DiT)
- Camera projection and ray computation
- Normalization layers (from ConvNeXt)
- Positional encodings
- Video/image I/O utilities
- Visualization utilities
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale


# =============================================================================
# DiT-style modulation and embeddings
# =============================================================================

def modulate(x, shift, scale):
    """Apply adaptive layer norm modulation (from DiT).

    Args:
        x: Input tensor of shape (B, C, H, W) or similar
        shift: Shift parameter of shape (B, C)
        scale: Scale parameter of shape (B, C)

    Returns:
        Modulated tensor: x * (1 + scale) + shift
    """
    return x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps/heights into vector representations using sinusoidal embeddings.

    Used for conditioning models on continuous values like height or timestep.

    Args:
        hidden_size: Output dimension of the embedding
        frequency_embedding_size: Dimension of the sinusoidal embedding (default: 256)
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings.

        Args:
            t: 1-D Tensor of N indices (may be fractional)
            dim: Output dimension
            max_period: Controls the minimum frequency of the embeddings

        Returns:
            (N, D) Tensor of positional embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        """Embed timestep/height values."""
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# =============================================================================
# Camera projection and ray computation
# =============================================================================

def depth_to_heights(depths, intrinsics, extrinsics, xy_bound=5000):
    """Convert depth maps to height maps in world coordinates.

    Args:
        depths: Depth maps of shape (B, num_cams, H, W)
        intrinsics: Camera intrinsics of shape (B, num_cams, 3, 3)
        extrinsics: Camera extrinsics (cam2world) of shape (B, num_cams, 4, 4)
        xy_bound: Bound for valid XY coordinates

    Returns:
        height_maps: Height values in world coordinates (B, num_cams, H, W)
        valid_within_xy_bound: Boolean mask for valid points within XY bounds
    """
    batch_size, num_cams, h, w = depths.shape
    focal_lengths = intrinsics[0, :, 0, 0]

    all_rays_o = []
    all_rays_d = []

    for i in range(num_cams):
        ray_o, ray_d = get_rays(extrinsics[0, i], h, w, focal_lengths[i], opengl=False)
        all_rays_o.append(ray_o.reshape(h, w, 3))
        all_rays_d.append(ray_d.reshape(h, w, 3))

    # Stack and reshape: (num_cams, h, w, 3) -> (3, num_cams, h, w)
    all_rays_o = torch.stack(all_rays_o, dim=0).permute(3, 0, 1, 2)
    all_rays_d = torch.stack(all_rays_d, dim=0).permute(3, 0, 1, 2)

    # Expand to batch dimension
    all_rays_o = all_rays_o.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
    all_rays_d = all_rays_d.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

    # Compute world coordinates
    world_xyz = depths.unsqueeze(1) * all_rays_d + all_rays_o
    height_maps = world_xyz[:, -1, :, :, :]

    # Filter points within XY bounds
    valid_within_xy_bound = (
        (world_xyz[:, 0, ...] >= -xy_bound) & (world_xyz[:, 0, ...] <= xy_bound) &
        (world_xyz[:, 1, ...] >= -xy_bound) & (world_xyz[:, 1, ...] <= xy_bound)
    )

    return height_maps, valid_within_xy_bound    

# =============================================================================
# Normalization layers
# =============================================================================

class LayerNorm(nn.Module):
    """LayerNorm supporting channels_last and channels_first data formats.

    From: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    Args:
        normalized_shape: Input shape from an expected input
        eps: Small constant for numerical stability
        data_format: 'channels_last' (B, H, W, C) or 'channels_first' (B, C, H, W)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
def get_rays(pose, h, w, focal, opengl=True):
    """Compute ray origins and directions for a camera.

    From: https://github.com/alibaba-yuanjing-aigclab/GeoLRM

    Args:
        pose: Camera-to-world transform (4, 4)
        h: Image height
        w: Image width
        focal: Focal length
        opengl: Use OpenGL coordinate convention if True

    Returns:
        rays_o: Ray origins of shape (h*w, 3)
        rays_d: Ray directions of shape (h*w, 3)
    """
    device = focal.device

    x, y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
    x = x.flatten().to(device)
    y = y.flatten().to(device)

    cx, cy = w * 0.5, h * 0.5
    y_sign = -1.0 if opengl else 1.0
    z_sign = -1.0 if opengl else 1.0

    camera_dirs = F.pad(
        torch.stack([
            (x - cx + 0.5) / focal,
            (y - cy + 0.5) / focal * y_sign,
        ], dim=-1),
        (0, 1),
        value=z_sign,
    )

    rays_d = camera_dirs @ pose[:3, :3].transpose(0, 1)
    rays_o = pose[:3, 3].unsqueeze(0).expand_as(rays_d)

    return rays_o, rays_d


def get_ray_embedding(input_Ks, input_c2ws, h, w):
    """Compute Plucker ray embeddings from camera parameters.

    Plucker coordinates encode rays using direction and moment (origin x direction).

    Args:
        input_Ks: Camera intrinsics of shape (B, 3, 3), normalized by image size
        input_c2ws: Camera-to-world transforms of shape (B, 4, 4)
        h: Image height
        w: Image width

    Returns:
        Ray embeddings of shape (B, 6, h, w) containing (ray_d, ray_o x ray_d)
    """
    # Scale intrinsics from normalized to pixel coordinates
    input_Ks[:, 0] = input_Ks[:, 0] * w
    input_Ks[:, 1] = input_Ks[:, 1] * h

    B = input_Ks.shape[0]
    rays_o, rays_d = [], []
    for b in range(B):
        rays_o_b, rays_d_b = get_rays(input_c2ws[b], h, w, input_Ks[b, 0, 0])
        rays_o.append(rays_o_b)
        rays_d.append(rays_d_b)

    rays_o = torch.stack(rays_o)
    rays_d = torch.stack(rays_d)

    # Plucker coordinates: (direction, moment)
    ray_embedding = torch.cat([rays_d, torch.cross(rays_o, rays_d, dim=-1)], dim=-1)
    return rearrange(ray_embedding, 'b (h w) c -> b c h w', h=h, w=w)


def project_volume_to_cameras(grid_shape, cam2world, camera_intrinsics,
                               grid_ranges=[(-5000, 5000), (-5000, 5000), (0, 4000)], voxel_size=50):
    """Project a 3D volume grid to camera pixel coordinates.

    Args:
        grid_shape: Shape of the 3D grid (x_dim, y_dim, z_dim)
        cam2world: Camera-to-world transform (4, 4)
        camera_intrinsics: Camera intrinsic matrix (3, 3)
        grid_ranges: World coordinate ranges for each dimension
        voxel_size: Size of each voxel in world units

    Returns:
        pixel_coords: Normalized pixel coordinates in [-1, 1] of shape (X, Y, Z, 2)
        valid_coords: Boolean mask for valid projections of shape (X, Y, Z)
        euclidean_depths: Euclidean distances from camera of shape (X, Y, Z, 1)
    """
    grid_ranges = torch.tensor(grid_ranges)

    # Create voxel grid
    voxel_indices = torch.meshgrid([
        torch.arange(0, grid_shape[0], dtype=torch.float32),
        torch.arange(0, grid_shape[1], dtype=torch.float32),
        torch.arange(0, grid_shape[2], dtype=torch.float32)
    ])
    voxel_indices = torch.stack(voxel_indices, dim=-1)

    # Convert to world coordinates (centered at voxel centers)
    world_coords = voxel_indices * voxel_size + grid_ranges[:, 0] + voxel_size / 2

    # Add homogeneous coordinate
    world_coords = torch.cat([world_coords, torch.ones(world_coords.shape[:-1] + (1,))], dim=-1)

    # Transform to camera coordinates
    cam_coords = torch.einsum('ji, xyzi -> xyzj', torch.linalg.inv(cam2world), world_coords)
    cam_coords = cam_coords[:, :, :, :3] / cam_coords[:, :, :, 3:]

    euclidean_depths = torch.sqrt(torch.sum(cam_coords ** 2, dim=-1)).unsqueeze(-1)

    # Project to pixel coordinates
    pixel_coords = torch.einsum('ji, xyzi -> xyzj', camera_intrinsics, cam_coords)
    pixel_coords = pixel_coords[:, :, :, :2] / pixel_coords[:, :, :, 2:]

    # Image dimensions from intrinsics
    w, h = camera_intrinsics[0, 2] * 2, camera_intrinsics[1, 2] * 2

    # Determine valid projections
    valid_coords = (
        (pixel_coords[..., 0] >= 0) & (pixel_coords[..., 0] < w) &
        (pixel_coords[..., 1] >= 0) & (pixel_coords[..., 1] < h)
    )

    # Normalize to [-1, 1] for grid_sample
    pixel_coords = pixel_coords / torch.tensor([w, h])
    pixel_coords = pixel_coords * 2 - 1

    return pixel_coords, valid_coords, euclidean_depths


def debug_projection(volume_grid, pixel_coords, output_img_shape=(640, 480), cam_idx=0):
    """Project a 3D volume onto a 2D image for visualization.

    Accumulates volume values at projected pixel locations.

    Args:
        volume_grid: 3D volume tensor
        pixel_coords: Normalized pixel coordinates from project_volume_to_cameras
        output_img_shape: Output image size (width, height)
        cam_idx: Camera index to use for projection

    Returns:
        2D image with accumulated volume values
    """
    volume_grid = volume_grid.squeeze().to(torch.float32)

    # Scale normalized coords [-1, 1] back to pixel coordinates
    pixel_coords = pixel_coords[cam_idx]
    pixel_coords = (pixel_coords + 1) / 2
    pixel_coords[:, :, :, 0] *= output_img_shape[0]
    pixel_coords[:, :, :, 1] *= output_img_shape[1]

    output_image = torch.zeros((output_img_shape[1], output_img_shape[0]), dtype=torch.float32, device=volume_grid.device)

    pixel_indices = torch.round(pixel_coords).to(torch.int16)
    valid_indices = (
        (pixel_indices[..., 0] >= 0) & (pixel_indices[..., 0] < output_img_shape[0]) &
        (pixel_indices[..., 1] >= 0) & (pixel_indices[..., 1] < output_img_shape[1])
    )

    # Zero out invalid indices and values
    pixel_indices[~valid_indices] = 0
    volume_grid[~valid_indices] = 0

    # Vectorized accumulation
    pixel_indices = pixel_indices.reshape(-1, 2).to(torch.int32)
    output_image = torch.index_put(output_image, (pixel_indices[:, 1], pixel_indices[:, 0]), volume_grid.reshape(-1), accumulate=True)

    return output_image




# =============================================================================
# Visualization utilities
# =============================================================================

def _volume_to_orthographic(volume):
    """Compute orthographic projections (mean along each axis) of a 3D volume."""
    z_sum = torch.mean(volume, dim=-1)
    y_sum = torch.mean(volume, dim=-2)
    x_sum = torch.mean(volume, dim=-3)
    return x_sum, y_sum, z_sum


def _to_numpy_flipped(tensor):
    """Convert tensor to numpy array, transpose and flip for visualization."""
    return np.flip(tensor.float().squeeze().cpu().numpy().T, axis=0)


def _to_numpy(tensor):
    """Convert tensor to numpy array for visualization."""
    return tensor.float().squeeze().cpu().numpy()


def _create_color_mask(projection, color_channel_to_keep):
    """Create a colored mask from a projection for overlay visualization.

    Args:
        projection: 2D numpy array
        color_channel_to_keep: 0 for red, 1 for green, 2 for blue
    """
    colored = torch.ones((*projection.shape, 3))
    mask = projection > 0
    for channel in range(3):
        if channel != color_channel_to_keep:
            colored[:, :, channel][mask] = 0
    return colored


@torch.no_grad()
def get_orthographic_sum_figure(pred_vol, gt_vol, coarse_cloud=None):
    """Create orthographic projection figure comparing predicted and ground truth volumes."""
    x_sum_pred, y_sum_pred, z_sum_pred = _volume_to_orthographic(pred_vol)
    x_sum_gt, y_sum_gt, z_sum_gt = _volume_to_orthographic(gt_vol)

    max_x = torch.max(x_sum_gt)
    max_y = torch.max(y_sum_gt)
    max_z = torch.max(z_sum_gt)

    num_rows = 2 if coarse_cloud is None else 4
    fig, axs = plt.subplots(num_rows, 3)

    # Prediction row
    axs[0, 0].imshow(_to_numpy_flipped(x_sum_pred), vmin=0, vmax=max_x * 2)
    axs[0, 0].set_title('X Pred')
    axs[0, 1].imshow(_to_numpy_flipped(y_sum_pred), vmin=0, vmax=max_y * 2)
    axs[0, 1].set_title('Y Pred')
    axs[0, 2].imshow(_to_numpy(z_sum_pred), vmin=0, vmax=max_z * 2)
    axs[0, 2].set_title('Z Pred')

    # Ground truth row
    axs[1, 0].imshow(_to_numpy_flipped(x_sum_gt), vmin=0, vmax=max_x * 2)
    axs[1, 0].set_title('X GT')
    axs[1, 1].imshow(_to_numpy_flipped(y_sum_gt), vmin=0, vmax=max_y * 2)
    axs[1, 1].set_title('Y GT')
    axs[1, 2].imshow(_to_numpy(z_sum_gt), vmin=0, vmax=max_z * 2)
    axs[1, 2].set_title('Z GT')

    if coarse_cloud is not None:
        x_sum_coarse, y_sum_coarse, z_sum_coarse = _volume_to_orthographic(coarse_cloud)

        # Coarse prediction row
        axs[2, 0].imshow(_to_numpy_flipped(x_sum_coarse), vmin=0, vmax=max_x * 2)
        axs[2, 0].set_title('X Coarse')
        axs[2, 1].imshow(_to_numpy_flipped(y_sum_coarse), vmin=0, vmax=max_y * 2)
        axs[2, 1].set_title('Y Coarse')
        axs[2, 2].imshow(_to_numpy(z_sum_coarse), vmin=0, vmax=max_z * 2)
        axs[2, 2].set_title('Z Coarse')

        # Alignment comparison row (coarse in red, GT in green)
        x_coarse_np = _to_numpy_flipped(x_sum_coarse)
        y_coarse_np = _to_numpy_flipped(y_sum_coarse)
        z_coarse_np = _to_numpy(z_sum_coarse)
        x_gt_np = _to_numpy_flipped(x_sum_gt)
        y_gt_np = _to_numpy_flipped(y_sum_gt)
        z_gt_np = _to_numpy(z_sum_gt)

        axs[3, 0].imshow(_create_color_mask(x_coarse_np, 0), alpha=0.5)
        axs[3, 0].imshow(_create_color_mask(x_gt_np, 1), alpha=0.5)
        axs[3, 0].set_title('X Coarse vs GT')
        axs[3, 1].imshow(_create_color_mask(y_coarse_np, 0), alpha=0.5)
        axs[3, 1].imshow(_create_color_mask(y_gt_np, 1), alpha=0.5)
        axs[3, 1].set_title('Y Coarse vs GT')
        axs[3, 2].imshow(_create_color_mask(z_coarse_np, 0), alpha=0.5)
        axs[3, 2].imshow(_create_color_mask(z_gt_np, 1), alpha=0.5)
        axs[3, 2].set_title('Z Coarse vs GT')

    return fig


@torch.no_grad()
def get_projection_figure(batch, visualisation_volume, projection_indices, right_projection_indices, cam_indices, fig_size=None, only_left=False):
    """Create a figure showing volume projections overlaid on camera images."""
    left_images = batch['left_images'].squeeze(0)
    right_images = batch['right_images'].squeeze(0)
    num_views, channels, height, width = left_images.shape

    depths = batch['depths'].squeeze(0)
    volumes = visualisation_volume.squeeze(0)

    if only_left:
        fig, axs = plt.subplots(num_views, 2)
    else:
        fig, axs = plt.subplots(num_views, 5)

    if fig_size is not None:
        px = 1 / plt.rcParams['figure.dpi']
        fig.set_size_inches(fig_size[0] * px, fig_size[1] * px)

    for plot_idx, i in enumerate(cam_indices):
        left_projected_image = debug_projection(torch.clone(volumes), torch.clone(projection_indices), cam_idx=i, output_img_shape=(width, height))
        right_projected_image = debug_projection(torch.clone(volumes), torch.clone(right_projection_indices), cam_idx=i, output_img_shape=(width, height))
        curr_axs1 = axs[plot_idx, 0] if num_views > 1 else axs[0]
        curr_axs2 = axs[plot_idx, 1] if num_views > 1 else axs[1]
        curr_axs1.imshow((left_projected_image > 1e-6).cpu().numpy())
        curr_axs2.imshow(left_images[plot_idx].permute(1, 2, 0).cpu().numpy(), alpha=1)

        if not only_left:
            curr_axs3 = axs[plot_idx, 2] if num_views > 1 else axs[2]
            curr_axs4 = axs[plot_idx, 3] if num_views > 1 else axs[3]
            curr_axs5 = axs[plot_idx, 4] if num_views > 1 else axs[4]


            curr_axs3.imshow((right_projected_image > 1e-6).cpu().numpy(), interpolation='none')
            curr_axs4.imshow(right_images[plot_idx].permute(1, 2, 0).cpu().numpy())
            curr_axs5.imshow(depths[plot_idx].cpu().numpy(), vmin=0, vmax=10000)

    return fig


@torch.no_grad()
def get_height_pred_figure(cloud_base_heights, delta_heights_pred, occupancy_logits, gt_volumes, sampled_feature_height_slice=None, voxel_size=50):
    """Create a figure comparing predicted and ground truth height properties."""
    z_dim = gt_volumes.shape[-1]
    grid_heights = torch.arange(z_dim) * voxel_size   # Start height is 0.

    grid_heights = grid_heights.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(gt_volumes.device)
    binary_gt = (gt_volumes > 0)

    valid_heights = (grid_heights * binary_gt).unsqueeze(0)
    for_min_height = valid_heights.clone()
    for_min_height[~binary_gt.unsqueeze(0)] = 20000 # Set to a high value to not affect the min

    gt_cbh = torch.min(for_min_height, dim=-1)[0]
    gt_cbh[gt_cbh > 8000] = 0
    gt_delta_heights = (torch.max(valid_heights, dim=-1)[0] - gt_cbh)

    gt_occupancy = (gt_cbh > 0)

    # For visualisation
    gt_cbh[~gt_occupancy] = 0

    if sampled_feature_height_slice is not None:
        fig, axs = plt.subplots(3, 4)
    else:
        fig, axs = plt.subplots(2, 2)

    axs[0, 0].imshow((cloud_base_heights).float().squeeze().cpu().numpy(), vmin=0.5, vmax=3)
    axs[0, 0].set_title('CBH Pred')

    axs[1, 0].imshow(gt_cbh.float().squeeze().cpu().numpy(), vmin=500, vmax=3000)
    axs[1, 0].set_title('CBH GT')

    axs[0, 1].imshow(delta_heights_pred.float().squeeze().cpu().numpy(), vmin=0, vmax=1)
    axs[0, 1].set_title('Delta Heights Pred')

    axs[1, 1].imshow(gt_delta_heights.float().squeeze().cpu().numpy(), vmin=0, vmax=1000)
    axs[1, 1].set_title('Delta Heights GT')

    if sampled_feature_height_slice is not None:
        _, feature_dim_multiple_slices, x_dim, y_dim = sampled_feature_height_slice.shape
        feature_dim = 384

        num_slices = feature_dim_multiple_slices // feature_dim
        assert feature_dim_multiple_slices % feature_dim == 0, "Feature dimension is not a multiple of the number of slices"

        for i in range(num_slices):
            curr_slice = sampled_feature_height_slice[:, i * feature_dim:(i + 1) * feature_dim, :, :].squeeze()
            if feature_dim == 3:
                axs[2, i].imshow(curr_slice.permute(1, 2, 0).cpu().numpy())
            else:
                cloud_pca = PCA(n_components=3).fit_transform(curr_slice.squeeze().reshape(feature_dim, -1).permute((1, 0)).cpu().numpy())
                cloud_pca = minmax_scale(cloud_pca)
                axs[2, i].imshow(cloud_pca.reshape(x_dim, y_dim, 3))
            axs[2, i].set_title(f'Height Slice {i} PCA')

    fig.tight_layout()
    return fig


def _move_tensors_to_device(tensors_dict, device):
    """Move a dictionary of tensors to the specified device."""
    return {k: v.to(device) for k, v in tensors_dict.items()}
