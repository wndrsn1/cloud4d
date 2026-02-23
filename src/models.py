"""
Cloud Volume Prediction Models.

This module implements a multi-stage model for predicting 3D cloud volumes
from multi-view stereo images:

- Stage 1: 2D cloud property prediction (cloud base height, thickness, LWP)
- Stage 2: 3D volume refinement using sparse 3D CNNs

The model uses DINOv2 features with LoftUp upsampling for rich visual features.
"""

import math
import torch
import numpy as np

from sparse_cnn import Sparse3DCNN
from sparse import SparseTensor
from torchvision import transforms

from utils import TimestepEmbedder, modulate
from unet import UNet

def make_binary_cloud(lwp, occupancy_logits, cloud_base_heights,
                      delta_heights_pred, x_dim=200, y_dim=200, z_dim=80, start_height=0, voxel_size=50):
    """Create a coarse 3D cloud volume from 2D cloud property predictions.

    Constructs a 3D cloud field by distributing liquid water content (LWC)
    vertically between the predicted cloud base and top heights, assuming
    an adiabatic (linear) LWC profile.

    Args:
        lwp: Liquid water path predictions (B, 1, X, Y)
        occupancy_logits: Cloud occupancy predictions (B, 1, X, Y)
        cloud_base_heights: Cloud base height predictions in km (B, 1, X, Y)
        delta_heights_pred: Cloud thickness predictions in km (B, 1, X, Y)
        x_dim: X dimension of output volume
        y_dim: Y dimension of output volume
        z_dim: Z dimension of output volume
        start_height: Starting height in meters
        voxel_size: Voxel size in meters

    Returns:
        Coarse cloud volume of shape (B, 1, X, Y, Z) with LWC values
    """
    batch_size = lwp.shape[0]
    cloud_mask = torch.nn.functional.sigmoid(occupancy_logits) > 0.5

    # Convert from km to m
    cloud_base_heights = cloud_base_heights * 1000
    delta_heights_pred = delta_heights_pred * 1000

    device = cloud_base_heights.device
    dtype = lwp.dtype

    # Ensure non-negative predictions
    cloud_base_heights = torch.clamp(cloud_base_heights, min=0.0)
    delta_heights_pred = torch.clamp(delta_heights_pred, min=0.0)
    lwp = torch.clamp(lwp, min=0.0)

    # Round heights to nearest voxel
    cloud_base_heights = torch.round(cloud_base_heights / voxel_size) * voxel_size
    delta_heights_pred = torch.round(delta_heights_pred / voxel_size) * voxel_size

    with torch.no_grad():
        xy_indices = torch.meshgrid([
            torch.arange(0, x_dim, dtype=torch.int, device=device),
            torch.arange(0, y_dim, dtype=torch.int, device=device)
        ])

        coarse_clouds = []
        for i in range(batch_size):
            cbh_height_indices = ((cloud_base_heights[i].flatten() - start_height) / voxel_size).int()
            valid_indices = (cbh_height_indices >= 0) & (cbh_height_indices < z_dim) & (cloud_mask[i].flatten() > 0.5)

            bottom_layer = torch.zeros(1, 1, x_dim, y_dim, z_dim, device=device, dtype=dtype)
            bottom_layer[:, :, xy_indices[0].flatten()[valid_indices], xy_indices[1].flatten()[valid_indices], cbh_height_indices[valid_indices]] = 1
            bottom_layer = torch.cumsum(bottom_layer, dim=-1)

            cloud_top_indices = ((cloud_base_heights[i].flatten() + delta_heights_pred[i].flatten() - start_height) / voxel_size).int()
            cloud_top_indices[cloud_top_indices >= z_dim] = z_dim - 1

            top_layer = torch.zeros(1, 1, x_dim, y_dim, z_dim, device=device, dtype=dtype)
            top_layer[:, :, xy_indices[0].flatten()[valid_indices], xy_indices[1].flatten()[valid_indices], cloud_top_indices[valid_indices]] = 1
            top_layer = torch.cumsum(top_layer, dim=-1)

            curr_cloud = bottom_layer - top_layer

            # Distribute the cloud based on the LWP (convert from kg/m^2 to kg/m^3)
            cloud_mask_3d = curr_cloud > 0
            lwp_3d = lwp[i].unsqueeze(0).unsqueeze(-1).repeat((1, 1, 1, 1, z_dim))
            delta_heights_pred_3d = delta_heights_pred[i].unsqueeze(0).unsqueeze(-1).repeat((1, 1, 1, 1, z_dim))

            curr_cloud[cloud_mask_3d] = lwp_3d[cloud_mask_3d] / delta_heights_pred_3d[cloud_mask_3d]

            # Assume adiabatic profile: linearly distribute the LWC
            cloud_base_heights_3d = cloud_base_heights[i].unsqueeze(0).unsqueeze(-1).repeat((1, 1, 1, 1, z_dim))
            grid_heights = torch.arange(start_height, start_height + z_dim * voxel_size, step=voxel_size, device=device)
            grid_heights = grid_heights.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0) + voxel_size / 2
            grid_heights = grid_heights.repeat((1, 1, x_dim, y_dim, 1))
            height_from_base = grid_heights - cloud_base_heights_3d
            curr_cloud[cloud_mask_3d] = 2 * curr_cloud[cloud_mask_3d] * height_from_base[cloud_mask_3d] / delta_heights_pred_3d[cloud_mask_3d]

            coarse_clouds.append(curr_cloud)

        coarse_clouds = torch.cat(coarse_clouds, dim=0)
        coarse_clouds = torch.clamp(coarse_clouds, min=0.0)

        return coarse_clouds


class Cloud4D(torch.nn.Module):
    """Multi-stage model for 3D cloud volume prediction from stereo images.

    The model consists of two stages that can be trained separately:
    - Stage 1: 2D cloud property prediction (CBH, thickness, LWP)
    - Stage 2: 3D volume refinement using sparse 3D CNNs
    - Stage 3: Evaluation mode with all stages

    Args:
        stage: Training/evaluation stage (1, 2, or 3)
        grid_shape: Shape of the 3D output grid (x, y, z)
        voxel_sizes: Size of voxels in each dimension (x, y, z)
        grid_ranges: World coordinate ranges [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        stage1_2d_cloud_checkpoint: Path to Stage 1 pretrained weights
        stage2_3d_refine_checkpoint: Path to Stage 2 pretrained weights
    """

    def __init__(self, stage, grid_shape, voxel_sizes, grid_ranges, stage1_2d_cloud_checkpoint=None,
                 stage2_3d_refine_checkpoint=None):
        super(Cloud4D, self).__init__()

        assert stage in [1, 2, 3]

        self.stage = stage
        self.grid_shape = grid_shape
        self.voxel_sizes = voxel_sizes
        self.grid_ranges = grid_ranges
        self.stage1_2d_cloud = Stage1_2D(grid_shape, voxel_sizes, grid_ranges, stage)
        self.stage2_3d_refine = Stage2_3DRefine(stage, grid_shape, voxel_sizes, self.stage1_2d_cloud.dino_dim)

        if stage1_2d_cloud_checkpoint is not None:
            print('Initialising pre-trained 2D cloud model')
            stage1_checkpoint = torch.load(stage1_2d_cloud_checkpoint, map_location='cpu')
            self.stage1_2d_cloud.load_state_dict(stage1_checkpoint['model_state_dict'])

        if stage2_3d_refine_checkpoint is not None:
            print('Initialising pre-trained 3D refine model')
            stage2_checkpoint = torch.load(stage2_3d_refine_checkpoint, map_location='cpu')
            self.stage2_3d_refine.load_state_dict(stage2_checkpoint['model_state_dict'])

    def forward(self, images, left_plucker_rays, right_plucker_rays, depths, projection_indices, valid_projection_indices, grid_depths, right_images=None,
                right_projection_indices=None, left_camera_poses=None, left_camera_intrinsics=None, right_camera_poses=None,
                right_camera_intrinsics=None, gt_cbh=None, gt_delta_heights=None, height_diff=0, outline_only=False, right_valid_projection_indices=None):
        num_batches, num_views, channels, height, width = images.shape
        num_batches, num_views, x_dim, y_dim, z_dim, hw_indices = projection_indices.shape

        all_images = torch.cat((images, right_images), dim=1)

        output = {}

        all_valid_projection_indices = torch.cat((valid_projection_indices, right_valid_projection_indices), dim=1)
        all_projection_indices = torch.cat((projection_indices, right_projection_indices), dim=1)

        if self.stage == 1:
            occupancy_logits, cloud_base_heights, delta_heights_pred, lwp_pred, all_valid_voxels = self.stage1_2d_cloud(
                all_projection_indices, all_images, all_valid_projection_indices
            )
        else:
            with torch.no_grad():
                occupancy_logits, cloud_base_heights, delta_heights_pred, lwp_pred, all_valid_voxels = self.stage1_2d_cloud(
                    all_projection_indices, all_images, all_valid_projection_indices
                )

        output['cloud_base_heights'] = cloud_base_heights
        output['delta_heights_pred'] = delta_heights_pred
        output['lwp_pred'] = lwp_pred
        output['occupancy_logits'] = occupancy_logits
        output['sampled_feature_height_slice'] = None

        coarse_clouds = make_binary_cloud(
            lwp_pred, occupancy_logits, cloud_base_heights, delta_heights_pred,
            x_dim=self.grid_shape[0], y_dim=self.grid_shape[1], z_dim=self.grid_shape[2],
            start_height=0, voxel_size=self.voxel_sizes[0]
        )

        output['coarse_clouds'] = coarse_clouds
        output['mask_from_dropped_cameras'] = all_valid_voxels

        mask_threshold = 1e-9
        mask_downsample_factor = 8
        mask = (coarse_clouds > mask_threshold).float()
        coarse_mask = torch.nn.functional.max_pool3d(mask, kernel_size=mask_downsample_factor, stride=mask_downsample_factor, padding=0)

        if (torch.sum(coarse_mask) < 1e-6):
            if self.training:
                # Select a random cube in the grid (this is partly to prevent some steps not having any gradients which caused some annoying issues)
                _, _, downsampled_x_dim, downsampled_y_dim, downsampled_z_dim = coarse_mask.shape
                max_width = math.floor(0.15 * downsampled_x_dim)  # Max width of the cube
                min_width = math.ceil(0.025 * downsampled_x_dim)  # Min width of the cube

                random_x = np.random.randint(0, downsampled_x_dim)
                random_y = np.random.randint(0, downsampled_y_dim)
                random_z = np.random.randint(0, downsampled_z_dim)

                random_width = np.random.randint(min_width, max_width)
                coarse_mask[:, :, random_x:min(random_x + random_width, downsampled_x_dim),
                            random_y:min(random_y + random_width, downsampled_y_dim),
                            random_z:min(random_z + random_width, downsampled_z_dim)] = 1
            else:
                output['output_vol'] = coarse_clouds.squeeze(1)
                output['empty_coarse_clouds'] = False
                empty_mask = torch.nn.functional.interpolate(coarse_mask.float(), scale_factor=mask_downsample_factor, mode='nearest')
                empty_mask = (empty_mask > 0.5).squeeze(1)
                output['sparsity_mask'] = empty_mask

                return output
        
        # Upsample mask again
        coarse_mask = torch.nn.functional.interpolate(coarse_mask.float(), scale_factor=mask_downsample_factor, mode='nearest')
        coarse_mask = (coarse_mask > 0.5).squeeze(1)

        output['sparsity_mask'] = coarse_mask
        output['empty_coarse_clouds'] = False

        if self.stage == 1 or output['empty_coarse_clouds']:
            output['output_vol'] = coarse_clouds.squeeze(1)

        elif self.stage == 2 or self.stage == 3:
            batch_size, num_views, channels, height, width = all_images.shape

            all_images = all_images.view(batch_size * num_views, channels, height, width)
        
            dino_inputs = self.stage1_2d_cloud.transforms(all_images)
            lr_feats = self.stage1_2d_cloud.dino_loftup.get_intermediate_layers(dino_inputs, reshape=True)[0].to(all_images.device)

            upsampled_dino_inputs = []
            for i in range(batch_size * num_views):
                with torch.no_grad():   #  Should not make a difference
                    curr_hr_feats = self.stage1_2d_cloud.upsampler(lr_feats[i:i+1, ...], dino_inputs[i:i+1, ...])
                
                # Use the linear layer to downsample the features, need to change channel dim to last. But use weight from stage 2 instead of stage 1
                curr_hr_feats = curr_hr_feats.permute(0, 2, 3, 1)
                curr_hr_feats = self.stage2_3d_refine.downsample_dino(curr_hr_feats)
                curr_hr_feats = curr_hr_feats.permute(0, 3, 1, 2)

                upsampled_dino_inputs.append(curr_hr_feats)

            upsampled_dino_inputs = torch.cat(upsampled_dino_inputs, dim=0)

            dense_output, sparse_output = self.stage2_3d_refine(coarse_clouds, upsampled_dino_inputs, torch.cat((projection_indices, right_projection_indices), dim=1), all_valid_projection_indices, coarse_mask)

            if not self.training:
                dense_output = torch.clamp(dense_output, min=0.0)

            output['output_vol'] = dense_output.squeeze(1)
            output['sparse_output'] = sparse_output
            
        return output


class Stage1_2D(torch.nn.Module):
    """Stage 1: Predict 2D cloud properties from multi-height feature slices.

    Uses DINOv2 features sampled at multiple height slices to predict:
    - Cloud base height (CBH)
    - Cloud thickness (delta height)
    - Liquid water path (LWP)

    Features are upsampled using LoftUp and processed through a Modern UNet.

    Args:
        grid_shape: Shape of the output grid (x, y, z)
        voxel_sizes: Voxel sizes in each dimension
        grid_ranges: World coordinate ranges
        stage: Current training stage (affects parameter freezing)
    """

    def __init__(self, grid_shape, voxel_sizes, grid_ranges, stage):
        super(Stage1_2D, self).__init__()

        self.grid_shape = grid_shape
        self.voxel_sizes = voxel_sizes
        self.grid_ranges = grid_ranges
        self.dino_dim = 384
        self.stage = stage

        self.transforms = transforms.Compose([
            transforms.CenterCrop((476, 630)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.target_heights = torch.arange(400, 4000, 200) + voxel_sizes[-1] / 2

        # Use grid_shape, grid_ranges and voxel_sizes to calculate the target height indices
        self.target_indices = torch.floor((self.target_heights - grid_ranges[-1][0]) / voxel_sizes[-1]).to(torch.int32)

        # Convert target heights to km
        self.target_heights = self.target_heights / 1000.0

        self.num_slices = len(self.target_heights)
        self.num_cameras = 6
        
        self.downsampled_dino_dim = 16
        self.downsample_dino = torch.nn.Linear(self.dino_dim, self.downsampled_dino_dim)

        unet_input_dim = self.downsampled_dino_dim * self.num_slices

        self.height_predictor = UNet(unet_input_dim, out_channels=3, model_channels=128, resolution=200, num_blocks=4, attn_resolutions=[20])

        self.height_embedder = TimestepEmbedder(self.downsampled_dino_dim)

        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(self.downsampled_dino_dim, 2 * self.downsampled_dino_dim, bias=True)
        )

        # Zero out adaLN weights
        torch.nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        torch.nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        self.dino_loftup = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.upsampler = torch.hub.load('andrehuang/loftup', 'loftup_dinov2s', pretrained=True)

        for param in self.upsampler.parameters():
            param.requires_grad = False

        for param in self.dino_loftup.parameters():
            param.requires_grad = False

        # Freeze DINO if not stage 1
        if stage != 1:
            for param in self.downsample_dino.parameters():
                param.requires_grad = False

            for param in self.height_predictor.parameters():
                param.requires_grad = False
            
            for param in self.height_embedder.parameters():
                param.requires_grad = False

            for param in self.adaLN_modulation.parameters():
                param.requires_grad = False

    def forward(self, projection_indices, all_images, valid_projection_indices):
        batch_size, num_views, channels, height, width = all_images.shape

        all_images = all_images.view(batch_size * num_views, channels, height, width)

        dino_inputs = self.transforms(all_images)
        lr_feats = self.dino_loftup.get_intermediate_layers(dino_inputs, reshape=True)[0].to(all_images.device)

        upsampled_dino_inputs = []
        for i in range(batch_size * num_views):
            with torch.no_grad():   #  Should not make a difference
                curr_hr_feats = self.upsampler(lr_feats[i:i+1, ...], dino_inputs[i:i+1, ...])
            
            # Use the linear layer to downsample the features, need to change channel dim to last
            curr_hr_feats = curr_hr_feats.permute(0, 2, 3, 1)
            curr_hr_feats = self.downsample_dino(curr_hr_feats)
            curr_hr_feats = curr_hr_feats.permute(0, 3, 1, 2)

            upsampled_dino_inputs.append(curr_hr_feats)

        upsampled_dino_inputs = torch.cat(upsampled_dino_inputs, dim=0)

        sampled_features = upsampled_dino_inputs
        feature_dim = self.downsampled_dino_dim

        # Flatten 3D to 2D
        _, _, x_dim, y_dim, z_dim, hw_indices = projection_indices.shape

        slice_projection_indices = projection_indices[:, :, :, :, self.target_indices.to(all_images.device), :]
        valid_slice_projection_indices = valid_projection_indices[:, :, :, :, self.target_indices.to(all_images.device)]
        slice_projection_indices = slice_projection_indices.view(batch_size * num_views, x_dim, y_dim * self.num_slices, hw_indices)
        
        
        # Computationally efficient implementation of the homography in Equation 1 using grid_sample
        sampled_features_height_slices = torch.nn.functional.grid_sample(sampled_features, slice_projection_indices.to(sampled_features.dtype), align_corners=False)
        sampled_features_height_slices = sampled_features_height_slices.view(batch_size, num_views, feature_dim, x_dim, y_dim, self.num_slices)

        slice_heights = self.target_heights.unsqueeze(0).repeat((batch_size, 1)).to(sampled_features_height_slices.device).flatten()

        height_embedding = self.height_embedder(slice_heights).view((batch_size, self.num_slices, feature_dim))
        total_embedding = height_embedding

        if self.training and self.stage == 1: 
            num_cameras_to_keep = torch.randint(1, num_views + 1, (batch_size,)).to(sampled_features_height_slices.device)

            # Not vectorised but should be negligible overhead
            mask = torch.zeros((batch_size, num_views, 1), dtype=torch.bool).to(sampled_features_height_slices.device)
            num_cameras_per_pixel = torch.zeros((batch_size, x_dim, y_dim, self.num_slices), dtype=torch.int32).to(sampled_features_height_slices.device)

            for i in range(batch_size):
                camera_indices = torch.from_numpy(np.random.choice(np.arange(0, num_views), size=num_cameras_to_keep[i].item(), replace=False)).to(torch.int32)
                mask[i, camera_indices] = True

                num_cameras_per_pixel[i, :, :, :] = torch.sum(valid_slice_projection_indices[i, camera_indices, ...], dim=0, keepdim=False)

            sampled_features_height_slices = sampled_features_height_slices * mask.view(batch_size, num_views, 1, 1, 1, 1).to(sampled_features_height_slices.dtype)
        else:
            num_cameras_per_pixel = torch.sum(valid_slice_projection_indices, dim=1, keepdim=False)
            mask = torch.ones((batch_size, num_views, 1), dtype=torch.bool).to(sampled_features_height_slices.device)

        all_valid_voxels = torch.any(num_cameras_per_pixel > 0, dim=-1, keepdim=True).repeat(1, 1, 1, z_dim)
        num_cameras_per_pixel = torch.clamp(num_cameras_per_pixel, min=1)

        sampled_features_height_slices_mean = torch.sum(sampled_features_height_slices, dim=1, keepdim=False)
        sampled_features_height_slices_mean = sampled_features_height_slices_mean / num_cameras_per_pixel.unsqueeze(1).repeat(1, feature_dim, 1, 1, 1).to(sampled_features_height_slices_mean.dtype)
        sampled_features_height_slices_mean = sampled_features_height_slices_mean.view(batch_size, feature_dim, x_dim, y_dim, self.num_slices)
        sampled_features_height_slices_mean = sampled_features_height_slices_mean.permute(0, 1, 4, 2, 3).reshape(batch_size, feature_dim * self.num_slices, x_dim, y_dim)

        total_feature_height_slices = sampled_features_height_slices_mean

        # Do adaLN modulation
        adaln_shift, adaln_scale = self.adaLN_modulation(total_embedding).chunk(2, dim=-1)

        adaln_shift = adaln_shift.reshape(batch_size, self.num_slices * feature_dim)
        adaln_scale = adaln_scale.reshape(batch_size, self.num_slices * feature_dim)

        # LN expects dino_dim to be last dimension but it is currently (batch, dino_dim * num_slices, x_dim, y_dim)
        total_feature_height_slices = total_feature_height_slices.permute(0, 2, 3, 1)
        total_feature_height_slices = modulate(total_feature_height_slices, adaln_shift, adaln_scale)
        total_feature_height_slices = total_feature_height_slices.permute(0, 3, 1, 2)

        height_preds = self.height_predictor(total_feature_height_slices)
        cloud_base_heights, delta_heights_pred, lwp_pred = height_preds[:, 0:1, :, :], height_preds[:, 1:2, :, :], height_preds[:, 2:3, :, :]
        occupany_logits = torch.ones_like(cloud_base_heights) * 99999   # Just do this as a placeholder, basically a mask of ones

        delta_heights_pred = torch.nn.functional.leaky_relu(delta_heights_pred)
        lwp_pred = torch.nn.functional.leaky_relu(lwp_pred)
        cloud_base_heights = torch.nn.functional.leaky_relu(cloud_base_heights)

        return occupany_logits, cloud_base_heights, delta_heights_pred, lwp_pred, all_valid_voxels
    
class Stage2_3DRefine(torch.nn.Module):
    """Stage 2: Refine 3D cloud volume using sparse 3D CNNs.

    Takes the coarse cloud volume from Stage 1 and refines it using:
    - 3D-projected DINOv2 features
    - Sparse 3D CNN with transformer blocks

    Only processes voxels within the coarse cloud mask for efficiency.

    Args:
        stage: Current training stage (affects parameter freezing)
        grid_shape: Shape of the 3D grid (x, y, z)
        voxel_sizes: Voxel sizes in each dimension
        dino_dim: Dimension of input DINOv2 features
    """

    def __init__(self, stage, grid_shape, voxel_sizes, dino_dim):
        super(Stage2_3DRefine, self).__init__()

        self.grid_shape = grid_shape
        self.voxel_sizes = voxel_sizes
        self.dino_dim = dino_dim
        self.downsample_dino_dim = 16

        self.downsample_dino = torch.nn.Linear(dino_dim, self.downsample_dino_dim)

        cnn_input_dim = 1 + self.downsample_dino_dim

        self.sparse_3d_cnn = Sparse3DCNN(in_channels=cnn_input_dim, model_channels=384, out_channels=1, num_io_res_blocks=2,
                                         io_block_channels=[128, 256], use_skip_connection=True, use_fp16=False,
                                         num_blocks=12, num_heads=12, mlp_ratio=4, qk_rms_norm=True, pe_mode="ape")

        if stage != 2:
            for param in self.sparse_3d_cnn.parameters():
                param.requires_grad = False

    def forward(self, input_volume, dino_features, projection_indices, valid_projection_indices, sparse_mask):
        # DINO features are batch_size * num_views, dino_dim, height, width
        _, dino_dim, h, w = dino_features.shape

        batch_size, num_views, x_dim, y_dim, z_dim, hw_indices = projection_indices.shape

        dino_features = dino_features.view(batch_size, num_views, dino_dim, h, w)

        dino_features_3d = []
        for i in range(batch_size):
            curr_projection_indices = projection_indices[i:i + 1, ...]
            curr_valid_projection_indices = valid_projection_indices[i:i + 1, ...]
            curr_dino_features = dino_features[i, ...]

            curr_projection_indices = curr_projection_indices.permute(1, 0, 2, 3, 4, 5)   # num_views, batch_size, x_dim, y_dim, z_dim, hw_indices
            curr_valid_projection_indices = curr_valid_projection_indices.permute(1, 0, 2, 3, 4)  # num_views, batch_size, x_dim, y_dim, z_dim
            active_projection_indices = curr_projection_indices[:, sparse_mask[i:i+1, ...], :].unsqueeze(2)  # num_views, num_active_voxels, 1, 2
            active_valid_projection_indices = curr_valid_projection_indices[:, sparse_mask[i:i+1, ...]]  # num_views, num_active_voxels

            curr_dino_features_3d = torch.nn.functional.grid_sample(curr_dino_features, active_projection_indices.to(dino_features.dtype), align_corners=False).squeeze(-1)   # num_views, dino_dim, num_active_voxels
            curr_dino_features_3d = torch.sum(curr_dino_features_3d, dim=0)
            num_valid_views = torch.sum(active_valid_projection_indices, dim=0, keepdim=False)
            num_valid_views = torch.clamp(num_valid_views, min=1)
            curr_dino_features_3d = curr_dino_features_3d / num_valid_views.unsqueeze(0).repeat(dino_dim, 1)
        
            dino_features_3d.append(curr_dino_features_3d.permute(1, 0))

        dino_features_3d = torch.cat(dino_features_3d, dim=0)  # num_active_voxels, dino_dim

        # Input volume is (batch_size, channel_size, x_dim, y_dim, z_dim)
        # Want a list of coordinates that are (batch_size * x_dim * y_dim * z_dim, 4). Where the four channels are (batch_idx, x, y, z)
        channel_size = input_volume.shape[1]
        batch_idx = torch.arange(batch_size).view(batch_size, 1, 1, 1).repeat(1, x_dim, y_dim, z_dim)
        x = torch.arange(x_dim).view(1, x_dim, 1, 1).repeat(batch_size, 1, y_dim, z_dim)
        y = torch.arange(y_dim).view(1, 1, y_dim, 1).repeat(batch_size, x_dim, 1, z_dim)
        z = torch.arange(z_dim).view(1, 1, 1, z_dim).repeat(batch_size, x_dim, y_dim, 1)

        input_coords = torch.stack((batch_idx, x, y, z), dim=-1).to(input_volume.device).to(torch.int32)
        input_volume = input_volume.permute(0, 2, 3, 4, 1).to(input_volume.device)

        # Sparsify the input volume
        input_volume = input_volume[sparse_mask, :]
        input_coords = input_coords[sparse_mask, :]

        # Scale DINO features
        sparse_cnn_input = SparseTensor(feats=torch.cat((input_volume, dino_features_3d / self.downsample_dino_dim), dim=1), coords=input_coords, shape=[x_dim, y_dim, z_dim])
        sparse_cnn_input._scale = (4, 4, 4)   # Need to be higher than 1 for correct down/upsampling with more than 2 io blocks
        
        output = self.sparse_3d_cnn(sparse_cnn_input)

        # Skip connection
        output = output.replace(output.feats + input_volume)
        output = output.replace(torch.nn.functional.leaky_relu(output.feats))

        dense_output = torch.zeros((batch_size, x_dim, y_dim, z_dim, 1), device=input_volume.device)
        dense_output[sparse_mask, :] = output.feats
        dense_output = dense_output.permute(0, 4, 1, 2, 3)

        return dense_output, output.feats

