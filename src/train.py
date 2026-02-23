"""
Cloud Volume Prediction Training Script.

Trains a multi-stage model for predicting 3D cloud volumes from stereo images.
The training uses distributed data parallelism via Accelerate and logs to W&B.
"""

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, set_seed
from datetime import timedelta

import os
import argparse

import numpy as np
import wandb
from data import CloudDataset
import torch
import torch.utils.data as data
from matplotlib import pyplot as plt

from models import Cloud4D
from utils import (
    debug_projection,
    get_orthographic_sum_figure,
    get_projection_figure,
    get_height_pred_figure,
    _move_tensors_to_device,
)


def step(batch, stage, left_plucker_rays, right_plucker_rays, projection_indices, valid_projection_indices, grid_depths, model,
         right_projection_indices=None,
         left_camera_poses=None, left_camera_intrinsics=None,
         right_camera_poses=None, right_camera_intrinsics=None, right_valid_projection_indices=None):
    """Execute a single training/inference step through the model."""
    left_images = batch['left_images']
    right_images = batch['right_images']
    depths = batch['depths']
    camera_indices = batch['camera_indices']
    device = camera_indices.device

    # Move all projection-related tensors to the current device
    tensors_to_move = {
        'projection_indices': projection_indices,
        'valid_projection_indices': valid_projection_indices,
        'grid_depths': grid_depths,
        'right_projection_indices': right_projection_indices,
        'left_plucker_rays': left_plucker_rays,
        'right_plucker_rays': right_plucker_rays,
        'left_camera_poses': left_camera_poses,
        'left_camera_intrinsics': left_camera_intrinsics,
        'right_camera_poses': right_camera_poses,
        'right_camera_intrinsics': right_camera_intrinsics,
        'right_valid_projection_indices': right_valid_projection_indices,
    }
    tensors = _move_tensors_to_device(tensors_to_move, device)

    # Select batch-specific data using camera indices
    batch_projection_indices = tensors['projection_indices'][camera_indices]
    batch_valid_projection_indices = tensors['valid_projection_indices'][camera_indices]
    batch_right_valid_projection_indices = tensors['right_valid_projection_indices'][camera_indices]
    batch_grid_depths = tensors['grid_depths'][camera_indices]
    batch_right_projection_indices = tensors['right_projection_indices'][camera_indices]
    batch_left_plucker_rays = tensors['left_plucker_rays'][camera_indices].squeeze(2)
    batch_right_plucker_rays = tensors['right_plucker_rays'][camera_indices].squeeze(2)
    batch_left_camera_poses = tensors['left_camera_poses'][camera_indices]
    batch_left_camera_intrinsics = tensors['left_camera_intrinsics'][camera_indices]
    batch_right_camera_poses = tensors['right_camera_poses'][camera_indices]
    batch_right_camera_intrinsics = tensors['right_camera_intrinsics'][camera_indices]

    # Clone tensors to ensure they are not modified in-place by the model
    output_dict = model(
        left_images,
        torch.clone(batch_left_plucker_rays),
        torch.clone(batch_right_plucker_rays),
        depths,
        torch.clone(batch_projection_indices),
        torch.clone(batch_valid_projection_indices),
        torch.clone(batch_grid_depths),
        right_images=right_images,
        right_projection_indices=torch.clone(batch_right_projection_indices),
        left_camera_poses=torch.clone(batch_left_camera_poses),
        left_camera_intrinsics=torch.clone(batch_left_camera_intrinsics),
        right_camera_poses=torch.clone(batch_right_camera_poses),
        right_camera_intrinsics=torch.clone(batch_right_camera_intrinsics),
        gt_cbh=None,
        gt_delta_heights=None,
        right_valid_projection_indices=torch.clone(batch_right_valid_projection_indices)
    )

    return output_dict

def _compute_height_grid(z_dim, voxel_size, start_height, x_dim, y_dim, device=None):
    """Create a height grid for cloud property extraction."""
    grid_heights = (torch.arange(z_dim) * voxel_size + start_height + voxel_size / 2).float()
    if device is not None:
        grid_heights = grid_heights.to(device)
    return grid_heights.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, x_dim, y_dim, 1)


def _extract_cloud_base_and_thickness(gt_volumes, grid_heights):
    """Extract cloud base height and thickness from volumes using height grid."""
    binary_gt = (gt_volumes > 1e-6)
    valid_heights = grid_heights * binary_gt

    # Find minimum height (cloud base)
    for_min_height = valid_heights.clone()
    for_min_height[~binary_gt] = 20000

    gt_cbh = torch.min(for_min_height, dim=-1)[0]
    gt_cbh[gt_cbh > 8000] = 0

    # Cloud thickness is max height minus base height
    gt_delta_heights = torch.max(valid_heights, dim=-1)[0] - gt_cbh

    return gt_cbh, gt_delta_heights


def gt_cbh_from_volumes(gt_volumes, voxel_size, start_height):
    """Compute cloud base height and thickness from ground truth volumes."""
    _, x_dim, y_dim, z_dim = gt_volumes.shape
    grid_heights = _compute_height_grid(z_dim, voxel_size, start_height, x_dim, y_dim)
    return _extract_cloud_base_and_thickness(gt_volumes, grid_heights)


def extract_physical_properties(gt_volumes, voxel_size, start_height):
    """Extract LWP, cloud base height, and cloud thickness from ground truth volumes."""
    _, x_dim, y_dim, z_dim = gt_volumes.shape
    grid_heights = _compute_height_grid(z_dim, voxel_size, start_height, x_dim, y_dim, device=gt_volumes.device)

    gt_cbh, gt_delta_heights = _extract_cloud_base_and_thickness(gt_volumes, grid_heights)

    # Liquid water path (multiply with voxel size for correct units kg/m^2)
    lwp = torch.sum(gt_volumes, dim=-1) * voxel_size

    return lwp, gt_cbh, gt_delta_heights

def get_all_losses(args, output_dict, gt_volumes, voxel_size, start_height, not_in_all_2d):
    """Compute all training losses based on the current training stage."""
    lwp_loss, cbh_loss, delta_height_loss = get_stage1_loss(output_dict, gt_volumes, voxel_size, start_height, not_in_all_2d)
    volume_loss = get_stage2_loss(output_dict, gt_volumes)

    if args.stage == 1:
        loss = args.lwp_lambda * lwp_loss + args.cbh_lambda * cbh_loss + args.delta_height_lambda * delta_height_loss
    elif args.stage == 2:
        loss = volume_loss
    else:
        loss = torch.tensor([0])

    return loss, cbh_loss, delta_height_loss, volume_loss, lwp_loss

def get_stage1_loss(output_dict, gt_volumes, voxel_size, start_height, not_in_all_2d):
    """Compute stage 1 losses (2D cloud property prediction)."""
    lwp, gt_cbh, gt_delta_heights = extract_physical_properties(gt_volumes, voxel_size, start_height)

    gt_occupancy = (gt_cbh > 0)
    gt_occupancy[not_in_all_2d] = 0

    pred_datatype = output_dict['cloud_base_heights'].dtype

    with torch.no_grad():
        # Zero out ground truth where there is no cloud
        gt_cbh[~gt_occupancy] = 0
        gt_delta_heights[~gt_occupancy] = 0
        lwp[~gt_occupancy] = 0

    lwp_loss = torch.nn.functional.l1_loss(output_dict['lwp_pred'].squeeze(1), lwp.to(pred_datatype), reduction='mean')
    cbh_loss = torch.nn.functional.l1_loss(output_dict['cloud_base_heights'].squeeze(1), gt_cbh.to(pred_datatype) / 1000, reduction='mean')
    delta_height_loss = torch.nn.functional.l1_loss(output_dict['delta_heights_pred'].squeeze(1), gt_delta_heights.to(pred_datatype) / 1000, reduction='mean')

    return lwp_loss, cbh_loss, delta_height_loss

def get_stage2_loss(output_dict, gt_volumes):
    """Compute stage 2 loss (3D volume prediction)."""
    gt_volumes = gt_volumes.to(output_dict['output_vol'].dtype)
    volume_loss = torch.nn.functional.l1_loss(output_dict['output_vol'], gt_volumes, reduction='mean')

    return volume_loss


def compute_visibility_mask(left_valid_projection_indices, right_valid_projection_indices, batch_volumes, grid_shape):
    """Compute visibility mask for all cameras. This is only used when cameras are randomly dropped during Stage 1 training.

    Returns:
        not_in_all: 3D mask indicating voxels not visible from all cameras
        not_in_all_2d: 2D mask indicating columns not visible from all cameras
    """
    not_in_all = ~torch.all(torch.cat((left_valid_projection_indices, right_valid_projection_indices), dim=0), dim=0)
    not_in_all = not_in_all.unsqueeze(0).repeat(batch_volumes.shape[0], 1, 1, 1)

    not_in_all_2d = torch.all(not_in_all, dim=-1)
    not_in_all_2d = torch.zeros_like(not_in_all_2d)  # Remove filtering for now
    not_in_all = not_in_all_2d.unsqueeze(-1).repeat(1, 1, 1, grid_shape[-1])

    return not_in_all, not_in_all_2d


def apply_camera_visibility_mask(output_dict, batch):
    """Apply camera visibility mask to predictions and ground truth (to handle random dropping of cameras"""
    mask = output_dict['mask_from_dropped_cameras']
    any_mask = torch.any(mask, dim=-1).unsqueeze(1)

    output_dict['cloud_base_heights'] = output_dict['cloud_base_heights'] * any_mask
    output_dict['delta_heights_pred'] = output_dict['delta_heights_pred'] * any_mask
    output_dict['lwp_pred'] = output_dict['lwp_pred'] * any_mask
    output_dict['output_vol'] = output_dict['output_vol'] * mask
    batch['volumes'] = batch['volumes'] * mask


def train(args):
    """Main training function."""
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    timeout = InitProcessGroupKwargs(timeout=timedelta(minutes=3))

    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[kwargs, timeout])
    set_seed(1)

    accelerator.init_trackers("predict_cloud_volume")
    os.makedirs(args.checkpoint_path, exist_ok=True)

    if args.pretrain:
        # Synthetic Terragen data configuration
        data_dir = args.data_dir or 'new_paired_cameras'
        volume_dir = args.volume_dir or './volumes/5km_crop'
        grid_shape = (200, 200, 160)
        voxel_sizes = (25, 25, 25)
        exr_images = False
        downsample_factor = 1
        start_height = 0
        scale_volumes = 1 / 1235.6852  # Scale with mean
        grid_ranges = [(-2500, 2500), (-2500, 2500), (0, 4000)]
    else:
        # MicroHH LES data configuration
        data_dir = args.data_dir or '../datasets/microhh_all_cases_dataset/images/'
        volume_dir = args.volume_dir or '../datasets/microhh_all_cases_dataset/5km_crop/'
        grid_shape = (200, 200, 160)
        voxel_sizes = (25, 25, 25)
        downsample_factor = 1
        exr_images = True
        start_height = 0
        scale_volumes = None
        grid_ranges = [(-2500, 2500), (-2500, 2500), (0, 4000)]

    if downsample_factor > 1:
        grid_shape = (grid_shape[0] // downsample_factor, grid_shape[1] // downsample_factor, grid_shape[2] // downsample_factor)
        voxel_sizes = (voxel_sizes[0] * downsample_factor, voxel_sizes[1] * downsample_factor, voxel_sizes[2] * downsample_factor)

    train_dataset = CloudDataset(grid_shape, voxel_sizes, volume_path=volume_dir, camera_folder_path=data_dir, split='train', get_projection_indices=True,
                                 augmentations=True, exr_images=exr_images, downsample_factor=downsample_factor,
                                 scale_volumes=scale_volumes, grid_ranges=grid_ranges)
    persistent_workers = args.num_workers > 0
    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent_workers)

    val_dataset = CloudDataset(grid_shape, voxel_sizes, volume_path=volume_dir, camera_folder_path=data_dir, split='val', get_projection_indices=False,
                               augmentations=False, exr_images=exr_images, downsample_factor=downsample_factor,
                               scale_volumes=scale_volumes, grid_ranges=grid_ranges)
    val_dataloader = data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent_workers)

    left_projection_indices = train_dataset.left_pixel_coords
    left_valid_projection_indices = train_dataset.left_valid_coords
    left_grid_depths = train_dataset.left_grid_depths
    left_plucker_rays = train_dataset.left_plucker_rays

    right_projection_indices = train_dataset.right_pixel_coords
    right_valid_projection_indices = train_dataset.right_valid_coords
    right_grid_depths = train_dataset.right_grid_depths

    right_plucker_rays = train_dataset.right_plucker_rays

    left_camera_poses = train_dataset.left_camera_poses
    left_camera_intrinsics = train_dataset.left_camera_intrinsics
    right_camera_poses = train_dataset.right_camera_poses
    right_camera_intrinsics = train_dataset.right_camera_intrinsics

    assert args.stage in [1, 2]

    model = Cloud4D(args.stage, grid_shape, voxel_sizes, grid_ranges,
                               stage1_2d_cloud_checkpoint=args.stage1_checkpoint,
                               stage2_3d_refine_checkpoint=args.stage2_checkpoint)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    keep_training = True
    curr_step = 0

    if args.pretrain:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=0)
        train_dataloader, val_dataloader, model, optimizer, scheduler = accelerator.prepare(
            train_dataloader, val_dataloader, model, optimizer, scheduler
        )
    else:
        train_dataloader, val_dataloader, model, optimizer = accelerator.prepare(
            train_dataloader, val_dataloader, model, optimizer
        )

    device = accelerator.device

    while keep_training:
        for batch in train_dataloader:
            left_valid_projection_indices = left_valid_projection_indices.to(device)
            right_valid_projection_indices = right_valid_projection_indices.to(device)

            # Compute visibility masks for handling random camera dropping during Stage 1 training
            not_in_all, not_in_all_2d = compute_visibility_mask(
                left_valid_projection_indices, right_valid_projection_indices, batch['volumes'], grid_shape
            )

            optimizer.zero_grad()

            output_dict = step(
                batch, args.stage, left_plucker_rays, right_plucker_rays,
                left_projection_indices, left_valid_projection_indices, left_grid_depths, model,
                right_projection_indices=right_projection_indices,
                left_camera_poses=left_camera_poses, left_camera_intrinsics=left_camera_intrinsics,
                right_camera_poses=right_camera_poses, right_camera_intrinsics=right_camera_intrinsics,
                right_valid_projection_indices=right_valid_projection_indices
            )

            apply_camera_visibility_mask(output_dict, batch)

            loss, cbh_loss, delta_height_loss, volume_loss, lwp_loss = get_all_losses(args, output_dict, batch['volumes'], voxel_sizes[0], start_height, not_in_all_2d)

            accelerator.backward(loss)
            optimizer.step()

            if args.pretrain:
                scheduler.step()

            curr_step += 1

            if curr_step % args.validation_frequency == 0:
                with torch.no_grad():
                    val_batch = next(iter(val_dataloader))

                    val_not_in_all, val_not_in_all_2d = compute_visibility_mask(
                        left_valid_projection_indices, right_valid_projection_indices, val_batch['volumes'], grid_shape
                    )

                    val_output_dict = step(
                        val_batch, args.stage, left_plucker_rays, right_plucker_rays,
                        left_projection_indices, left_valid_projection_indices, left_grid_depths, model,
                        right_projection_indices=right_projection_indices,
                        left_camera_poses=left_camera_poses, left_camera_intrinsics=left_camera_intrinsics,
                        right_camera_poses=right_camera_poses, right_camera_intrinsics=right_camera_intrinsics,
                        right_valid_projection_indices=right_valid_projection_indices
                    )

                    apply_camera_visibility_mask(val_output_dict, val_batch)

                    val_loss, val_cbh_loss, val_delta_height_loss, val_volume_loss, val_lwp_loss = get_all_losses(args, val_output_dict, val_batch['volumes'], voxel_sizes[0], start_height, val_not_in_all_2d)

                    # Visualization
                    new_output_vol = output_dict['output_vol']
                    new_val_output_vol = val_output_dict['output_vol']

                    new_output_vol[not_in_all] = 0
                    new_val_output_vol[val_not_in_all] = 0

                    val_pred_volumes = new_val_output_vol
                    pred_volumes = new_output_vol

                    # Change to batch size one just for visualization
                    for key in val_batch:
                        val_batch[key] = val_batch[key][:1]

                    for key in batch:
                        batch[key] = batch[key][:1]

                    val_pred_volumes = val_pred_volumes[:1]
                    pred_volumes = pred_volumes[:1]

                    for key in ['cloud_base_heights', 'delta_heights_pred',
                                'coarse_clouds', 'lwp_pred', 'occupancy_logits', 'sampled_feature_height_slice']:
                        if key in val_output_dict and val_output_dict[key] is not None:
                            val_output_dict[key] = val_output_dict[key][:1]

                        if key in output_dict and output_dict[key] is not None:
                            output_dict[key] = output_dict[key][:1]

                    # Project volume onto images
                    val_projection_fig = get_projection_figure(val_batch, val_pred_volumes, torch.clone(left_projection_indices), torch.clone(right_projection_indices),
                                                            val_batch['camera_indices'].squeeze(0))
                    train_projection_fig = get_projection_figure(batch, pred_volumes, torch.clone(left_projection_indices), torch.clone(right_projection_indices),
                                                                batch['camera_indices'].squeeze(0))

                    val_orthographic_sum_fig = get_orthographic_sum_figure(val_pred_volumes, val_batch['volumes'], coarse_cloud=val_output_dict['coarse_clouds'])
                    train_orthographic_sum_fig = get_orthographic_sum_figure(pred_volumes, batch['volumes'], coarse_cloud=output_dict['coarse_clouds'])

                    val_height_pred_fig = get_height_pred_figure(val_output_dict['cloud_base_heights'], val_output_dict['delta_heights_pred'], val_output_dict['lwp_pred'], val_output_dict['occupancy_logits'], val_batch['volumes'], val_output_dict['sampled_feature_height_slice'], voxel_size=voxel_sizes[-1])
                    train_height_pred_fig = get_height_pred_figure(output_dict['cloud_base_heights'], output_dict['delta_heights_pred'], output_dict['lwp_pred'], output_dict['occupancy_logits'], batch['volumes'], output_dict['sampled_feature_height_slice'], voxel_size=voxel_sizes[-1])

                    accelerator.log({
                        'Training Total Loss': loss.item(),
                        'Training CBH Loss': cbh_loss.item(),
                        'Training Delta Height Loss': delta_height_loss.item(),
                        'Training LWP Loss': lwp_loss.item(),
                        'Training Volume Loss': volume_loss.item(),
                        'Validation Total Loss': val_loss.item(),
                        'Validation CBH Loss': val_cbh_loss.item(),
                        'Validation Delta Height Loss': val_delta_height_loss.item(),
                        'Validation LWP Loss': val_lwp_loss.item(),
                        'Validation Volume Loss': val_volume_loss.item(),
                        'Validation Projection': val_projection_fig,
                        'Train Projection': train_projection_fig,
                        'Validation Orthographic Figure': val_orthographic_sum_fig,
                        'Train Orthographic Figure': train_orthographic_sum_fig,
                        'Validation Height Prediction Figure': val_height_pred_fig,
                        'Train Height Prediction Figure': train_height_pred_fig,
                    }, step=curr_step)

                    plt.close('all')
                    model.train()

            else:
                accelerator.log({
                    'Training Total Loss': loss.item(),
                    'Training CBH Loss': cbh_loss.item(),
                    'Training Delta Height Loss': delta_height_loss.item(),
                    'Training LWP Loss': lwp_loss.item(),
                    'Training Volume Loss': volume_loss.item(),
                }, step=curr_step)

            if curr_step % args.save_every == 0 and accelerator.is_main_process:
                # Get unwrapped model for DDP compatibility
                unwrapped_model = accelerator.unwrap_model(model)
                if args.stage == 1:
                    accelerator.save({'model_state_dict': unwrapped_model.stage1_2d_cloud.state_dict()},
                                f'{args.checkpoint_path}/stage1_model_{curr_step}.pth')
                elif args.stage == 2:
                    accelerator.save({'model_state_dict': unwrapped_model.stage2_3d_refine.state_dict()},
                                f'{args.checkpoint_path}/stage2_model_{curr_step}.pth')

            if curr_step >= args.steps:
                accelerator.end_training()
                keep_training = False
                break
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training (default: 1)')
    parser.add_argument('--steps', type=int, default=50000, help='number of steps to train (default: 10000)')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate (default: 1e-5)')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers to use for data loading (default: 4)')
    parser.add_argument('--save-every', type=int, default=1000, help='how many steps to wait before saving the model (default: 1000)')
    parser.add_argument('--validation_frequency', type=int, default=25, help='how many steps to wait before evaluating on validation set (default: 25)')

    # Loss weights for stage 1 training
    parser.add_argument('--cbh_lambda', type=float, default=0.1, help='weight for cloud base height loss (default: 0.1)')
    parser.add_argument('--delta_height_lambda', type=float, default=0.1, help='weight for delta height loss (default: 0.1)')
    parser.add_argument('--lwp_lambda', type=float, default=1.0, help='weight for liquid water path loss (default: 1.0)')

    parser.add_argument('--stage1_checkpoint', type=str, default=None, help='path to stage 1 checkpoint for resuming or initializing stage 2 (default: None)')
    parser.add_argument('--stage2_checkpoint', type=str, default=None, help='path to stage 2 checkpoint for resuming (default: None)')

    parser.add_argument('--checkpoint_path', type=str, help='path to save checkpoints', default='checkpoints')

    parser.add_argument('--stage', type=int, help='1: train 2D cloud properties, 2: train 3D refinement', required=True)

    # Add pretraining flag
    parser.add_argument('--pretrain', action='store_true', help='pretrain the model on terragen data')

    # Add data path arguments
    parser.add_argument('--data_dir', type=str, default=None, help='path to camera/image data directory')
    parser.add_argument('--volume_dir', type=str, default=None, help='path to volume data directory')
    args = parser.parse_args()

    train(args)
