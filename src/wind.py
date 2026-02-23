import os

import torch
import numpy as np
import cv2

from functools import reduce
import netCDF4

from glob import glob
from matplotlib import pyplot as plt

def wind_example(volume_dir, device='cuda', voxel_size=25, sample_every=3, num_height_slices_to_sum=5):
    """Load a sequence of predicted cloud volumes and estimate wind using cotracker_wind.

    Args:
        volume_dir: Directory containing predicted volume files matching '*pred*.npz'.
            Each file must contain an array under the key 'a' with shape (x, y, z).
            Filenames must encode the time as the third underscore-separated field in
            HHMMSS format (e.g. 'scene_001_143000_pred.npz').
        device: PyTorch device string. Default: 'cuda'.
        voxel_size: Physical size of each voxel in metres. Default: 25.
        sample_every: Stride for subsampling the file list along time. Default: 3.
        num_height_slices_to_sum: Number of vertical voxels to sum into each height slice.
            Default: 5.

    Returns:
        u_winds: numpy array of shape (num_time_windows, num_slices) — eastward wind in m/s.
        v_winds: numpy array of shape (num_time_windows, num_slices) — northward wind in m/s.
        output_times: list of mean times (seconds since midnight) per output time window.
    """

    # Define a colormap to convert summed brightness values to RGB for cotracker input
    colormap = plt.cm.viridis

    volume_files = sorted(glob(os.path.join(volume_dir, '*pred*.npz')))[::sample_every]

    num_slices = 160 // num_height_slices_to_sum
    sample_grid = np.load(volume_files[0])['a']
    x_dim, y_dim = sample_grid.shape[0], sample_grid.shape[1]

    frames = torch.zeros((len(volume_files), num_slices, x_dim, y_dim, 3), device=device)

    for idx, path in enumerate(volume_files):
        grid = np.load(path)['a']

        # Sum voxels within each height slice to improve visibility of features
        grid = np.stack(np.split(grid, grid.shape[-1] // num_height_slices_to_sum, axis=-1), axis=-1)
        grid = torch.from_numpy(grid).to(device)
        grid = torch.sum(grid, dim=-2)
        grid = grid / torch.max(grid)

        # Apply colormap: (x, y, num_slices) -> (num_slices, x, y) -> (num_slices, x, y, 3)
        grid = grid.permute(2, 0, 1)
        grid = torch.from_numpy(
            np.stack([colormap(grid[i].cpu().numpy())[:, :, :3] for i in range(grid.shape[0])], axis=0)
        ).to(device)
        frames[idx] = grid

    def get_time(path):
        time = os.path.basename(path).split('_')[2]
        return time[-6:] if len(time) > 6 else time

    times = [get_time(p) for p in volume_files]

    return cotracker_wind(frames, times, voxel_size=voxel_size)



def cotracker_wind(all_frames, times, voxel_size=25, active_num_pixels_threshold=1250, window_size=10, key_frame_period=1, group_frames_together=20):
    """Estimate horizontal wind profiles from a sequence of 3D cloud field frames using CoTracker.

    For each height slice, bright cloud features are tracked across a sliding window of frames
    using CoTracker. The tracked displacements are converted to wind velocities using the known
    voxel size and time spacing. Results are grouped into time windows and the top 10% of
    wind magnitudes are used to produce a robust estimate.

    Results are also written to a NetCDF file at
    'cloudnet_data/wind_predictions_11May_stage2_cotracker_5min_fullLWP_20230815_try5.nc'.

    Args:
        all_frames: Float tensor of shape (num_frames, num_slices, x_dim, y_dim, channels).
            A time sequence of 3D cloud fields split into height slices, with an RGB colormap
            applied. Produced by summing voxel columns within each height slice and mapping the
            result through a colormap (see wind_example.py).
        times: List of time strings in HHMMSS format (e.g. '143000' for 14:30:00), one per frame.
        voxel_size: Physical size of each voxel in metres. Default: 25.
        active_num_pixels_threshold: Minimum number of bright pixels required at a key frame
            for tracking to run. Frames with fewer active pixels are skipped. Default: 1250.
        window_size: Number of frames on each side of a key frame used as the tracking window. Default: 10.
        key_frame_period: Step size between key frames. Default: 1.
        group_frames_together: Number of consecutive key frames whose tracked wind estimates
            are pooled to produce one output time window. Default: 20.

    Returns:
        output_u_winds: numpy array of shape (num_time_windows, num_slices) — eastward wind
            component in m/s for each output time window and height slice.
        output_v_winds: numpy array of shape (num_time_windows, num_slices) — northward wind
            component in m/s for each output time window and height slice.
        output_times: list of length num_time_windows — mean time (in seconds since midnight)
            for each output time window.
    """
    num_frames, num_slices, x_dim, y_dim, channels = all_frames.shape

    heights = np.arange(0, 4000, voxel_size)
    split_heights = np.stack(np.split(heights, num_slices), axis=0)
    mean_heights = np.mean(split_heights, axis=1)

    # Cotracker expects in: B T C H W
    all_frames = all_frames.permute(1, 0, 4, 2, 3).to(torch.float32)  # Change to (1, num_frames, height_dim, x_dim, y_dim)
    all_frames = all_frames * 255

    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(all_frames.device)

    # Convert times to tensor of seconds
    times = torch.tensor([int(time[:2]) * 3600 + int(time[2:4]) * 60 + int(time[4:6]) for time in times]).to(all_frames.device)

    key_frames = range(0, num_frames - 2 * window_size, key_frame_period)
    grouped_time_starts = range(0, len(key_frames), group_frames_together)

    output_u_winds = np.zeros((len(grouped_time_starts), num_slices))
    output_v_winds = np.zeros((len(grouped_time_starts), num_slices))

    for slice_idx in range(num_slices):
        frames = all_frames[slice_idx:slice_idx + 1, :, :, :, :].to(torch.float32)

        all_u_winds = []
        all_v_winds = []
        all_times = []

        for idx in key_frames:
            # Get initial points for cotracker (format is [time, x_coord, y_coord])
            # Find threshold to get top 20% brightness
            brightness_threshold = torch.quantile(frames[:, idx:idx + 1, 0, :, :], 0.5)
            active_pixels = (frames[:, idx:idx + 1, 0, :, :] > brightness_threshold)

            _, num_keyframes, _, _ = active_pixels.shape
            x_indices, y_indices = torch.meshgrid((torch.arange(x_dim), torch.arange(y_dim)))
            x_indices = x_indices.to(frames.device)
            y_indices = y_indices.to(frames.device)
            selected_x_pixels = x_indices.unsqueeze(0).unsqueeze(0).repeat(1, num_keyframes, 1, 1)[active_pixels]
            selected_y_pixels = y_indices.unsqueeze(0).unsqueeze(0).repeat(1, num_keyframes, 1, 1)[active_pixels]
            selected_timesteps = key_frame_period * torch.arange(num_keyframes).to(frames.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x_dim, y_dim)[active_pixels]
            queries = torch.stack((selected_timesteps, selected_x_pixels, selected_y_pixels), dim=-1).to(frames.device)
            queries = queries.unsqueeze(0)

            # Sample 25 random queries
            queries = queries[:, torch.randperm(queries.shape[1])[:25], :].to(torch.float32)

            # Swap x and y dimensions
            queries = queries[..., [0, 2, 1]]

            if torch.sum(active_pixels) > active_num_pixels_threshold:
                curr_frame_window = frames[:, idx:idx + 2 * window_size, :, :, :]
                pred_tracks, pred_visibility = cotracker(curr_frame_window, grid_size=10, queries=queries)

                _, _, num_queries, _ = pred_tracks.shape

                curr_times = times[idx:idx + 2 * window_size].unsqueeze(0)

                curr_u_wind = []
                curr_v_wind = []

                for i in range(num_queries):
                    if torch.sum(pred_visibility[..., i]) > 5:
                        visible_coords = pred_tracks[:, :, i, :][pred_visibility[..., i], :]
                        timesteps = curr_times[pred_visibility[..., i]]

                        # Use first and last points
                        delta_x = visible_coords[-1, 0] - visible_coords[0, 0]
                        delta_y = visible_coords[-1, 1] - visible_coords[0, 1]
                        delta_t = timesteps[-1] - timesteps[0]

                        u_wind = delta_x * voxel_size / delta_t
                        v_wind = delta_y * voxel_size / delta_t

                        curr_u_wind.extend([torch.median(u_wind).item()])
                        curr_v_wind.extend([torch.median(v_wind).item()])

                all_u_winds.append(curr_u_wind)
                all_v_winds.append(curr_v_wind)
                all_times.append(times[idx:idx + 2 * window_size].cpu().numpy())
            else:
                all_u_winds.append([])
                all_v_winds.append([])
                all_times.append(times[idx:idx + 2 * window_size].cpu().numpy())

        output_times = []

        # Group all_u_winds into X minute intervals
        for i in grouped_time_starts:
            curr_u_wind = list(reduce(lambda x, y: x + y, all_u_winds[i:i + group_frames_together], []))
            curr_v_wind = list(reduce(lambda x, y: x + y, all_v_winds[i:i + group_frames_together], []))
            output_times.append(np.mean(all_times[i:i + group_frames_together]))

            if len(curr_u_wind) == 0:
                continue

            magnitudes = np.sqrt(np.array(curr_u_wind) ** 2 + np.array(curr_v_wind) ** 2)
            top_quartile_indices = np.where(magnitudes > np.quantile(magnitudes, 0.90))[0]

            output_u_winds[i // group_frames_together, slice_idx] = np.median(np.array(curr_u_wind)[top_quartile_indices])
            output_v_winds[i // group_frames_together, slice_idx] = np.median(np.array(curr_v_wind)[top_quartile_indices])

    return output_u_winds, output_v_winds, output_times
