"""Dataset class for loading cloud volumes and stereo camera images."""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from utils import project_volume_to_cameras, get_ray_embedding
from augmentor import ImageAugmentor

# Constants
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
DEFAULT_DEPTH = 1000
GAMMA = 1.0 / 2.2

# Air density lookup table for LWC conversion (height in meters -> density in kg/m^3)
AIR_DENSITY_HEIGHTS = [0, 1000, 2000, 3000, 4000]
AIR_DENSITIES = [1.225, 1.112, 1.007, 0.9093, 0.8194]


def load_camera_calibration(path, prefix='left'):
    """Load camera pose and intrinsics from numpy files."""
    camera_pose = torch.from_numpy(
        np.load(os.path.join(path, f'{prefix}_cam2world.npy'))
    ).to(torch.float32)
    camera_intrinsics = torch.from_numpy(
        np.load(os.path.join(path, f'{prefix}_intrinsic.npy'))
    ).to(torch.float32)
    return camera_pose, camera_intrinsics


def adjust_intrinsics_for_dino(intrinsics, patch_size=14):
    """Adjust camera intrinsics for DINO's patch-based processing.

    DINO processes images in patches of size patch_size x patch_size.
    The principal point needs to be adjusted to account for the cropped
    image dimensions (rounded down to nearest multiple of patch_size).

    Args:
        intrinsics: 3x3 camera intrinsic matrix
        patch_size: DINO patch size (default: 14)

    Returns:
        Adjusted intrinsics matrix
    """
    curr_w = intrinsics[0, 2] * 2
    curr_h = intrinsics[1, 2] * 2

    # Round down to nearest multiple of patch_size
    cropped_w = int((curr_w // patch_size) * patch_size)
    cropped_h = int((curr_h // patch_size) * patch_size)

    intrinsics[0, 2] = cropped_w / 2
    intrinsics[1, 2] = cropped_h / 2

    return intrinsics


def load_exr_image(path, apply_augmentation=False):
    """Load and tonemap an EXR image.

    Args:
        path: Path to the EXR file
        apply_augmentation: If True, randomize tonemapping parameters

    Returns:
        Tonemapped image as uint8 numpy array
    """
    bgr_color = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_UNCHANGED)
    rgb_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2RGB)

    inv_gamma = 1.0 / GAMMA

    if apply_augmentation:
        percentile = np.random.randint(80, 95)
        target_brightness = np.random.uniform(0.7, 0.9)
    else:
        percentile = 90
        target_brightness = 0.8

    # Compute brightness using CCIR601 YIQ method
    brightness = 0.3 * rgb_color[:, :, 0] + 0.59 * rgb_color[:, :, 1] + 0.11 * rgb_color[:, :, 2]
    brightness_percentile = np.percentile(brightness, percentile)

    if brightness_percentile < 1e-4:
        scale = 0.0
    else:
        scale = np.power(target_brightness, inv_gamma) / brightness_percentile

    rgb_tonemapped = np.power(np.maximum(scale * rgb_color, 0), GAMMA)
    image = np.clip(rgb_tonemapped, 0, 1)

    return (image * 255).astype(np.uint8)


def load_standard_image(path):
    """Load a standard image file (PNG, JPG, etc.)."""
    return np.array(Image.open(path))


class CloudDataset(data.Dataset):
    """Dataset for loading 3D cloud volumes with multi-view stereo images.

    Supports both synthetic (Terragen, MicroHH) and real-world (Chilbolton) data.
    """

    def __init__(
        self,
        grid_shape,
        voxel_sizes,
        volume_path,
        camera_folder_path,
        split='train',
        get_projection_indices=False,
        skip_gt_volume=False,
        augmentations=False,
        load_colour_corrected_images=False,
        exr_images=False,
        downsample_factor=1,
        file_idx_function=None,
        scale_volumes=None,
        grid_ranges=None,
        robust_chilbolton_loading=False,
    ):
        if grid_ranges is None:
            grid_ranges = [(-5000, 5000), (-5000, 5000), (0, 4000)]

        # Store configuration
        self.exr_images = exr_images
        self.downsample_factor = downsample_factor
        self.scale_volumes = scale_volumes
        self.grid_ranges = grid_ranges
        self.robust_chilbolton_loading = robust_chilbolton_loading
        self.camera_folder_path = camera_folder_path
        self.skip_gt_volume = skip_gt_volume
        self.file_index_function = file_idx_function
        self.augmentations = augmentations

        # Set up image directories
        if load_colour_corrected_images:
            self.left_dir = 'corrected_left_images'
            self.right_dir = 'corrected_right_images'
        else:
            self.left_dir = 'left_images'
            self.right_dir = 'right_images'

        # Initialize augmentors if needed
        if augmentations:
            self.image_augmentor = ImageAugmentor()

        # Initialize data storage
        self.left_camera_images = []
        self.right_camera_images = []
        self.files = []

        # Camera projection data
        self._projection_data = {
            'left_pixel_coords': [],
            'left_valid_coords': [],
            'left_grid_depths': [],
            'left_plucker_rays': [],
            'right_pixel_coords': [],
            'right_valid_coords': [],
            'right_grid_depths': [],
            'right_plucker_rays': [],
            'left_camera_poses': [],
            'left_camera_intrinsics': [],
            'right_camera_poses': [],
            'right_camera_intrinsics': [],
        }

        # Load data based on configuration
        if robust_chilbolton_loading:
            self._init_robust_chilbolton(camera_folder_path)

        self._load_camera_views(
            camera_folder_path, split, get_projection_indices,
            grid_shape, voxel_sizes, grid_ranges
        )

        if get_projection_indices:
            self._stack_projection_data()

        if not skip_gt_volume:
            self._load_volume_files(volume_path, split)

    def _init_robust_chilbolton(self, camera_folder_path):
        """Initialize timestamp list for robust Chilbolton loading."""
        all_sets = []
        for view in sorted(os.listdir(camera_folder_path)):
            if '.txt' in view:
                continue

            for side in [self.left_dir, self.right_dir]:
                dir_path = os.path.join(camera_folder_path, view, side)
                if os.path.exists(dir_path):
                    timestamps = {f.split('.')[0] for f in os.listdir(dir_path)}
                    all_sets.append(timestamps)

        self.all_times = sorted(set.union(*all_sets)) if all_sets else []

    def _should_include_view(self, view):
        """Check if a view should be included based on configuration."""
        if '.txt' in view:
            return False
        # Skip non-camera directories
        if view in ['volumes', '5km_crop', 'depth_maps']:
            return False
        return True

    def _get_image_dir(self, view, side):
        """Get the appropriate image directory for a view."""
        return self.left_dir if side == 'left' else self.right_dir

    def _load_camera_views(self, camera_folder_path, split, get_projection_indices,
                           grid_shape, voxel_sizes, grid_ranges):
        """Load image paths and projection data for all camera views."""
        for view in sorted(os.listdir(camera_folder_path)):
            if not self._should_include_view(view):
                continue

            view_path = os.path.join(camera_folder_path, view)

            # Load image file paths
            if not self.robust_chilbolton_loading:
                self._load_view_images(view_path, view, split)

            # Compute projection indices if requested
            if get_projection_indices:
                self._compute_projection_indices(
                    view_path, grid_shape, voxel_sizes, grid_ranges
                )

    def _load_view_images(self, view_path, view, split):
        """Load image file paths for a single view."""
        left_dir = self._get_image_dir(view, 'left')
        right_dir = self._get_image_dir(view, 'right')

        left_files = sorted(os.listdir(os.path.join(view_path, left_dir)))
        right_files = sorted(os.listdir(os.path.join(view_path, right_dir)))

        left_paths, right_paths = [], []

        for i in range(len(left_files)):
            # Train/val split: use 90% for training, 10% for validation
            if split == 'train' and i % 10 == 0:
                continue
            elif split == 'val' and i % 10 != 0:
                continue

            left_paths.append(os.path.join(view_path, left_dir, left_files[i]))
            right_paths.append(os.path.join(view_path, right_dir, right_files[i]))

        self.left_camera_images.append(left_paths)
        self.right_camera_images.append(right_paths)

    def _compute_projection_indices(self, view_path, grid_shape, voxel_sizes, grid_ranges):
        """Compute projection indices for a camera view."""
        assert voxel_sizes[0] == voxel_sizes[1] == voxel_sizes[2], \
            "Voxel sizes must be equal in all dimensions"

        left_pose, left_intrinsics = load_camera_calibration(view_path, 'left')
        right_pose, right_intrinsics = load_camera_calibration(view_path, 'right')

        # Adjust intrinsics for DINO's patch-based processing
        left_intrinsics = adjust_intrinsics_for_dino(left_intrinsics)
        right_intrinsics = adjust_intrinsics_for_dino(right_intrinsics)

        # Compute projection coordinates
        left_coords, left_valid, left_depths = project_volume_to_cameras(
            grid_shape, left_pose, left_intrinsics,
            voxel_size=voxel_sizes[0], grid_ranges=grid_ranges
        )
        right_coords, right_valid, right_depths = project_volume_to_cameras(
            grid_shape, right_pose, right_intrinsics,
            voxel_size=voxel_sizes[0], grid_ranges=grid_ranges
        )

        # Compute Plucker ray embeddings (clone to avoid in-place modification issues)
        left_rays = get_ray_embedding(
            left_intrinsics.unsqueeze(0).clone(),
            left_pose.unsqueeze(0).clone(),
            IMAGE_HEIGHT, IMAGE_WIDTH
        )
        right_rays = get_ray_embedding(
            right_intrinsics.unsqueeze(0).clone(),
            right_pose.unsqueeze(0).clone(),
            IMAGE_HEIGHT, IMAGE_WIDTH
        )

        # Store projection data
        self._projection_data['left_pixel_coords'].append(left_coords)
        self._projection_data['left_valid_coords'].append(left_valid)
        self._projection_data['left_grid_depths'].append(left_depths)
        self._projection_data['left_plucker_rays'].append(left_rays)
        self._projection_data['right_pixel_coords'].append(right_coords)
        self._projection_data['right_valid_coords'].append(right_valid)
        self._projection_data['right_grid_depths'].append(right_depths)
        self._projection_data['right_plucker_rays'].append(right_rays)
        self._projection_data['left_camera_poses'].append(left_pose)
        self._projection_data['left_camera_intrinsics'].append(left_intrinsics)
        self._projection_data['right_camera_poses'].append(right_pose)
        self._projection_data['right_camera_intrinsics'].append(right_intrinsics)

    def _stack_projection_data(self):
        """Stack projection data lists into tensors and set as attributes."""
        for key, value in self._projection_data.items():
            if value:
                setattr(self, key, torch.stack(value, dim=0))

    def _load_volume_files(self, volume_path, split):
        """Load volume file paths, matching with available images."""
        if self.exr_images:
            file_indices = self._get_exr_file_indices()
        else:
            file_indices = {
                int(x.split('/')[-1].split('.')[0][:4])
                for x in self.left_camera_images[0]
            }

        for i, volume_file in enumerate(sorted(os.listdir(volume_path))):
            if self.exr_images:
                prefix = volume_file.split('_')[0]
                volume_index = int(volume_file.split('_')[1].split('.')[0])
                if volume_index in file_indices.get(prefix, set()):
                    self.files.append(os.path.join(volume_path, volume_file))
            else:
                # Apply same train/val split as images
                if split == 'train' and i % 10 == 0:
                    continue
                elif split == 'val' and i % 10 != 0:
                    continue
                self.files.append(os.path.join(volume_path, volume_file))

    def _get_exr_file_indices(self):
        """Extract file indices from EXR image paths."""
        indices = {'cabauw': set(), 'arm': set(), 'rico': set(),
                   'jaenschwalde': set(), 'bomex': set()}
        for path in self.left_camera_images[0]:
            filename = path.split('/')[-1]
            prefix = filename.split('_')[0]
            idx = int(filename.split('.')[0].split('_')[1])
            if prefix in indices:
                indices[prefix].add(idx)
        return indices

    def __len__(self):
        if self.robust_chilbolton_loading:
            return len(self.all_times)
        return len(self.left_camera_images[0])

    def _load_volume(self, index):
        """Load and preprocess a volume at the given index."""
        # Determine the correct key for the numpy file
        key = 'arr_0' if (self.exr_images and self.grid_ranges[0][0] == -5000) else 'a'
        volume = torch.from_numpy(np.load(self.files[index])[key]).to(torch.float32)

        if self.downsample_factor > 1:
            volume = torch.nn.functional.interpolate(
                volume.unsqueeze(0).unsqueeze(0),
                scale_factor=1 / self.downsample_factor,
                mode='trilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)

        if self.scale_volumes is not None:
            volume *= self.scale_volumes

        # Convert mixing ratio to liquid water content for MicroHH data
        if self.exr_images:
            target_heights = np.linspace(0, 4000, volume.shape[-1])
            air_densities = np.interp(target_heights, AIR_DENSITY_HEIGHTS, AIR_DENSITIES)
            air_densities = torch.from_numpy(air_densities).to(torch.float32)
            volume *= air_densities[None, None, :]

        return volume

    def _load_image(self, path):
        """Load an image from path, handling EXR and standard formats."""
        if self.exr_images:
            return load_exr_image(path, apply_augmentation=self.augmentations)
        return load_standard_image(path)

    def _load_depths(self):
        """Return placeholder depth maps (depths are not used by the model)."""
        return torch.ones((3, IMAGE_HEIGHT, IMAGE_WIDTH)) * DEFAULT_DEPTH

    def _load_images_standard(self, index):
        """Load stereo images using standard loading."""
        def load_camera_images(camera_paths):
            images = []
            for cam_paths in camera_paths:
                img = self._load_image(cam_paths[index])
                if self.augmentations:
                    img = self.image_augmentor(img)
                images.append(torch.from_numpy(img).permute(2, 0, 1))
            return torch.stack(images, dim=0)

        left_images = load_camera_images(self.left_camera_images)
        right_images = load_camera_images(self.right_camera_images)
        depths = self._load_depths()

        return left_images, right_images, depths, torch.zeros(3), torch.zeros(3)

    def _load_images_robust(self, index):
        """Load stereo images with robust Chilbolton loading (handles missing images)."""
        timestamp = self.all_times[index]
        views = ['perspective_1', 'perspective_2', 'perspective_3']

        left_images, right_images = [], []
        dropped_left = torch.zeros(3)
        dropped_right = torch.zeros(3)

        for idx, view in enumerate(views):
            # Load left image
            left_path = os.path.join(
                self.camera_folder_path, view, 'left_images', f'{timestamp}.png'
            )
            if os.path.exists(left_path):
                img = torch.from_numpy(self._load_image(left_path)).permute(2, 0, 1)
                left_images.append(img)
            else:
                left_images.append(torch.zeros((3, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.uint8))
                dropped_left[idx] = 1

            # Load right image
            right_path = os.path.join(
                self.camera_folder_path, view, 'right_images', f'{timestamp}.png'
            )
            if os.path.exists(right_path):
                img = torch.from_numpy(self._load_image(right_path)).permute(2, 0, 1)
                right_images.append(img)
            else:
                right_images.append(torch.zeros((3, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.uint8))
                dropped_right[idx] = 1

        left_images = torch.stack(left_images, dim=0)
        right_images = torch.stack(right_images, dim=0)
        depths = torch.ones((3, IMAGE_HEIGHT, IMAGE_WIDTH)) * DEFAULT_DEPTH

        return left_images, right_images, depths, dropped_left, dropped_right

    def _get_vdb_index(self, index):
        """Get the VDB index for a given sample index."""
        if self.robust_chilbolton_loading:
            timestamp = self.all_times[index]
            parts = timestamp.split('_')
            return torch.tensor(int(parts[0][-3:] + parts[1])).to(torch.int64)

        if self.file_index_function is not None:
            return torch.tensor(
                int(self.file_index_function(self.left_camera_images[0][index]))
            ).to(torch.int32)

        filename = self.left_camera_images[0][index].split('/')[-1]
        if self.exr_images:
            idx = int(filename.split('.')[0].split('_')[1])
        else:
            idx = int(filename.split('.')[0][:4])
        return torch.tensor(idx).to(torch.int32)

    def __getitem__(self, index):
        # Load volume if needed
        volume = self._load_volume(index) if not self.skip_gt_volume else None

        # Load images based on loading mode
        if self.robust_chilbolton_loading:
            left_images, right_images, depths, dropped_left, dropped_right = \
                self._load_images_robust(index)
        else:
            left_images, right_images, depths, dropped_left, dropped_right = \
                self._load_images_standard(index)

        # Normalize images to [0, 1] float
        if left_images.dtype == torch.uint8:
            left_images = left_images.to(torch.float32) / 255.0
            right_images = right_images.to(torch.float32) / 255.0
        elif left_images.dtype != torch.float32:
            raise ValueError('Images should be float32 or uint8')

        depths = depths.to(torch.float32)

        # Get VDB index for tracking
        vdb_index = self._get_vdb_index(index)

        # Select camera views
        indices = torch.arange(3)

        left_images = left_images[indices, :3, ...]
        right_images = right_images[indices, :3, ...]
        depths = depths[indices, ...]

        output = {
            'left_images': left_images,
            'right_images': right_images,
            'depths': depths,
            'camera_indices': indices,
            'vdb_index': vdb_index,
            'dropped_left_cameras': dropped_left,
            'dropped_right_cameras': dropped_right,
        }

        if not self.skip_gt_volume:
            output['volumes'] = volume

        return output
