import torch
import torch.nn as nn
import torch.nn.functional as F

class GridPreprocessLayer(nn.Module):
    def __init__(self, params,scale_coords=False):
        super().__init__()
        self.output_size = params.grid_output_size
        self.coord_scale = params.grid_coord_scale
        self.scale_coords=scale_coords
        # Can add learnable parameters here if needed
        
    def create_coordinate_grid(self, batch_size, height, width, origin, resolution, device):
        """Create coordinate grids for a batch of maps."""
        # Create pixel coordinate grids
        y_coords = torch.arange(height, device=device, dtype=torch.float32)
        x_coords = torch.arange(width, device=device, dtype=torch.float32)

        pixel_y, pixel_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        pixel_x = pixel_x.unsqueeze(0).expand(batch_size, -1, -1)
        pixel_y = pixel_y.unsqueeze(0).expand(batch_size, -1, -1)
        origin_x = origin[:, 0:1].unsqueeze(-1)  
        origin_y = origin[:, 1:2].unsqueeze(-1)
        # print(origin_x.shape)

        resolution = resolution.unsqueeze(-1).unsqueeze(-1) 
        
        world_x = origin_x + pixel_x * resolution
        world_y = origin_y + (height - 1 - pixel_y) * resolution
        
        coord_grid = torch.stack([world_x, world_y], dim=1)
        
        if self.scale_coords:
            coord_grid=coord_grid / self.coord_scale
        else:
            coord_grid=coord_grid

        return coord_grid
    
    def forward(self, input_batch):
        
        map_images = input_batch['map_image']
        origins = input_batch['map_origin']
        resolutions = input_batch['map_resolution']
        
        batch_size = map_images.shape[0]
        device = map_images.device
        
        # scene_grid = F.interpolate(
        #     map_images, 
        #     size=self.output_size, 
        #     mode='bilinear', 
        #     align_corners=False
        # )

        # scene_grid = (scene_grid - 0.5) / 0.5
        
        scene_coord = self.create_coordinate_grid(
            batch_size,
            self.output_size[0], 
            self.output_size[1],
            origins,
            resolutions,
            device
        )

        scene_grid=input_batch['map_image']

        
        return scene_grid, scene_coord
    

# class GridPreprocessLayer(nn.Module):
#     def __init__(self, params):
#         super().__init__()
#         self.output_size = params.grid_output_size
#         self.coord_scale = params.grid_coord_scale
#         self.patch_size = 16  
        
#     def create_coordinate_grid(self, batch_size, height, width, origin, resolution, device):
#         """Create coordinate grids for a batch of maps."""
#         # Create pixel coordinate grids
#         y_coords = torch.arange(height, device=device, dtype=torch.float32)
#         x_coords = torch.arange(width, device=device, dtype=torch.float32)
#         pixel_y, pixel_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
#         pixel_x = pixel_x.unsqueeze(0).expand(batch_size, -1, -1)
#         pixel_y = pixel_y.unsqueeze(0).expand(batch_size, -1, -1)
        
#         origin_x = origin[:, 0:1].unsqueeze(-1)  
#         origin_y = origin[:, 1:2].unsqueeze(-1)
#         resolution = resolution.unsqueeze(-1).unsqueeze(-1) 
        
#         world_x = origin_x + pixel_x * resolution
#         world_y = origin_y + (height - 1 - pixel_y) * resolution
        
#         coord_grid = torch.stack([world_x, world_y], dim=1)
        
#         return coord_grid / self.coord_scale
    
#     def world_to_pixel_coords(self, world_pos, origins, resolutions, height):
#         """Convert world coordinates to pixel coordinates in the interpolated grid."""
#         # world_pos: [batch, 2] (x, y)
#         print("World", world_pos.shape)
#         print("Origin:",origins.shape)

#         pixel_x = (world_pos[:, 0] - origins[:, 0]) / resolutions
#         pixel_y = height - 1 - (world_pos[:, 1] - origins[:, 1]) / resolutions
#         return pixel_x, pixel_y
    
#     def get_patch_center_coords(self, pixel_x, pixel_y, grid_height, grid_width):
#         """Get patch indices and their center coordinates in pixel space."""
#         n_patches_h = grid_height // self.patch_size
#         n_patches_w = grid_width // self.patch_size
        
#         # Clamp to valid range
#         pixel_x = torch.clamp(pixel_x, 0, grid_width - 1)
#         pixel_y = torch.clamp(pixel_y, 0, grid_height - 1)
        
#         patch_x = (pixel_x / self.patch_size).long()
#         patch_y = (pixel_y / self.patch_size).long()
        
#         # Clamp to valid patch indices
#         patch_x = torch.clamp(patch_x, 0, n_patches_w - 1)
#         patch_y = torch.clamp(patch_y, 0, n_patches_h - 1)
        
#         # Calculate center pixel coordinates of the patch
#         # Account for boundaries to ensure we can get a 3x3 neighborhood
#         center_patch_x = torch.clamp(patch_x, 1, n_patches_w - 2)
#         center_patch_y = torch.clamp(patch_y, 1, n_patches_h - 2)
        
#         # Center of the patch in pixel coordinates
#         center_pixel_x = (center_patch_x.float() + 0.5) * self.patch_size
#         center_pixel_y = (center_patch_y.float() + 0.5) * self.patch_size
        
#         return patch_x, patch_y, center_pixel_x, center_pixel_y
    
#     def extract_patch_neighborhood(self, scene_grid, center_pixel_x, center_pixel_y):
#         """Extract 3x3 patch neighborhood using grid_sample."""
#         batch_size, channels, height, width = scene_grid.shape
        
#         # Target is 3x3 patches = 48x48 pixels
#         target_h = 3 * self.patch_size
#         target_w = 3 * self.patch_size
        
#         # Create a sampling grid relative to center
#         # grid_sample expects coordinates in [-1, 1] range
#         y_offset = torch.linspace(-target_h/2, target_h/2, target_h, device=scene_grid.device)
#         x_offset = torch.linspace(-target_w/2, target_w/2, target_w, device=scene_grid.device)
#         grid_y, grid_x = torch.meshgrid(y_offset, x_offset, indexing='ij')
        
#         # Add center position for each batch element
#         # [batch, height, width, 2]
#         sampling_grid = torch.stack([
#             grid_x.unsqueeze(0).expand(batch_size, -1, -1),
#             grid_y.unsqueeze(0).expand(batch_size, -1, -1)
#         ], dim=-1)
        
#         # Add center offsets
#         sampling_grid[:, :, :, 0] += center_pixel_x.view(-1, 1, 1)
#         sampling_grid[:, :, :, 1] += center_pixel_y.view(-1, 1, 1)
        
#         # Normalize to [-1, 1] range for grid_sample
#         sampling_grid[:, :, :, 0] = 2.0 * sampling_grid[:, :, :, 0] / (width - 1) - 1.0
#         sampling_grid[:, :, :, 1] = 2.0 * sampling_grid[:, :, :, 1] / (height - 1) - 1.0
        
#         # Extract subgrids using grid_sample
#         scene_subgrid = F.grid_sample(
#             scene_grid,
#             sampling_grid,
#             mode='bilinear',
#             padding_mode='border',
#             align_corners=True
#         )
        
#         return scene_subgrid
    
#     def create_subgrid_coordinates(self, batch_size, center_pixel_x, center_pixel_y, 
#                                             origins, resolutions, device):
#         """Create coordinate grids for the extracted subgrids."""
#         target_h = 3 * self.patch_size
#         target_w = 3 * self.patch_size
        
#         # Create relative pixel coordinates
#         y_offset = torch.arange(target_h, device=device, dtype=torch.float32) - target_h / 2
#         x_offset = torch.arange(target_w, device=device, dtype=torch.float32) - target_w / 2
#         grid_y, grid_x = torch.meshgrid(y_offset, x_offset, indexing='ij')
        
#         # Expand for batch
#         grid_y = grid_y.unsqueeze(0).expand(batch_size, -1, -1)
#         grid_x = grid_x.unsqueeze(0).expand(batch_size, -1, -1)
        
#         # Add center positions
#         pixel_y = grid_y + center_pixel_y.view(-1, 1, 1)
#         pixel_x = grid_x + center_pixel_x.view(-1, 1, 1)
        
#         # Convert to world coordinates
#         origin_x = origins[:, 0].view(-1, 1, 1)
#         origin_y = origins[:, 1].view(-1, 1, 1)
#         resolution = resolutions.view(-1, 1, 1)
        
#         full_height = self.output_size[0]
#         world_x = origin_x + pixel_x * resolution
#         world_y = origin_y + (full_height - 1 - pixel_y) * resolution
        
#         coord_grid = torch.stack([world_x, world_y], dim=1)
        
#         return coord_grid / self.coord_scale
    
#     def forward(self, input_batch):
#         map_images = input_batch['map_image']
#         origins = input_batch['map_origin']
#         resolutions = input_batch['map_resolution']
#         human_pos = input_batch['human_pos']  # [batch, time, 2]

#         batch_size = map_images.shape[0]
#         device = map_images.device
        
#         # Interpolate full scene grid
#         scene_grid = F.interpolate(
#             map_images, 
#             size=self.output_size, 
#             mode='bilinear', 
#             align_corners=False
#         )
#         scene_grid = (scene_grid - 0.5) / 0.5
        
#         # Get the last human position across time dimension
#         human_pos_last = human_pos[:, -1, :]  # [batch, 2]
        
#         # Convert world coordinates to pixel coordinates in interpolated grid
#         pixel_x, pixel_y = self.world_to_pixel_coords(
#             human_pos_last, origins, resolutions, self.output_size[0]
#         )
        
#         # Get patch indices and center coordinates
#         patch_x, patch_y, center_pixel_x, center_pixel_y = self.get_patch_center_coords(
#             pixel_x, pixel_y, self.output_size[0], self.output_size[1]
#         )
        
#         # Extract 3x3 patch neighborhood
#         scene_subgrid = self.extract_patch_neighborhood(
#             scene_grid, center_pixel_x, center_pixel_y
#         )
        
#         # Create coordinate grid for full scene
#         scene_coord = self.create_coordinate_grid(
#             batch_size,
#             self.output_size[0], 
#             self.output_size[1],
#             origins,
#             resolutions,
#             device
#         )
        
#         # Create coordinate grid for subgrid
#         scene_subgrid_coord = self.create_subgrid_coordinates(
#             batch_size, center_pixel_x, center_pixel_y, origins, resolutions, device
#         )
        
#         out = input_batch.copy()
#         out['scene/grid'] = scene_grid  # Full grid
#         out['scene/coord'] = scene_coord  # Full grid coordinates
#         out['scene/subgrid'] = scene_subgrid  # 3x3 patch neighborhood (48x48)
#         out['scene/subgrid_coord'] = scene_subgrid_coord  # Subgrid coordinates
#         out['scene/patch_indices'] = torch.stack([patch_x, patch_y], dim=1)  # Center patch indices
        
#         return out