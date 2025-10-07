import torch
import torch.nn as nn
import torch.nn.functional as F
from .embed import SinusoidalEmbeddingLayer

class ConvOccupancyGridEncoderLayer(nn.Module):
    """Uses the frame on the current step."""
    
    def __init__(self, 
                 params,
                 hidden_size):
        super().__init__()
        
        self.num_filters = params.num_conv_filters
        drop_prob = params.drop_prob
        layers = []
        in_channels = 3  # occ_grid (1) + coord_grid (2)
        
        for i, num_filter in enumerate(self.num_filters):
            if i == 0 or i == 1:
                strides = 2
                use_pooling = False  # Don't pool when using stride=2
            else:
                strides = 1
                use_pooling = False  # Pool when stride=1
            
            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_filter,
                kernel_size=3,
                stride=strides,
                padding=1
            )
    
            layers.append(conv_layer)
            layers.append(nn.BatchNorm2d(num_filter))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout2d(params.drop_prob))
            
            if use_pooling:
                pooling_layer = nn.MaxPool2d(kernel_size=2, stride=1)
                layers.append(pooling_layer)
            
            in_channels = num_filter
        
        # Flatten
        layers.append(nn.Flatten())
        layers.append(nn.LazyLinear(hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(drop_prob))
        layers.append(nn.LayerNorm(hidden_size))
        
        self.seq_layers = nn.Sequential(*layers)
    
    def forward(self, input_batch):
        input_batch = input_batch.copy()
        
        occ_grid = input_batch['scene/grid']
        coord_grid = input_batch['scene/coord']
        
        grid = torch.cat([occ_grid, coord_grid], dim=1)
        
        # Apply convolutional layers to grid
        occ_grid = self.seq_layers(grid)
        
        out = occ_grid

        return out


# To be modified to return 1 channel only
class ConvGridEncoderLayer(nn.Module):
    """Uses the frame on the current step."""
    
    def __init__(self, 
                 params):
        super().__init__()
        
        self.num_filters = params.num_conv_filters
        drop_prob = params.drop_prob
        layers = []
        in_channels = 1
        
        for i, num_filter in enumerate(self.num_filters):
            if i == 0 or i == 1:
                strides = 2
                use_pooling = False  
            else:
                strides = 1
                use_pooling = False
            
            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_filter,
                kernel_size=2,
                stride=1,
                padding=1
            )
    
            layers.append(conv_layer)
            layers.append(nn.BatchNorm2d(num_filter))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout2d(drop_prob))
            
            if use_pooling:
                pooling_layer = nn.MaxPool2d(kernel_size=2, stride=1)
                layers.append(pooling_layer)
            
            in_channels = num_filter
        
        self.seq_layers = nn.Sequential(*layers)
    
    def forward(self, input_batch):

        # input_batch = input_batch.copy()
        occ_grid = self.seq_layers(input_batch)
        
        return occ_grid

class TransformerOccupancyGridEncoderLayer(nn.Module):
    """Uses the frame on the current step."""
    
    def __init__(self, 
                 params,
                 prefilter=True,
                 hidden_size=128):
        super().__init__()
        
        self.patch_size=params.patch_size
        self.num_channels=1
        self.prefilter=prefilter

        if self.prefilter:
            self.filter=ConvGridEncoderLayer(params)
        else:
            self.filter=nn.Identity()
        
        self.occ_embedding_layer=nn.Linear(self.num_channels*(self.patch_size**2), hidden_size)
        
        self.sin_embedding_layer=SinusoidalEmbeddingLayer(hidden_size=512)
        self.orig_embedding_layer=nn.Linear(2*512, hidden_size)
        self.pre_lnorm=nn.LayerNorm(hidden_size)

    def _create_patches(self, grid):
        patches=grid.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches=patches.permute(0, 2, 3, 1, 4, 5)

        return patches

    def _process_occ_patches(self, occ_grid):
        
        b=occ_grid.shape[0]
        
        occ_patches=self._create_patches(occ_grid)
        flat_occ_patches=occ_patches.reshape(b, -1, self.num_channels * self.patch_size * self.patch_size)
        
        return flat_occ_patches

    def _create_coord_origins(self,coord_grid):
        
        coord_patches=self._create_patches(coord_grid)
        patch_origins = coord_patches[:, :, :, :, -1, 0] 
        b, num_h, num_w, _ = patch_origins.shape
        patch_origins = patch_origins.reshape(b, num_h * num_w, 2) 

        return patch_origins
    
    def forward(self, scene_grid, scene_coord):
        
        filtered_occ_grid=self.filter(scene_grid)
        flat_occ_patches=self._process_occ_patches(filtered_occ_grid)
        occ_embedding=self.occ_embedding_layer(flat_occ_patches)

    

        patch_origins=self._create_coord_origins(scene_coord)
        origin_embeddings=self.sin_embedding_layer(patch_origins)
        origin_embeddings=origin_embeddings.flatten(-2)
        origin_embeddings = self.orig_embedding_layer(origin_embeddings)

        out=occ_embedding+origin_embeddings
        out=self.pre_lnorm(out)

        return out

class PointCloudEncoderLayer(nn.Module):
    """Retrieves the point cloud at the current timestep."""
    
    def __init__(self, params):
        super().__init__()
        self.current_step_idx = params.num_history_steps
        self.embedding_layer = SinusoidalEmbeddingLayer(
            hidden_size=params.feature_embedding_size
        )
    
    def forward(self, input_batch, training=None):
        
        input_batch = input_batch.copy()
        pc = input_batch['scene/pc'][:, self.current_step_idx, ..., :2]
        pc = torch.where(torch.isnan(pc), torch.zeros_like(pc), pc)
        pc_emb = self.embedding_layer(pc)
        pc_emb = torch.cat([pc_emb[..., 0, :], pc_emb[..., 1, :]], dim=-1)
        input_batch['scene_hidden_vec'] = pc_emb
        
        return input_batch
