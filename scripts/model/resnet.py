import torch.nn as nn

class ResnetBlock1D(nn.Module):
    """1D Residual block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, pool=False):
        super().__init__()
        
        # First convolution layer
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        
        # Second convolution layer
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        
        # Skip connection to match dimensions if necessary
        if in_channels != out_channels:
            self.skip_connection = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
        
        self.activation = nn.SiLU()
        
        # Pooling layer
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()
        
    def forward(self, input_batch):
        
        b,a,t,h=input_batch.shape
        
        reshaped_input_batch=input_batch.transpose(2, 3).view(b*a, h, t)
        residual = self.skip_connection(reshaped_input_batch)
        out=self.norm1(reshaped_input_batch)
        out=self.conv1(out)
        out=self.norm2(out)
        out=self.conv2(out)
        
        out=out+residual

        out=self.activation(out)

        out=self.pool(out)
        # print(out.shape)
        h=out.shape[-2]
        t=out.shape[-1]
        out = out.permute(0, 2, 1).view(b, a, t, h)

        return out



class ResnetBlock1DConvUpsample(nn.Module):
    """1D Residual block with learnable upsampling using ConvTranspose1d."""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        
        # First convolution layer
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        
        # Second convolution layer
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        
        # Skip connection to match dimensions if necessary
        if in_channels != out_channels:
            self.skip_connection = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
        
        self.activation = nn.SiLU()
        
        # Upsampling layer using ConvTranspose1d (always upsample)
        self.upsample = nn.ConvTranspose1d(out_channels, out_channels, 
                                         kernel_size=2, stride=2, padding=0)
        
        # Skip connection upsampling
        if in_channels != out_channels:
            self.skip_upsample = nn.ConvTranspose1d(out_channels, out_channels,
                                                   kernel_size=2, stride=2, padding=0)
        else:
            self.skip_upsample = nn.ConvTranspose1d(in_channels, in_channels,
                                                   kernel_size=2, stride=2, padding=0)
        
    def forward(self, input_batch):
        
        b, a, t, h = input_batch.shape
        
        reshaped_input_batch = input_batch.transpose(2, 3).view(b*a, h, t)
        
        # Apply skip connection and upsample if needed
        residual = self.skip_connection(reshaped_input_batch)
        residual = self.skip_upsample(residual)
        
        # Forward pass through conv layers
        out = self.norm1(reshaped_input_batch)
        out = self.activation(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.conv2(out)
        
        # Upsample before adding residual
        out = self.upsample(out)
        
        # Add residual connection
        out = out + residual
        out = self.activation(out)
        
        h_new = out.shape[-2]
        t_new = out.shape[-1]
        out = out.permute(0, 2, 1).view(b, a, t_new, h_new)
        return out