import torch.nn as nn

class ResnetBlock1D(nn.Module):
    """1D Residual block."""
    def __init__(self, params):
        super().__init__()
        
        self.conv1d=nn.Conv1d(in_channels=params.hidden_size, out_channels=params.hidden, kernel_size=3, stride=1, padding=1)
        self.ff_layer=nn.Linear(in_features=params.hidden_size, out_features=params.hi)
        self.max_pool=nn.MaxPool1d(kernel_size=3, stride=2,padding=1)
        
        

    def forward(self, input_batch):
        
        out=input_batch

        return out
