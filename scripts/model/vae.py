import torch, torch.nn as nn, torch.nn.functional as F
from .encoder import HumanRobotInteractionEncoder
from .decoder import HumanRobotInteractionDecoder
from .head import LstmHead

class VAE(nn.Module):
    def __init__(self,
                 params):
        super().__init__()
        
        hidden_size=64
        self.encoder = HumanRobotInteractionEncoder(params)
        self.mu = nn.Linear(hidden_size, params.latent_dim)
        self.logvar = nn.Linear(hidden_size, params.latent_dim)
        
        self.decoder=HumanRobotInteractionDecoder(params)

    def reparametrize(self, mu, logvar):
        
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        z=mu+std*eps

        return z
    
    def forward(self,input_batch):
        
        enc_output=self.encoder(input_batch)
        mu=self.mu(enc_output)
        logvar=self.logvar(enc_output)

        z=self.reparametrize(mu, logvar)
        out=self.decoder(z)*6.425

        return out, mu, logvar
