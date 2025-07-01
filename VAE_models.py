from functions import layer_dimensions
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


#What Works Now
#VAE, Encoder, Decoder: All inherit from nn.Module and have correct method signatures.
#Layer Construction: Linear layers are now built with correct input/output sizes.
#Forward Pass: The VAE can now encode, sample, and decode.

class Encoder(nn.Module):
    def __init__(self, in_dim, latent_dim, num_layers):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.layer_dims = layer_dimensions(in_dim, latent_dim * 2, num_layers) # times 2 to get mu, logvar 
        self.linears = nn.ModuleList([
            nn.Linear(self.layer_dims[i], self.layer_dims[i+1])
            for i in range(num_layers-1)
        ])
        self.activation_funcs = nn.ModuleList([nn.SiLU() for _ in range(num_layers-2)]) # (Num layers - 1) weight matrices, no activation function on last layer.
    def forward(self, x):
        for i, layer in enumerate(self.linears):
            x = (layer(x))
            if i < len(self.activation_funcs):
                x = self.activation_funcs[i](x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim, out_dim, num_layers):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.layer_dims = layer_dimensions(latent_dim, out_dim, num_layers)
        self.linears = nn.ModuleList([
            nn.Linear(self.layer_dims[i], self.layer_dims[i+1])
            for i in range(num_layers-1)
        ])
        self.activation_funcs = nn.ModuleList([nn.SiLU() for _ in range(num_layers-2)])
    def forward(self, x):
        for i, layer in enumerate(self.linears[:-1]):
            x = (layer(x))
            if i < len(self.activation_funcs):
                x = self.activation_funcs[i](x)
        x = self.linears[-1](x) # removed sigmoid because of amp numerical instability (see loss)
        return x

class VAE(nn.Module):
    def __init__(self, in_dim, latent_dim, n_layers, beta=1.0):
        super().__init__()
        self.encoder = Encoder(in_dim, latent_dim, n_layers)
        self.decoder = Decoder(latent_dim, in_dim, n_layers)
        self.softplus = nn.Softplus()
        self.beta = beta

    def encode(self, x, eps: float = 1e-8):
        encoded = self.encoder(x)
        mu, logvar = torch.chunk(encoded, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)
        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)
    
    def reparameterize(self, dist):
        return dist.rsample()
            
    def decode(self, z):
        decoded = self.decoder(z)
        return decoded

    def forward(self, x, compute_loss: bool = True):
        dist = self.encode(x)
        z = self.reparameterize(dist)
        recon_x = self.decode(z)

        if not compute_loss:
            return VAEOutput(
                z_dist=dist,
                z_sample=z,
                x_recon=torch.sigmoid(recon_x),
                loss=None,
                loss_recon=None,
                loss_kl=None,
            )
        # compute loss terms 
        loss_recon = F.binary_cross_entropy_with_logits(recon_x, x, reduction='none').sum(-1).mean()
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
        )
        loss_kl = self.beta * torch.distributions.kl.kl_divergence(dist, std_normal).mean()
                
        loss = loss_recon + loss_kl
        
        return VAEOutput(
            z_dist=dist,
            z_sample=z,
            x_recon=torch.sigmoid(recon_x),
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=loss_kl,
        )


@dataclass
class VAEOutput:
    """
    Dataclass for VAE output.
    
    Attributes:
        z_dist (torch.distributions.Distribution): The distribution of the latent variable z.
        z_sample (torch.Tensor): The sampled value of the latent variable z.
        x_recon (torch.Tensor): The reconstructed output from the VAE.
        loss (torch.Tensor): The overall loss of the VAE.
        loss_recon (torch.Tensor): The reconstruction loss component of the VAE loss.
        loss_kl (torch.Tensor): The KL divergence component of the VAE loss.
    """
    z_dist: torch.distributions.Distribution
    z_sample: torch.Tensor
    x_recon: torch.Tensor
    
    loss: torch.Tensor | None
    loss_recon: torch.Tensor | None
    loss_kl: torch.Tensor | None

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def load_vae_model(checkpoint_path, device):
    """Load VAE model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate model
    model = VAE(
        checkpoint['config']['in_dim'],
        checkpoint['config']['latent_dim'], 
        checkpoint['config']['n_layers']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint