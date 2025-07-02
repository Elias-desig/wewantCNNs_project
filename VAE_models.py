from functions import layer_dimensions, conv_dimension
from dataclasses import dataclass
import os
from datetime import datetime
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
        loss_recon = F.binary_cross_entropy_with_logits(recon_x, x, reduction='none')
        if x.ndim == 4:
            loss_recon = loss_recon.sum(dim=(1,2,3)).mean() # sum over all dims except batch for conv output
        else:
            loss_recon = loss_recon.sum(-1).mean() # sum over last dimension for flat output
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

class CVAE_Encoder(nn.Module):
    def __init__(self, image_size, latent_dim):
        super().__init__()
        self.image_size = image_size # assumed to [w, h], grayscale
        self.latent_dim = latent_dim
        self.covn_1 = nn.Conv2d(1, 32, 3, stride=2, padding=1) # output: (Batch, 32, image_h / 2, image_w / 2)
        self.covn_2 = nn.Conv2d(32, 64, 3, stride=2, padding=1) # output: (Batch, 64, image_h / 4, image_w / 4)
        self.covn_3 = nn.Conv2d(64, 128, 3, stride=2, padding=1) # output: (Batch, 128, image_h / 8, image_w / 8)
        
        size1 = conv_dimension(*image_size, padding=1, stride=2, kernel_size=3)
        size2 = conv_dimension(*size1, padding=1, stride=2, kernel_size=3)  
        size3 = conv_dimension(*size2, padding=1, stride=2, kernel_size=3)
        

        self.linear = nn.Linear(128 * size3[0] * size3[1], latent_dim * 2)
    def forward(self, x):
        x = F.silu(self.covn_1(x))
        x = F.silu(self.covn_2(x))
        x = F.silu(self.covn_3(x))
        x = torch.flatten(x, start_dim=1) # preserve batch dim
        x = self.linear(x)
        return x
class CVAE_Decoder(nn.Module):
    def __init__(self, image_size, latent_dim):
        super().__init__()
        self.image_size = image_size # assumed to [w, h], grayscale
        self.latent_dim = latent_dim
        self.conv_h = image_size[0] // 8
        self.conv_w = (image_size[1] + 7) // 8

        self.size1 = conv_dimension(*image_size, padding=1, stride=2, kernel_size=3)
        self.size2 = conv_dimension(*self.size1, padding=1, stride=2, kernel_size=3)  

        self.linear = nn.Linear(latent_dim, 128 * self.conv_h * self.conv_w)
        self.decovn_1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)  # output: (Batch, 64, image_h / 4, image_w / 4)
        self.decovn_2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1) # output: (Batch, 32, image_h /2, image_w / 2)
        self.decovn_3 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1) # output: (Batch, 1, image_h, image_w2)
    def forward(self, x):
        batch_size = x.size(0)
        x = self.linear(x)
        x = x.view(-1, 128, self.conv_h, self.conv_w)
        x = F.silu(self.decovn_1(x, output_size=(batch_size, 64, *self.size2)))
        x = F.silu(self.decovn_2(x, output_size=(batch_size, 32, *self.size1)))
        x = self.decovn_3(x, output_size=(x.size(0), 1, *self.image_size))

        return x
class CVAE(VAE):
    def __init__(self, image_size, latent_dim, beta=1.0):
        super().__init__(in_dim=1, latent_dim=latent_dim, n_layers=2, beta=beta)
        self.encoder = CVAE_Encoder(image_size, latent_dim)
        self.decoder = CVAE_Decoder(image_size, latent_dim)


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
    
def load_vae_model(checkpoint_path, device, conv):
    """Load VAE model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if conv:
        model = CVAE(
            checkpoint['config']['image_size'],
            checkpoint['config']['latent_dim'], 
        ).to(device)
    else:
        model = VAE(
            checkpoint['config']['in_dim'],
            checkpoint['config']['latent_dim'], 
            checkpoint['config']['n_layers']
        ).to(device)        
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint

def save_checkpoint(model, optimizer, epoch, loss, config, conv, in_dims):
    if not os.path.isdir('./models'):
        os.mkdir('./models')
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if conv:
        if not os.path.isdir('./models/conv_vae'):
            os.mkdir('./models/conv_vae')
        checkpoint ={
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'config': {
                        'image_size': in_dims,
                        'latent_dim': config.vae.latent_dim,
                        'batch_size': config.vae.batch_size,
                        'lr': config.vae.lr
                    },
                    'timestamp': timestamp            
                    }
        torch.save(checkpoint, f'./models/conv_vae/c_vae_checkpoint_{timestamp}.pt')    
    else:
        if not os.path.isdir('./models/vae'):
            os.mkdir('./models/vae')            
        checkpoint ={
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'config': {
                        'in_dim': in_dims,
                        'latent_dim': config.vae.latent_dim,
                        'n_layers': config.vae.n_layers,
                        'batch_size': config.vae.batch_size,
                        'lr': config.vae.lr
                    },
                    'timestamp': timestamp            
                    }
        torch.save(checkpoint, f'./models/vae/vae_checkpoint_{timestamp}.pt')   