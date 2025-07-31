try:
    from .functions import layer_dimensions, conv_dimension
except ImportError:
    from functions import layer_dimensions, conv_dimension
from dataclasses import dataclass
import os
from datetime import datetime
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


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
        std = (0.5 * logvar).exp() + eps
        return torch.distributions.Independent(torch.distributions.Normal(mu, std), 1)
    
    def reparameterize(self, dist):
        return dist.rsample()
            
    def decode(self, z):
        decoded = self.decoder(z)
        return decoded

    def forward(self, x, compute_loss: bool = True):
        dist = self.encode(x)
        z = self.reparameterize(dist)
        recon_logits = self.decode(z)

        if not compute_loss:
            return VAEOutput(
                z_dist=dist,
                z_sample=z,
                x_recon=torch.sigmoid(recon_logits),
                loss=None,
                loss_recon=None,
                loss_kl=None,
            )
        # Use a combination of losses for better reconstruction
        recon_x = torch.sigmoid(recon_logits)
        
        # L1 loss (less blurry than MSE)
        loss_l1 = F.l1_loss(recon_x, x, reduction='none')
        
        # Spectral loss (preserve frequency content)
        fft_real = torch.fft.rfft2(recon_x)
        fft_target = torch.fft.rfft2(x)
        loss_spectral = F.mse_loss(fft_real.real, fft_target.real) + F.mse_loss(fft_real.imag, fft_target.imag)
        
        # Combine losses
        if x.ndim == 4:
            loss_recon = loss_l1.sum(dim=(1,2,3)).mean() + 0.1 * loss_spectral
        else:
            loss_recon = loss_l1.sum(-1).mean() + 0.1 * loss_spectral
        std_normal = torch.distributions.Independent(
            torch.distributions.Normal(
                torch.zeros_like(z), torch.ones_like(z)
            ), 1
        )
        loss_kl = self.beta * torch.distributions.kl.kl_divergence(dist, std_normal).mean()
                
        loss = loss_recon + loss_kl
        
        return VAEOutput(
            z_dist=dist,
            z_sample=z,
            x_recon=torch.sigmoid(recon_logits),
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=loss_kl,
        )
    def set_beta(self, beta):
        self.beta = beta
class CVAE_Encoder(nn.Module):
    def __init__(self, image_size, latent_dim):
        super().__init__()
        self.image_size = image_size # assumed to [w, h], grayscale
        self.latent_dim = latent_dim
        self.covn_1 = spectral_norm(nn.Conv2d(1, 32, 3, stride=2, padding=1)) # output: (Batch, 32, image_h / 2, image_w / 2)
        self.covn_2 = spectral_norm(nn.Conv2d(32, 64, 3, stride=2, padding=1)) # output: (Batch, 64, image_h / 4, image_w / 4)
        self.covn_3 = spectral_norm(nn.Conv2d(64, 128, 3, stride=2, padding=1)) # output: (Batch, 128, image_h / 8, image_w / 8)
        
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
        self.image_size = image_size # [w, h], grayscale
        self.latent_dim = latent_dim
        self.conv_h = image_size[0] // 8
        self.conv_w = (image_size[1] + 7) // 8

        self.size1 = conv_dimension(*image_size, padding=1, stride=2, kernel_size=3)
        self.size2 = conv_dimension(*self.size1, padding=1, stride=2, kernel_size=3) 
        self.size3 = conv_dimension(*self.size2, padding=1, stride=2, kernel_size=3) 

        self.linear = nn.Linear(latent_dim, 128 * self.size3[0] * self.size3[1])
        self.decovn_1 = spectral_norm(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1))  # output: (Batch, 64, image_h / 4, image_w / 4)
        self.decovn_2 = spectral_norm(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)) # output: (Batch, 32, image_h /2, image_w / 2)
        self.decovn_3 = spectral_norm(nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)) # output: (Batch, 1, image_h, image_w2)
    def forward(self, x):
        batch_size = x.size(0)
        x = self.linear(x)
        x = x.view(-1, 128, self.size3[0], self.size3[1])
        x = F.silu(self.decovn_1(x, output_size=(batch_size, 64, *self.size2)))
        x = F.silu(self.decovn_2(x, output_size=(batch_size, 32, *self.size1)))
        x = self.decovn_3(x, output_size=(x.size(0), 1, *self.image_size))

        return x
class CVAE(VAE):
    def __init__(self, image_size, latent_dim, beta=1.0):
        super().__init__(in_dim=1, latent_dim=latent_dim, n_layers=2, beta=beta)
        self.encoder = CVAE_Encoder(image_size, latent_dim)
        self.decoder = CVAE_Decoder(image_size, latent_dim)

class CVAE_Encoder_Deep(nn.Module):
    def __init__(self, image_size, latent_dim):
        super().__init__()
        self.image_size = image_size  # [128, 172]
        self.latent_dim = latent_dim
        
        # Calculate dimensions through the network
        # Input: 128×172
        self.conv1 = spectral_norm(nn.Conv2d(1, 64, 4, stride=2, padding=1))    # 64×86
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1))  # 32×43
        self.conv3 = spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1))  # 16×22 (note: 43//2 + 1 = 22)
        self.conv4 = spectral_norm(nn.Conv2d(256, 512, 4, stride=2, padding=1))  # 8×11
        
        # Calculate final feature map size
        size1 = conv_dimension(*image_size, padding=1, stride=2, kernel_size=4) # 64×86
        size2 = conv_dimension(*size1, padding=1, stride=2, kernel_size=4) # 32×43  
        size3 = conv_dimension(*size2, padding=1, stride=2, kernel_size=4) # 16×22
        size4 = conv_dimension(*size3, padding=1, stride=2, kernel_size=4) # 8×11
        
        self.final_size = size4
        
        # Group normalization layers
        self.gn1 = nn.GroupNorm(8, 64)
        self.gn2 = nn.GroupNorm(16, 128)
        self.gn3 = nn.GroupNorm(32, 256)
        self.gn4 = nn.GroupNorm(32, 512)
        
        # Linear layer for mu and logvar
        self.linear = nn.Linear(512 * size4[0] * size4[1], latent_dim * 2)
        
    def forward(self, x):
        x = F.silu(self.gn1(self.conv1(x)))
        x = F.silu(self.gn2(self.conv2(x)))
        x = F.silu(self.gn3(self.conv3(x)))
        x = F.silu(self.gn4(self.conv4(x)))
        
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x

class CVAE_Decoder_Deep(nn.Module):
    def __init__(self, image_size, latent_dim):
        super().__init__()
        self.image_size = image_size  # [128, 172]
        self.latent_dim = latent_dim
        
        # Calculate the sizes we need to hit at each upsampling step
        # We want to go: 8×11 -> 16×22 -> 32×43 -> 64×86 -> 128×172
        size1 = conv_dimension(*image_size, padding=1, stride=2, kernel_size=4)      # 64×86
        size2 = conv_dimension(*size1, padding=1, stride=2, kernel_size=4)           # 32×43
        size3 = conv_dimension(*size2, padding=1, stride=2, kernel_size=4)           # 16×22
        size4 = conv_dimension(*size3, padding=1, stride=2, kernel_size=4)           # 8×11
        
        self.size1 = size1
        self.size2 = size2
        self.size3 = size3
        self.size4 = size4
        
        # Initial projection
        self.linear = nn.Linear(latent_dim, 512 * size4[0] * size4[1])
        
        # Decoder blocks with careful size management
        self.deconv1 = spectral_norm(nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1))
        self.deconv2 = spectral_norm(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1))
        self.deconv3 = spectral_norm(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1))
        self.deconv4 = spectral_norm(nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1))
        
        # Additional refinement layers
        self.refine1 = spectral_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.refine2 = spectral_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.refine3 = spectral_norm(nn.Conv2d(64, 64, 3, padding=1))
        self.refine4 = spectral_norm(nn.Conv2d(32, 32, 3, padding=1))
        
        # Group normalization
        self.gn1 = nn.GroupNorm(32, 256)
        self.gn2 = nn.GroupNorm(16, 128)
        self.gn3 = nn.GroupNorm(8, 64)
        self.gn4 = nn.GroupNorm(8, 32)
        
        # Final output layer
        self.final_conv = spectral_norm(nn.Conv2d(32, 1, 3, padding=1))
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project to feature map
        x = self.linear(x)
        x = x.view(batch_size, 512, self.size4[0], self.size4[1])
        
        # Decoder path with explicit output sizes
        x = self.deconv1(x, output_size=(batch_size, 256, *self.size3))  # 8×11 -> 16×22
        x = F.silu(self.gn1(self.refine1(x)))
        
        x = self.deconv2(x, output_size=(batch_size, 128, *self.size2))  # 16×22 -> 32×43
        x = F.silu(self.gn2(self.refine2(x)))
        
        x = self.deconv3(x, output_size=(batch_size, 64, *self.size1))   # 32×43 -> 64×86
        x = F.silu(self.gn3(self.refine3(x)))
        
        x = self.deconv4(x, output_size=(batch_size, 32, *self.image_size))  # 64×86 -> 128×172
        x = F.silu(self.gn4(self.refine4(x)))
        
        # Final output
        x = self.final_conv(x)
        
        # Ensure exact output size (safety check)
        if x.shape[-2:] != tuple(self.image_size):
            x = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=False)
            
        return x

class CVAE_Deep(VAE):
    def __init__(self, image_size, latent_dim, beta=1.0):
        super().__init__(in_dim=1, latent_dim=latent_dim, n_layers=2, beta=beta)
        self.encoder = CVAE_Encoder_Deep(image_size, latent_dim)
        self.decoder = CVAE_Decoder_Deep(image_size, latent_dim)
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
    

def save_checkpoint(model, optimizer, epoch, loss, config, in_dims):
    if not os.path.isdir('./models'):
        os.mkdir('./models')
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if isinstance(model, CVAE):
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
        print(f'Model saved at: ./models/conv_vae/c_vae_checkpoint_{timestamp}.pt')    
    elif isinstance(model, CVAE_Deep):
        if not os.path.isdir('./models/deep_conv_vae'):
            os.mkdir('./models/deep_conv_vae')
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
        torch.save(checkpoint, f'./models/deep_conv_vae/deep_c_vae_checkpoint_{timestamp}.pt')
        print(f'Model saved at: ./models/deep_conv_vae/deep_c_vae_checkpoint_{timestamp}.pt')           
    elif isinstance(model, VAE):
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
        print(f'Model saved at: ./models/vae/vae_checkpoint_{timestamp}.pt')   