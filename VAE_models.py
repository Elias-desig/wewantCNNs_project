import torch
from torch import nn
from functions import layer_dimensions

#What Works Now
#VAE, Encoder, Decoder: All inherit from nn.Module and have correct method signatures.
#Layer Construction: Linear layers are now built with correct input/output sizes.
#Forward Pass: The VAE can now encode, sample, and decode.

class Encoder(nn.Module):
    def __init__(self, in_dim, latent_dim, num_layers):
        super().__init__()
        self.layer_dims = layer_dimensions(in_dim, latent_dim, num_layers)
        self.linears = nn.ModuleList([
            nn.Linear(self.layer_dims[i], self.layer_dims[i+1])
            for i in range(num_layers-1)
        ])
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(num_layers-1)])
    def forward(self, x):
        for i, layer in enumerate(self.linears):
            x = self.relus[i](layer(x))
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim, out_dim, num_layers):
        super().__init__()
        self.layer_dims = layer_dimensions(latent_dim, out_dim, num_layers)
        self.linears = nn.ModuleList([
            nn.Linear(self.layer_dims[i], self.layer_dims[i+1])
            for i in range(num_layers-1)
        ])
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(num_layers-2)])
    def forward(self, x):
        for i, layer in enumerate(self.linears[:-1]):
            x = self.relus[i](layer(x))
        x = torch.sigmoid(self.linears[-1](x))
        return x

class VAE(nn.Module):
    def __init__(self, in_dim, latent_dim, n_layers):
        super().__init__()
        self.encoder = Encoder(in_dim, latent_dim, n_layers)
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.log_var = nn.Linear(latent_dim, latent_dim)
        self.decoder = Decoder(latent_dim, in_dim, n_layers)
    def reparameterization(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps
    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterization(mu, log_var)
        decoded = self.decoder(z)
        return encoded, mu, log_var, decoded
    
