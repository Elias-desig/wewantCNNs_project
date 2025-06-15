import torch 
from torch import nn
from functions import layer_dimensions

class Encoder(nn.module):
  def __init__(self, in_dim, latent_dim, num_layers):
    super().__init__()
    self.layer_dims = layer_dimensions(in_dim, latent_dim, num_layers)
    self.linears = nn.ModuleList([nn.Linear(self.layer_dims[i]) for i in range(num_layers)])
    self.relus   = nn.ModuleList([nn.ReLU() for _ in range(num_layers)])
  def forward(self. x):
    for i, layer in enumerate(self.linears):
      x = self.relus[i](layer(x))
    return x

class Decoder(nn.module):
  def __init__(self, in_dim, latent_dim, num_layers):
    super().__init__()
    self.layer_dims = layer_dimensions(in_dim, latent_dim, num_layers)
    self.linears = nn.ModuleList([nn.Linear(self.layer_dims[i]) for i in range(num_layers)])
    self.relus   = nn.ModuleList([nn.ReLU() for _ in range(num_layers-1)])
  def forward(self. x):
    for i, layer in enumerate(self.linears):
      x = self.relus[i](layer(x))
    x = nn.sigmoid(x)
    return x
class VAE(nn.module):
  def __init__(self, in_dim, latent_dim, n_layers):
    self.mu = nn.linear(latent_dim, latent_dim)
    self.log_var = nn.linear(latent_dim, latent_dim)
    self.encoder = Encoder(in_dim, latent_dim, n_layers)
    self.decoder = Decoder(in_dim, latent_dim, n_layers)
  def sampling(self, mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.rand_like(std)
    return mu + std * eps
