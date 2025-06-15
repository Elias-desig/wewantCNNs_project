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
