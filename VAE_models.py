import torch 
from torch import nn

class Encoder(nn.module):
  def __init__(self, in_dim, latent_dim, num_layers):
    super().__init__()
    self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
