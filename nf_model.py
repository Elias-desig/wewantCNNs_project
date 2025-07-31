import torch
import torch.nn as nn
import numpy as np

# Minimal RealNVP block for 1D data
def create_mask(dim, parity):
    mask = torch.zeros(dim)
    mask[parity::2] = 1
    return mask

class RealNVPBlock(nn.Module):
    def __init__(self, dim, hidden_dim=64, parity=0):
        super().__init__()
        self.dim = dim
        self.mask = nn.Parameter(create_mask(dim, parity), requires_grad=False)
        self.s = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim), nn.Tanh()
        )
        self.t = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim)
        )
    def forward(self, x):
        x1 = x * self.mask
        s = self.s(x1) * (1 - self.mask)
        t = self.t(x1) * (1 - self.mask)
        y = x1 + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det_jacobian = s.sum(dim=1)
        return y, log_det_jacobian
    def inverse(self, y):
        y1 = y * self.mask
        s = self.s(y1) * (1 - self.mask)
        t = self.t(y1) * (1 - self.mask)
        x = y1 + (1 - self.mask) * ((y - t) * torch.exp(-s))
        return x

# Old toy NormalizingFlow for reference
class NormalizingFlow(nn.Module):
    def __init__(self, input_data):
        super().__init__()
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        self.values = input_tensor
        self.weights = nn.Parameter(torch.rand(len(input_tensor))) # random parameters at the beginning
        self.bias = nn.Parameter(torch.rand(len(input_tensor)))
    def lin_flow(self):
        self.z = self.values * self.weights + self.bias
        return self.z
    def norm_prob(self, mean=0, std_dev=1):
        coeff = 1 / (std_dev * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((self.z - mean) / std_dev) ** 2
        self.pz = coeff * torch.exp(exponent)
        return self.pz
    def change_of_variable(self, log_det_jacobian):
        self.log_px = torch.log(self.pz + 1e-8) + log_det_jacobian
        return self.log_px
    def training_step(self, learning_rate=0.01, log_det_jacobian=0.0):
        self.lin_flow()
        self.norm_prob()
        log_px = self.change_of_variable(log_det_jacobian)
        loss = -torch.mean(log_px)
        loss.backward()
        with torch.no_grad():
            self.weights -= learning_rate * self.weights.grad
            self.bias -= learning_rate * self.bias.grad
            self.weights.grad.zero_()
            self.bias.grad.zero_()
        return loss.item()


class AutoregressiveFlow(nn.Module): # Model for Autoreg flows
    def __init__(self, input_data):
        super().__init__()
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        self.values = input_tensor

        self.input_dim = len(input_tensor)

        # Masked MLP for autoreg
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * self.input_dim)  # scale (s) und shift (t) want to shift g in such a direction that it maximises x
        )