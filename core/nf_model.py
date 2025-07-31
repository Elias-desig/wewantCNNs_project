import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass


@dataclass
class MaskedFlowOutput:
    z: torch.Tensor
    s: torch.Tensor
    log_det: torch.Tensor
    log_px: torch.Tensor | None = None
    loss: torch.Tensor | None = None


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

        mask = torch.zeros(out_features,
                           in_features)  # same shape like weight matrix at the beginning fully connected no sparse
        self.register_buffer('mask', mask)

    def set_mask(self, mask):
        self.mask.copy_(mask)

    def forward(self, input):
        return nn.functional.linear(input, self.weight * self.mask, self.bias)  # if mask 0 output will be 0 -> masking


def create_autoregressive_mask(in_features, out_features):
    mask = torch.zeros(out_features, in_features)
    for i in range(out_features):
        for j in range(in_features):
            if j <= i:  # ensures that model does not look in the future
                mask[i, j] = 1
    return mask


# Neuronal Netowrk on which we apply to mask
class MLP_Masked(nn.Module):
    def __init__(self, input_dim=22016, hidden_dims=[1024, 1024]):
        super().__init__()
        self.input_dim = input_dim
        self.architecture = nn.ModuleList()
        in_dim = input_dim

        for h_dim in hidden_dims:
            linear = MaskedLinear(in_dim, h_dim)
            mask = create_autoregressive_mask(in_dim, h_dim)
            linear.set_mask(mask)
            layer = nn.Sequential(linear, nn.ReLU())
            self.architecture.append(layer)
            in_dim = h_dim

        self.output_layer = MaskedLinear(in_dim, 2 * input_dim)
        out_mask = create_autoregressive_mask(in_dim, 2 * input_dim)
        self.output_layer.set_mask(out_mask)

    def forward(self, x, compute_loss=False):
        out = x
        for layer in self.architecture:
            out = layer(out)
        output = self.output_layer(out)

        z, s = invertible_function(output, x)

        if compute_loss:
            log_p_z = norm_log_prob(z)
            log_det = log_det_jacobian(s)
            log_px = change_of_variable(log_p_z, log_det)
            loss = Loss(log_px)
            return MaskedFlowOutput(z=z, s=s, log_det=log_det, log_px=log_px, loss=loss)
        else:
            return z, s

    def inverse(self, z):
        dummy_x = torch.zeros_like(z)  # we unfortunatley need this for our architecture if we weould have coupling layers this would all be not that complicated and sketchy but we have autoreg flow somehow
        out = dummy_x
        for layer in self.architecture:
            out = layer(out)
        output = self.output_layer(out)

        dim = z.shape[1]
        s = output[:, :dim]
        t = output[:, dim:]
        s = torch.clamp(s, min=-5, max=5)

        x = z * torch.exp(s) + t
        return x

def invertible_function(output_of_network, real_data):
    dim = real_data.shape[1]
    s = output_of_network[:, :dim]
    t = output_of_network[:, dim:]

    # we have to seet boundaries for s otherwise it goes into millions or milliarde
    s = torch.clamp(s, min=-5, max=5)

    z = (real_data - t) * torch.exp(-s)
    return z, s


def norm_log_prob(z, mean=0.0, std_dev=1.0):
    pi = torch.tensor(np.pi, dtype=z.dtype, device=z.device)
    coeff = -0.5 * torch.log(2 * pi) - torch.log(torch.tensor(std_dev, dtype=z.dtype, device=z.device))
    exponent = -0.5 * ((z - mean) / std_dev) ** 2
    return coeff + exponent


def log_det_jacobian(s):
    return torch.sum(s, dim=1)


def change_of_variable(log_p_z, log_det_jacobian):
    log_p_z_sum = torch.sum(log_p_z, dim=1)
    log_p_x = log_p_z_sum + log_det_jacobian
    return log_p_x


def Loss(log_p_x):
    # Mittelwert über Batch
    return -torch.mean(log_p_x)