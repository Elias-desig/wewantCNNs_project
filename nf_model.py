import torch
import torch.nn as nn
import numpy as np


class LinearFlows(nn.Module): # This is a model that leanrs linear flows, only for illustration will not be used for actually transforming our data since linear fllows are not sufficiently expressive
    def __init__(self, input_data): # Create Object
        super().__init__()
        input_tensor = torch.tensor(input_data, dtype=torch.float32)

        self.values = input_tensor
        self.weights = nn.Parameter(torch.rand(len(input_tensor))) # random parameters at the beginning
        self.bias = nn.Parameter(torch.rand(len(input_tensor)))

    def lin_flow(self): # simple equation for linear flows
        self.z = self.values * self.weights + self.bias
        return self.z # linear transformed point

    def norm_prob(self, mean=0, std_dev=1): # claulcation prob of linear transformed point equation for normal distribution
        coeff = 1 / (std_dev * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((self.z - mean) / std_dev) ** 2
        self.pz = coeff * torch.exp(exponent)
        return self.pz

    def change_of_variable(self, log_det_jacobian): # equation to compute p(x)
        self.log_px =.log(self.pz + 1e-8 torch) + log_det_jacobian
        return self.log_px

    def training_step(self, learning_rate=0.01, log_det_jacobian=0.0): # optimising parameters
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