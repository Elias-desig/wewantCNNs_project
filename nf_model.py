import torch
import torch.nn as nn
import numpy as np

class NormalizingFlow(nn.Module):
    def __init__(self, input_data):
        super().__init__()
        input_tensor = torch.tensor(input_data, dtype=torch.float32)

        self.values = input_tensor
        self.weights = nn.Parameter(torch.rand(len(input_tensor)))
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