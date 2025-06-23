class NormaFlows(nn.Module):
    def __init__(self, input):
        super().__init__()
        input = torch.tensor(input, dtype=torch.float32)

        self.values = input
        self.weights = nn.Parameter(torch.rand(len(input)))
        self.bias = nn.Parameter(torch.rand(len(input)))

    def lin_flows(self):
        self.z = self.values * self.weights + self.bias
        return self.z

    def norm_prob(self, mean=0, std_dev=1):
        coeff = 1 / (std_dev * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((self.z - mean) / std_dev) ** 2
        self.pz = coeff * torch.exp(exponent)
        return self.pz

    def change_of_variable(self):
        # log |det J| = Sum log |w_i| für diagonale Transformation
        log_det_jacobian = torch.sum(torch.log(torch.abs(self.weights)))
        self.log_px = torch.log(self.pz) + log_det_jacobian
        return self.log_px

    def training_step(self, learning_rate=0.01):
        # Loss = negative log-likelihood (NLL)
        self.lin_flows()
        self.norm_prob()
        log_px = self.change_of_variable()
        loss = -torch.mean(log_px)

        # Optimieren
        loss.backward()  # Gradienten berechnen
        with torch.no_grad():
            self.weights -= learning_rate * self.weights.grad
            self.bias -= learning_rate * self.bias.grad

            self.weights.grad.zero_()
            self.bias.grad.zero_()

        return loss.item()