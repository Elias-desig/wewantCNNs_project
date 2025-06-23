class NormaFlows(nn.Module, ):
    def __init__(self, input):
        super().__init__()  # <--- hier richtig
        self.weights = np.random.random(len(input))
        self.bias = np.random.random(len(input))
        self.values = input

    def lin_flows(self):
        self.z = np.matmul(self.values, self.weights) + self.bias
        return self.z

    def norm_prob(self, mean=0, std_dev=1):
        coeff = 1 / (std_dev * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((self.z - mean) / std_dev) ** 2
        self.px = coeff * np.exp(exponent)
        return self.px

    def change_of_variable(self):
        x = (self.values - self.bias) / self.weights
        jacobian = np.abs(1 / np.prod(
            self.weights))  # to do funktion für jacobian schreiben, da das nur für vektoren bsisher funktioner
        self.py = self.px * jacobian  # Fehler: gibt noch zwei werte aus, muss mathemathisch angepasst werden

        return self.py





