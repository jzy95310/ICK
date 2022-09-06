import torch

class Erf(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.erf(x)

ACTIVATIONS = {
    "relu": torch.nn.ReLU(),
    "sigmoid": torch.nn.Sigmoid(),
    "tanh": torch.nn.Tanh(),
    "leaky_relu": torch.nn.LeakyReLU(),
    "elu": torch.nn.ELU(),
    "selu": torch.nn.SELU(),
    "softplus": torch.nn.Softplus(),
    "softmax": torch.nn.Softmax(),
    "erf": Erf(),
}