import torch
from torch import nn

class StaticCoefficientModel(nn.Module):
    def __init__(self, num_mods):
        super(StaticCoefficientModel, self).__init__()
        self.num_mods = num_mods
        self.coefs = nn.Linear(num_mods, 1, bias=False)

    def forward(self, scores):
        return self.coefs(scores)
