import torch
from torch import nn


class AdaIN(nn.Module):
    def __init__(self, n_features):
        super(AdaIN, self).__init__()

        self.n_features = n_features

        self.momentum = .1
        self.eps = 1e-5
        self.gamma = None
        self.bias = None

        self.register_buffer('adaptive_mean', torch.zeros(n_features))
        self.register_buffer('adaptive_var', torch.zeros(n_features))

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        adaptive_mean = self.adaptive_mean.repeat(b)
        adaptive_var = self.adaptive_var.repeat(b)

        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = nn.functional.batch_norm(
            x_reshaped, adaptive_mean, adaptive_var, self.gamma, self.bias,
            True, self.momentum, self.eps
        )

        return out.view(b, c, *x.size()[2:])
