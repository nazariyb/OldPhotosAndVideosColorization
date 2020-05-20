import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, color_dim=313, conv_dim=64):
        super(Discriminator, self).__init__()

        # img_size = 256
        inp_dim = 3 + color_dim

        self.body = nn.Sequential(
            nn.Conv2d(inp_dim, conv_dim, 4, 2, 1),
            nn.LeakyReLU(.01, inplace=True),

            nn.Conv2d(conv_dim, conv_dim * 2, 4, 2, 1),
            nn.LeakyReLU(.01, inplace=True),

            nn.Conv2d(conv_dim * 2, conv_dim * 4, 4, 2, 1),
            nn.LeakyReLU(.01, inplace=True),

            nn.Conv2d(conv_dim * 4, conv_dim * 8, 4, 2, 1),
            nn.LeakyReLU(.01, inplace=True),

            nn.Conv2d(conv_dim * 8, conv_dim * 16, 4, 2, 1),
            nn.LeakyReLU(.01, inplace=True),

            nn.Conv2d(conv_dim * 16, conv_dim * 32, 4, 2, 1),
            nn.LeakyReLU(.01, inplace=True),

            nn.Conv2d(conv_dim * 32, conv_dim * 32, 3, 1, 1, bias=False)
        )

        k_size = int(256 / 32)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(conv_dim * 8 * k_size * k_size),
            nn.Linear(conv_dim * 8 * k_size * k_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, ab_img, l_img, color_feat):
        x = torch.cat([ab_img, l_img, color_feat], dim = 1)
        out = self.body(x)

        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out
