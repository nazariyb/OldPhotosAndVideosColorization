import torch
from torch import nn

from AdaIN import AdaIN
from MLP import MLP


class Generator(nn.Module):
    def __init__(self, ngf=64, color_dim=313):
        super(Generator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, ngf, 4, 2, 1),
        )

        self.conv2 = nn.Sequential(
            nn.LeakyReLU(inplace = True, negative_slope = 0.2),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1),
            AdaIN(ngf * 2),
        )

        self.conv3 = nn.Sequential(
            nn.LeakyReLU(inplace = True, negative_slope = 0.2),
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1),
            AdaIN(ngf * 4),
        )

        self.conv4 = nn.Sequential(
            nn.LeakyReLU(inplace = True, negative_slope = 0.2),
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1),
            AdaIN(ngf * 8),
        )

        self.conv5 = nn.Sequential(
            nn.LeakyReLU(inplace = True, negative_slope = 0.2),
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1),
            AdaIN(ngf * 8),
        )

        self.conv6 = nn.Sequential(
            nn.LeakyReLU(inplace = True, negative_slope = 0.2),
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1),
            AdaIN(ngf * 8),
        )

        self.conv7 = nn.Sequential(
            nn.LeakyReLU(inplace = True, negative_slope = 0.2),
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1),
            AdaIN(ngf * 8),
        )

        self.conv8 = nn.Sequential(
            nn.LeakyReLU(inplace = True, negative_slope = 0.2),
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1),
        )

        self.conv_transp1 = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1),
            AdaIN(ngf * 8),
            nn.Dropout(.5),
        )

        self.conv_transp2 = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1),
            AdaIN(ngf * 8),
            nn.Dropout(.5),
        )

        self.conv_transp3 = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1),
            AdaIN(ngf * 8),
            nn.Dropout(.5),
        )

        self.conv_transp4 = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1),
            AdaIN(ngf * 8),
        )

        self.conv_transp5 = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1),
            AdaIN(ngf * 4),
        )

        self.conv_transp6 = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1),
            AdaIN(ngf * 2),
        )

        self.conv_transp7 = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1),
            AdaIN(ngf),
        )

        self.conv_transp8 = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(ngf * 2, 2, 4, 2, 1),
            nn.Tanh()
        )

        self.body = nn.Sequential(
            *self.conv1,
            *self.conv2,
            *self.conv3,
            *self.conv4,
            *self.conv5,
            *self.conv6,
            *self.conv7,
            *self.conv8,
            *self.conv_transp1,
            *self.conv_transp2,
            *self.conv_transp3,
            *self.conv_transp4,
            *self.conv_transp5,
            *self.conv_transp6,
            *self.conv_transp7,
            *self.conv_transp8,
        )

        adain_params_num = self.get_adain_params_num()
        self.mlp = MLP(color_dim, adain_params_num, adain_params_num)

    def forward(self, x, color_feat):
        adain_params = self.mlp(color_feat)
        self.set_adain_params(adain_params)

        e1 = self.conv1(x)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        e6 = self.conv6(e5)
        e7 = self.conv7(e6)
        e8 = self.conv8(e7)

        d1_ = self.conv_transp1(e8)
        d1 = torch.cat([d1_, e7], dim=1)

        d2_ = self.conv_transp2(d1)
        d2 = torch.cat([d2_, e6], dim=1)

        d3_ = self.conv_transp3(d2)
        d3 = torch.cat([d3_, e5], dim=1)

        d4_ = self.conv_transp4(d3)
        d4 = torch.cat([d4_, e4], dim=1)

        d5_ = self.conv_transp5(d4)
        d5 = torch.cat([d5_, e3], dim=1)

        d6_ = self.conv_transp6(d5)
        d6 = torch.cat([d6_, e2], dim=1)

        d7_ = self.conv_transp7(d6)
        d7 = torch.cat([d7_, e1], dim=1)

        d8 = self.conv_transp8(d7)

        return d8


    def get_adain_params_num(self):
        adain_params_num = 0
        for layer in self.body:
            if layer.__class__.__name__ == 'AdaIN':
                adain_params_num += 2 * layer.n_features
        return adain_params_num

    def set_adain_params(self, params):
        for layer in self.body:
            if layer.__class__.__name__ == 'AdaIN':
                    mean = params[:, :layer.n_features]
                    std = params[:, layer.n_features:2 * layer.n_features]
                    layer.bias = mean.contiguous().view(-1)
                    layer.gamma = std.contiguous().view(-1)
                    if params.size(1) > 2 * layer.n_features:
                        params = params[:, 2 * layer.n_features:]
