from torchvision import models
from torch import nn


class ImageEmbedding(nn.Module):
    def __init__(self, out_dim=512):
        super(ImageEmbedding, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)

        self.body = [layers for layers in self.resnet18.children()]
        self.body.pop(-1)

        self.body = nn.Sequential(*self.body)
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.body(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x
