import torch.nn as nn
import numpy as np

class LinBlock(nn.Module):
    def __init__(self, in_feat, out_feat, normalize=True):
        super(LinBlock, self).__init__()
        self.linear = nn.Linear(in_feat, out_feat)
        self.bn = nn.BatchNorm1d(out_feat, 0.8)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.normalize = normalize

    def forward(self, x):
        x = self.linear(x)
        if self.normalize:
            x = self.bn(x)
        x = self.activation(x)
        return x

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity