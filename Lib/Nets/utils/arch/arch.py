import torch
import torch.nn as nn


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""
Author: Alessandro Cattoi
Description: This file defines all the architecture employed in this work
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Discriminator(nn.Module):
    """
    Define the discriminator architecture
    """
    def __init__(self, input_c, use_bias):
        super(Discriminator, self).__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))

        self.model = nn.Sequential(
            # fixed first
            nn.Conv2d(input_c, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # n_layers_D = 3 means 2 layer for discriminator + 1 fixed
            # 1
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=use_bias),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 2
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=use_bias),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 3 fixed
            nn.Conv2d(256, 512, 4, padding=1, bias=use_bias),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # fixed last
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Generator(nn.Module):
    """
    Define the generator architecture
    """
    def __init__(self, input_c, output_c, use_dropout, use_bias):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # First fixed layer
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_c, 64, 7, bias=use_bias),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # Downsampling fixed at 2
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=use_bias),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=use_bias),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            #nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=use_bias),
            #nn.InstanceNorm2d(512),
            #nn.ReLU(inplace=True),

            # Residual blocks
            ResidualBlock(256, use_dropout, use_bias),
            ResidualBlock(256, use_dropout, use_bias),
            ResidualBlock(256, use_dropout, use_bias),

            ResidualBlock(256, use_dropout, use_bias),
            ResidualBlock(256, use_dropout, use_bias),
            ResidualBlock(256, use_dropout, use_bias),

            ResidualBlock(256, use_dropout, use_bias),
            ResidualBlock(256, use_dropout, use_bias),
            ResidualBlock(256, use_dropout, use_bias),

            # Upsampling fixed at 2
            #nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1, bias=use_bias),
            #nn.InstanceNorm2d(256),
            #nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # Last layer fixed
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_c, 7),
            #nn.Tanh(),
            nn.Conv2d(output_c, output_c, 1),
            #nn.Conv2d(output_c, output_c, 1),
        )


    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    """
    Residual block definition
    """
    def __init__(self, in_channels, use_dropout, use_bias):
        super(ResidualBlock, self).__init__()

        if use_dropout:
            self.res = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(in_channels, in_channels, 3, bias=use_bias),
                                     nn.InstanceNorm2d(in_channels),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(0.5),
                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(in_channels, in_channels, 3, bias=use_bias),
                                     nn.InstanceNorm2d(in_channels))
        else:
            self.res = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(in_channels, in_channels, 3, bias=use_bias),
                                     nn.InstanceNorm2d(in_channels),
                                     nn.ReLU(inplace=True),
                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(in_channels, in_channels, 3, bias=use_bias),
                                     nn.InstanceNorm2d(in_channels))

    def forward(self, x):
        return x + self.res(x)


class PrintLayer(nn.Module):
    """
    This layer can be inserted to debug networks
    """
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x


class newSN(nn.Module):
    """
    This class implemetes the architecture to perform the classification task
    """
    def __init__(self, output_c, use_bias, use_dropout):
        super(newSN, self).__init__()
        self.model = nn.Sequential(
            #ResidualBlock(256, use_dropout, use_bias),
            #ResidualBlock(256, use_dropout, use_bias),

            # Upsampling fixed at 2
            # nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1, bias=use_bias),
            # nn.InstanceNorm2d(256),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # Last layer fixed
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_c, 7)
        )

    def forward(self, x):
        return self.model(x)
