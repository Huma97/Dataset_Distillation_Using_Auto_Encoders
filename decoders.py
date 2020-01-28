import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ReparamModule


class Generator(nn.Module):
  def __init__(self, nc=1, nz=100, ngf=64):
    super(Generator, self).__init__()
    self.main = nn.Sequential(
      # input is Z, going into a convolution
      nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
      nn.BatchNorm2d(ngf * 8),
      nn.ReLU(True),
      # state size. (ngf*8) x 4 x 4
      nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 4),
      nn.ReLU(True),
      # state size. (ngf*4) x 8 x 8
      nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 2),
      nn.ReLU(True),
      # state size. (ngf*2) x 16 x 16
      nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf),
      nn.ReLU(True),
      nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=2, bias=False),
      nn.Tanh()
    )

  def forward(self, x):
    x = self.main(x)

    return x

class ConvAutoencoder(nn.Module):
  def __init__(self):
    super(ConvAutoencoder, self).__init__()
    ## encoder ##
    self.encoder = nn.Sequential(
      nn.Conv2d(1, 16, 3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(16, 4, 3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2, 2)
    )

    ## decoder##
    # self.decoder = nn.Sequential(
    #   nn.ConvTranspose2d(4, 16, 2, stride=2),
    #   nn.ReLU(),
    #   nn.ConvTranspose2d(16, 1, 2, stride=2),
    #   nn.Sigmoid()
    # )

    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(4, 64, 2, stride=2, padding=0),
      nn.ReLU(),
      nn.BatchNorm2d(64),
      nn.ConvTranspose2d(64, 128, 3, stride=1, padding=0),
      nn.ReLU(),
      nn.BatchNorm2d(128),
      nn.ConvTranspose2d(128, 32, 2, stride=2, padding=2),
      nn.ReLU(),
      nn.BatchNorm2d(32),
      nn.Conv2d(32, 1, 1, stride=1, padding=0),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)

    return x

class LinearAutoencoder(nn.Module):
  def __init__(self):
    super(LinearAutoencoder, self).__init__()
    ## encoder ##
    self.encoder = nn.Sequential(
      nn.Linear(28 * 28, 512),
      nn.ReLU(),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Linear(256, 196)
    )

    ## decoder ##
    self.decoder = nn.Sequential(
      nn.Linear(196, 256),
      nn.ReLU(),
      nn.Linear(256, 512),
      nn.ReLU(),
      nn.Linear(512, 28 * 28),
      nn.Sigmoid()
    )

  def forward(self, x):
    # add layer, with relu activation function
    x = self.encoder(x)
    # output layer (sigmoid for scaling from 0 to 1)
    x = self.decoder(x)
    return x