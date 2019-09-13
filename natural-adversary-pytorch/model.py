import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
from torch.distributions import Normal
import ipdb

class Generator(nn.Module):
    def __init__(self, z_dim, image_dim, conv_dim):
        super(Generator, self).__init__()
        linear = []
        linear.append(nn.Linear(z_dim, 64 * conv_dim))
        linear.append(nn.ReLU(inplace=True))

        conv = []
        conv.append(nn.ConvTranspose2d(conv_dim*4, conv_dim*2, kernel_size=3, stride=2, padding=1, bias=False)) # 4 x 4 -> 7 x 7
        conv.append(nn.ReLU(inplace=True))
        conv.append(nn.ConvTranspose2d(conv_dim*2, conv_dim, kernel_size=3, stride=2, padding=1, bias=False))   # 7 x 7-> 13 x 13
        conv.append(nn.ReLU(inplace=True))
        conv.append(nn.ConvTranspose2d(conv_dim, image_dim, kernel_size=4, stride=2, padding=0, bias=False))    # 13 x 13 -> 28 x 28
        conv.append(nn.Sigmoid())

        self.linear = nn.Sequential(*linear)
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        out = self.linear(x)
        out = out.view(out.size(0), 64*4, 4, 4)
        out = self.conv(out)
        return out

class Inverter(nn.Module):
    def __init__(self, z_dim, image_dim, conv_dim):
        super(Inverter, self).__init__()
        conv = []
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(image_dim, conv_dim, kernel_size=3, stride=2, padding=1)    # 28 x 28 -> 15 x 15
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim*2, kernel_size=3, stride=2, padding=1)   # 15 x 15 ->  8 x 8
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=3, stride=2, padding=1) # 8 x 8 -> 4 x 4
        self.relu3 = nn.LeakyReLU(inplace=True)

        linear = []
        linear.append(nn.Linear(conv_dim*64, 512))
        linear.append(nn.LeakyReLU(inplace=True))
        linear.append(nn.Linear(512, z_dim))

        self.conv = nn.Sequential(*conv)
        self.linear = nn.Sequential(*linear)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.flatten(start_dim=1)
        out = self.linear(x).view(-1,self.z_dim,1,1)
        return out

class DCGAN(nn.Module):
    def __init__(self, batch_size, nz, nc=3, ngf=100):
        super(DCGAN, self).__init__()
        """
        Initialize a DCGAN. Perturbations from the GAN are added to the inputs to
        create adversarial attacks.
        - num_channels is the number of channels in the input
        - ngf is size of the conv layers
        """
        self.batch_size = batch_size
        self.z_dim = nz
        # self.generator = nn.Sequential(
        self.generator = nn.Sequential(
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
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            nn.Tanh()
        )

    def forward(self,inputs):
        return self.generator(inputs)

class Discriminator(nn.Module):
    def __init__(self, image_dim, conv_dim):
        super(Discriminator, self).__init__()
        conv = []
        conv.append(nn.Conv2d(image_dim, conv_dim, kernel_size=3, stride=2, padding=1))    # 28 x 28 -> 15 x 15
        conv.append(nn.LeakyReLU(inplace=True))
        conv.append(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=3, stride=2, padding=1))   # 15 x 15 -> 8 x 8
        conv.append(nn.LeakyReLU(inplace=True))
        conv.append(nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=3, stride=2, padding=1)) # 8 x 8 -> 4 x 4
        conv.append(nn.LeakyReLU(inplace=True))

        linear = []
        linear.append(nn.Linear(64*64, 1))

        self.conv = nn.Sequential(*conv)
        self.linear = nn.Sequential(*linear)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), 64*64)
        out = self.linear(out)
        return out

class CifarDiscriminator(nn.Module):
    def __init__(self):
        super(CifarDiscriminator, self).__init__()
        ndf = 32
        nc = 3
        # input is (nc) x 64 x 64
        self.conv1 = nn.Conv2d(nc, ndf, 3, 2, 1, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ndf * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 8)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.relu4(x)
        x = self.conv5(x)
        output = self.sig(x)
        return output.view(-1, 1).squeeze(1)

