import torch
from torch import nn, optim

class FC(nn.Module):
    def __init__(self, input_size, classes):
        super(FC, self).__init__()

        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 800)
        self.fc3 = nn.Linear(800, 400)
        self.fc4 = nn.Linear(400, classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return x

class BlackAttack(nn.Module):
    def __init__(self, input_size, latent):
        """
        A modified VAE. Latent is Gaussian (0, sigma) of dimension latent.
        Decode latent to a noise vector of `input_size`,

        Note the Gaussian \mu is not learned since input `x` acts as mean

        Args:
            input_size: size of image, 784 in case of MNIST
            latent: size of multivar Gaussian params
        """
        super(BlackAttack, self).__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, 400)
        self.fc_sig = nn.Linear(400, latent)
        self.fc3 = nn.Linear(latent, 400)
        self.fc4 = nn.Linear(400, input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_sig(h1)

    def reparameterize(self, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std)

    def decode(self, z):
        """
        Final layer should probably not have activation?
        """
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(logvar)
        delta = self.decode(z)

        return delta, logvar
