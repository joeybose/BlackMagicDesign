import torch
from torch import nn, optim

class VAE(nn.Module):
    def __init__(self, input_size, latent):
        """
        input_size: size of image, 784 in case of MNIST
        latent: size of multivar Gaussian params
        """
        super(VAE, self).__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, 400)
        self.fc21 = nn.Linear(400, latent)
        self.fc22 = nn.Linear(400, latent)
        self.fc3 = nn.Linear(latent, 400)
        self.fc4 = nn.Linear(400, input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class BlackAttack(nn.Module):
    def __init__(self, input_size, latent):
        """
        A modified VAE. Latent is Gaussian (0, sigma) of dimension latent.
        Decode latent to a noise vector of `input_size`, constrain to l_\infty
        ball, then add the noise to the original input.

        Note the Gaussian \mu is not learned since input `x` acts as mean

        Args:
            input_size: size of image, 784 in case of MNIST
            latent: size of multivar Gaussian params
        """
        super(VAE, self).__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, 400)
        self.fc_sig = nn.Linear(400, latent)
        self.fc3 = nn.Linear(latent, 400)
        self.fc4 = nn.Linear(400, input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(logvar)
        delta = self.decode(z)
        x_prime = x + delta

        return x_prime, delta, logvar
