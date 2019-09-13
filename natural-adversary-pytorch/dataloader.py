import torch
from torchvision import datasets, transforms

def get_loader(dataset, batch_size):
    if dataset == 'mnist':
        # MNIST dataset.
        train_dataset = datasets.MNIST(root='./data/mnist',
                                       train=True,
                                       download=True,
                                       transform=transforms.ToTensor())

        test_dataset = datasets.MNIST(root='./data/mnist',
                                      train=False,
                                      transform=transforms.ToTensor())

        # Data loader.
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2,
                                                   drop_last=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2,
                                                  drop_last=True)
        return train_loader, test_loader