"""
Based on
https://adversarial-ml-tutorial.org/linear_models/
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data():
mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())

train_idx = mnist_train.train_labels <= 1
mnist_train.train_data = mnist_train.train_data[train_idx]
mnist_train.train_labels = mnist_train.train_labels[train_idx]

test_idx = mnist_test.test_labels <= 1
mnist_test.test_data = mnist_test.test_data[test_idx]
mnist_test.test_labels = mnist_test.test_labels[test_idx]

train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)

