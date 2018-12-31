import torch
import torchvision
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.models import resnet50
import torchvision.utils as vutils
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import os
from models import Net

class Normalize(nn.Module):
    """
    Normalize an image as part of a torch nn.Module
    """
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

def to_cuda(model):
    cuda_stat = torch.cuda.is_available()
    if cuda_stat:
        model = torch.nn.DataParallel(model,\
                device_ids=range(torch.cuda.device_count())).cuda()
    return model

def tensor_to_cuda(x):
    cuda_stat = torch.cuda.is_available()
    if cuda_stat:
        x = x.cuda()
    return x

def display_tensor(tensor):
    plt.imshow((tensor)[0].detach().numpy().transpose(1,2,0))
    plt.show()

def plot_image_to_comet(args,image,name):
    fig = plt.figure()
    im1 = fig.figimage(image)
    plt.savefig(name)
    args.experiment.log_image(name,overwrite=True)
    plt.clf()

def load_imagenet_classes():
    with open("references/adver_robust/introduction/imagenet_class_index.json") as f:
        imagenet_classes = {int(i):x[1] for i,x in json.load(f).items()}
    return imagenet_classes

def get_data():
    """
    Data loader. For now, just a test sample
    """
    pig_img = Image.open("references/adver_robust/introduction/pig.jpg")
    preprocess = transforms.Compose([
       transforms.Resize(224),
       transforms.ToTensor(),
    ])
    pig_tensor = tensor_to_cuda(preprocess(pig_img)[None,:,:,:])
    return pig_tensor

def load_unk_model(args):
    """
    Load an unknown model. Used for convenience to easily swap unk model
    """
    if args.mnist:
        if os.path.exists("mnist_cnn.pt"):
            model = Net().to(args.device)
            model.load_state_dict(torch.load("mnist_cnn.pt"))
        else:
            model = main_mnist(args)
    else:
        # load pre-trained ResNet50
        model = resnet50(pretrained=True)
    model.eval()
    return model

def main_mnist(args):
    model = Net().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_loader, test_loader = load_mnist()
    for epoch in range(1,11):
        train_mnist(args, model, args.device, train_loader, optimizer, epoch)
        test_mnist(args, model, args.device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(),"mnist_cnn.pt")
    return model

def train_mnist(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test_mnist(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def load_cifar():
    """
    Load and normalize the training and test data for CIFAR10
    """
    print('==> Preparing data..')
    transform_train = transforms.Compose([
	# transforms.RandomCrop(32, padding=4),
	# transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
	transforms.ToTensor(),
	# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=1024,shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset,batch_size=128,shuffle=False, num_workers=8)
    return trainloader, testloader

def load_mnist():
    """
    Load and normalize the training and test data for MNIST
    """
    print('==> Preparing data..')
    trainloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1024, shuffle=True)
    testloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1024, shuffle=True)
    return trainloader, testloader

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x
