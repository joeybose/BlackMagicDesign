import torch
import torchvision
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.models import resnet50
import torchvision.utils as vutils
from torchvision.utils import save_image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
from random import randint
from PIL import Image
import os
from models import Net
from cnn_models import *
import ipdb

class Normalize(nn.Module):
    """
    Normalize an image as part of a torch nn.Module
    """
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        num = (x - self.mean.type_as(x)[None,:,None,None])
        denom = self.std.type_as(x)[None,:,None,None]
        return num / denom

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
    save = to_img(image.cpu().data)
    save_image(save, name)
    args.experiment.log_image(name,overwrite=False)

def load_imagenet_classes():
    with open("references/adver_robust/introduction/imagenet_class_index.json") as f:
        imagenet_classes = {int(i):x[1] for i,x in json.load(f).items()}
    return imagenet_classes

def get_single_data(args):
    """
    Data loader. For now, just a test sample
    """
    if args.mnist:
        trainloader, testloader = load_mnist(normalize=False)
        tensor,target = trainloader.dataset[randint(1,\
            100)]
        tensor = tensor_to_cuda(tensor.unsqueeze(0))
        target = tensor_to_cuda(target.unsqueeze(0))
        args.classes = 10
    elif args.cifar:
        trainloader, testloader = load_cifar(args,normalize=True)
        tensor,target = trainloader.dataset[randint(1,\
            100)]
        tensor = tensor_to_cuda(tensor.unsqueeze(0))
        target = tensor_to_cuda(target.unsqueeze(0))
        args.classes = 10
    else:
        pig_img = Image.open("references/adver_robust/introduction/pig.jpg")
        preprocess = transforms.Compose([
           transforms.Resize(224),
           transforms.ToTensor(),
        ])
        tensor = tensor_to_cuda(preprocess(pig_img)[None,:,:,:])
        source_class = 341 # pig class
        target = tensor_to_cuda(torch.LongTensor([source_class]))
        args.classes = 1000

    # Get flat input size
    args.input_size = tensor[0][0].flatten().shape[0]
    return tensor, target

def get_data(args):
    """
    Data loader. For now, just a test sample
    """
    if args.mnist:
        trainloader, testloader = load_mnist(normalize=False)
        args.classes = 10
    elif args.cifar:
        trainloader, testloader = load_cifar(args,normalize=True)
        args.classes = 10
    else:
        raise NotImplementedError
    return trainloader,testloader

def load_unk_model(args):
    """
    Load an unknown model. Used for convenience to easily swap unk model
    """
    if args.mnist:
        if os.path.exists("mnist_cnn.pt"):
            model = Net().to(args.device)
            model.load_state_dict(torch.load("mnist_cnn.pt"))
            model.eval()
        else:
            model = main_mnist(args)
    if args.cifar:
        if os.path.exists("cifar_densenet121.pt"):
            model = DenseNet121().to(args.device)
            model.load_state_dict(torch.load("cifar_densenet121.pt"))
            model.eval()
        else:
            model = main_cifar(args)
    else:
        # load pre-trained ResNet50
        model = resnet50(pretrained=True).to(args.device)
    model.eval()
    return model

def main_mnist(args):
    model = Net().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_loader, test_loader = load_mnist(normalize=False)
    for epoch in range(1,11):
        train_classifier(args, model, args.device, train_loader, optimizer, epoch)
        test_classifier(args, model, args.device, test_loader)

    torch.save(model.state_dict(),"mnist_cnn.pt")
    return model

def main_cifar(args):
    architectures = [
		    (VGG, 'VGG16', 50),
		    (resnet.ResNet18, 'res18', 500),
		    (densenet.densenet_cifar, 'dense121', 500),
		    (googlenet.GoogLeNet, 'googlenet', 500),
		    (LeNet, 'lenet', 250)
	    ]
    for init_func, name, epochs in architectures:
        print("Training %s" %(name))
        model = init_func().to(args.device)
        # model = densenet_cifar().to(args.device)
        model = nn.DataParallel(model)
        optimizer = optim.Adam(model.parameters(),lr=1e-4)
        train_loader, test_loader = load_cifar(args,normalize=True)
        for epoch in range(1,500):
            train_classifier(args, model, args.device, train_loader, optimizer, epoch)
            test_classifier(args, model, args.device, test_loader)
        torch.save(model.state_dict(),"cifar_"+name+".pt")
    return model

def train_classifier(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    train_loss = 0
    correct = 0
    total = 0
    early_stop_param = 0.01
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        train_loss += loss.item()
        running_loss = loss.data[0]
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: %d [%d/%d %.0f] \tLoss: %.6f | Acc: %.3f' %(epoch,\
                            batch_idx *len(data),len(train_loader.dataset),\
                            100.*batch_idx/len(train_loader),loss.item(),\
                            100.*correct/total))
	    if early_stopping:
		if running_loss < early_stop_param:
		    print("Early Stopping !!!!!!!!!!")
		    break
	    running_loss = 0.0

def test_classifier(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            # sum up batch loss
            test_loss += loss.item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'\
            .format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

def load_cifar(args,normalize=False):
    """
    Load and normalize the training and test data for CIFAR10
    """
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
	transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,
                                                shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset,batch_size=args.batch_size,
                                                 shuffle=False, num_workers=8)
    return trainloader, testloader

def load_mnist(normalize=True):
    """
    Load and normalize the training and test data for MNIST
    """
    print('==> Preparing data..')
    if normalize:
        mnist_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])
    else:
        mnist_transforms = transforms.Compose([
                           transforms.ToTensor(),
                       ])
    trainloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=mnist_transforms),\
                               batch_size=1024, shuffle=True)
    testloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=mnist_transforms),\
                batch_size=1024, shuffle=True)
    return trainloader, testloader

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

def freeze_model(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False
    return model


