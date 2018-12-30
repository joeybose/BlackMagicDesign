import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image

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

def load_unk_model():
    """
    Load an unknown model. Used for convenience to easily swap unk model
    """
    # load pre-trained ResNet50
    model = resnet50(pretrained=True)
    model.eval();
    return model
def load_cifar():
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

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=1024,shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset,batch_size=128,shuffle=False, num_workers=8)
    return trainloader, testloader

