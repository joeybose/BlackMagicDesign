"""
Based on:
https://adversarial-ml-tutorial.org/
"""
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch
from torch import nn, optim
from torchvision.models import resnet50
import json

def pig():
    """
    Simple test to display image
    """
    # read the image, resize to 224 and convert to PyTorch Tensor
    pig_img = Image.open("introduction/pig.jpg")
    preprocess = transforms.Compose([
       transforms.Resize(224),
       transforms.ToTensor(),
    ])
    pig_tensor = preprocess(pig_img)[None,:,:,:]

    # plot image (note that numpy using HWC whereas Pytorch user CHW, so we need to convert)
    # plt.imshow(pig_tensor[0].numpy().transpose(1,2,0))
    # plt.show()
    return pig_tensor

def display_tensor(tensor):
    plt.imshow((tensor)[0].detach().numpy().transpose(1,2,0))
    plt.show()

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

def load_imagenet_classes():
    with open("introduction/imagenet_class_index.json") as f:
        imagenet_classes = {int(i):x[1] for i,x in json.load(f).items()}
    return imagenet_classes

def white_box_untargeted(model, image, normalize, epsilon):
    source_class = 341 # pig class
    # Create noise vector
    delta = torch.zeros_like(image, requires_grad=True)
    # Optimize noise vector to fool model
    opt = optim.SGD([delta], lr=1e-1)

    for t in range(30):
        pred = model(normalize(pig_tensor + delta))
        loss = -nn.CrossEntropyLoss()(pred, torch.LongTensor([source_class]))
        if t % 5 == 0:
            print(t, loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()
        # Clipping is equivalent to projecting back onto the l_\infty ball
        # This technique is known as projected gradient descent (PGD)
        delta.data.clamp_(-eps, eps)
    return pred, delta

def white_box_targeted(model, image, normalize, eps):
    source_class = 341 # pig class
    target_class = 404 # airliner class
    # Create noise vector
    delta = torch.zeros_like(image, requires_grad=True)
    # Optimize noise vector (only) to fool model
    opt = optim.SGD([delta], lr=5e-3)

    for t in range(100):
        pred = model(normalize(pig_tensor + delta))
        loss = (-nn.CrossEntropyLoss()(pred, torch.LongTensor([source_class])) +
            nn.CrossEntropyLoss()(pred, torch.LongTensor([target_class])))
        if t % 10 == 0:
            print(t, loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()
        # Clipping is equivalent to projecting back onto the l_\infty ball
        # This technique is known as projected gradient descent (PGD)
        delta.data.clamp_(-eps, eps)
    return pred, delta

if __name__ == "__main__":
    epsilon = 2./255
    # Normalize image for ImageNet
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # load pre-trained ResNet50, and put into evaluation mode (necessary to e.g. turn off batchnorm)
    model = resnet50(pretrained=True)
    model.eval();
    pig_tensor = pig() # get a sample
    # Predictions for the 1000 classes (logits)
    pred = model(normalize(pig_tensor))

    # Load label names
    imagenet_classes = load_imagenet_classes()

    # SETUP
    print("*"*79)
    print("Starting model")
    print("Prediction: ", imagenet_classes[pred.max(dim=1)[1].item()])
    print("CE loss: ", nn.CrossEntropyLoss()(model(normalize(pig_tensor)),torch.LongTensor([341])).item())

    # WHITE BOX UNTARGETED
    def untargeted():
        print("*"*79)
        print("White box attack untargeted")
        pred, delta = white_box_attack(model, pig_tensor, normalize, epsilon)
        print("True class probability:", nn.Softmax(dim=1)(pred)[0,341].item())
        max_class = pred.max(dim=1)[1].item()
        print("Predicted class: ", imagenet_classes[max_class])
        print("Predicted probability:", nn.Softmax(dim=1)(pred)[0,max_class].item())
        display_tensor(pig_tensor + delta)
        display_tensor(delta)
    untargeted()

    # WHITE BOX TARGETED
    def targeted():
        print("*"*79)
        print("White box attack targeted")
        pred, delta = white_box_targeted(model, pig_tensor, normalize, epsilon)
        print("True class probability:", nn.Softmax(dim=1)(pred)[0,341].item())
        max_class = pred.max(dim=1)[1].item()
        print("Predicted class: ", imagenet_classes[max_class])
        print("Predicted probability:", nn.Softmax(dim=1)(pred)[0,max_class].item())
        display_tensor(pig_tensor + delta)
        display_tensor(delta)
    targeted()







