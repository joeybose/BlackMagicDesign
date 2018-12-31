from PIL import Image
from torchvision import transforms
import torch
from models import *
from torch import nn, optim
from torchvision.models import resnet50
from torchvision.models.vgg import VGG
import torchvision.models.densenet as densenet
import torchvision.models.alexnet as alexnet
from torchvision.utils import save_image
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import json
import argparse
from utils import *
import ipdb

def white_box_untargeted(args, image, target, model, normalize=None):
    epsilon = 100.
    # Create noise vector
    delta = torch.zeros_like(image,requires_grad=True).to(args.device)
    # Optimize noise vector (only) to fool model
    opt = optim.SGD([delta], lr=1e-1)
    tensor = image
    print("Target is %d" %(target))
    for t in range(50):
        if normalize is not None:
            pred = model(normalize(tensor + delta))
        else:
            pred = model(tensor + delta)
        out = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
        loss = -nn.CrossEntropyLoss()(pred, target)
        if args.comet:
            args.experiment.log_metric("Whitebox CE loss",loss,step=t)
        if t % 5 == 0:
            print(t, out[0][0], loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()
        # Clipping is equivalent to projecting back onto the l_\infty ball
        # This technique is known as projected gradient descent (PGD)
        delta.data.clamp_(-epsilon, epsilon)
    ipdb.set_trace()
    if args.comet:
        if not args.mnist:
            clean_image = (tensor)[0].detach().cpu().numpy().transpose(1,2,0)
            adv_image = (tensor + delta)[0].detach().cpu().numpy().transpose(1,2,0)
            delta_image = (delta)[0].detach().cpu().numpy().transpose(1,2,0)
        else:
            clean_image = (tensor)[0].detach()
            adv_image = (tensor + delta)[0].detach()
            delta_image = (delta)[0].detach().cpu()
        plot_image_to_comet(args,clean_image,"clean.png")
        plot_image_to_comet(args,adv_image,"Adv.png")
        plot_image_to_comet(args,delta_image,"delta.png")
    return out, delta

def loss_func(pred, targ):
    """
    Want to maximize CE, so return negative since optimizer -> gradient descent
    Args:
        pred: model prediction
        targ: true class, we want to decrease probability of this class
    """
    loss = -nn.CrossEntropyLoss()(pred, torch.LongTensor([targ]))
    return loss

def linf_constraint(grad):
    """
    Constrain delta to l_infty ball
    """
    return torch.sign(grad)

def reinforce(log_prob, f, **kwargs):
    """
    Based on
    https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
    """
    policy_loss = (-log_prob) * f
    return policy_loss

def relax_black(log_prob, f, f_cv):
    """
    Returns policy loss equivalent to:
    (f(x) - c(x))*grad(log(policy)) + grad(c(x))
    The l_infty constraint should appear elsewhere
    Args:
        f: unknown function
        f_cv: control variate

    Checkout https://github.com/duvenaud/relax/blob/master/pytorch_toy.py
    """
    ac = f - f_cv
    g_cont_var = 0 # grad from control variate
    policy_loss = (-log_prob) * ac + f_cv
    return policy_loss
