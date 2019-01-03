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
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
from advertorch.utils import clamp
from torch import optim
from torch.autograd import Variable
import json
import argparse
from tqdm import tqdm
from utils import *
import ipdb
from advertorch.attacks import LinfPGDAttack

def whitebox_pgd(args, image, target, model, normalize=None):
    adversary = LinfPGDAttack(
	model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
	nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
	targeted=False)
    adv_image = adversary.perturb(image, target)
    print("Target is %d" %(target))
    pred = model(adv_image)
    out = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
    print("Adv Target is %d" %(out))
    clean_image = (image)[0].detach()
    adv_image = adv_image[0].detach()
    if args.comet:
        plot_image_to_comet(args,clean_image,"clean.png")
        plot_image_to_comet(args,adv_image,"Adv.png")
    return pred, clamp(clean_image - adv_image,0.,1.)

def white_box_untargeted(args, image, target, model, enc=None, dec=None, \
        vae=None, ae= None, normalize=None):
    epsilon = 0.3
    # Create noise vector
    delta = torch.zeros_like(image,requires_grad=True).to(args.device)
    # Optimize noise vector (only) to fool model
    x = image

    use_vae = True if (vae is not None) else False
    use_ae = True if (ae is not None) else False

    print("Target is %d" %(target))
    for t in range(args.PGD_steps):
        if normalize is not None:
            if use_vae:
                x = x.view(x.size(0), -1).unsqueeze(0)
                z, mu, logvar = vae(x)
                z = z.clamp(0, 1)
                x = z.view(z.size(0), 1, 28, 28)
            elif use_ae:
                x = ae(x)
            pred = model(normalize(x + delta))
        else:
            if use_vae:
                x = x.view(x.size(0), -1).unsqueeze(0)
                z, mu, logvar = vae(x)
                z = z.clamp(0, 1)
                x = z.view(z.size(0), 1, 28, 28)
            elif use_ae:
                x = ae(x)
            pred = model(x.detach() + delta)
            recon_pred = model(x.detach())
        out = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
        recon_out = recon_pred.max(1, keepdim=True)[1] # get the index of the max log-probability
        loss = nn.CrossEntropyLoss(reduction="sum")(pred, target)
        recon_image = (x)[0].detach()
        if args.comet:
            args.experiment.log_metric("Whitebox CE loss",loss,step=t)
            plot_image_to_comet(args,recon_image,"recon.png")
        if t % 5 == 0:
            print(t, out[0][0], recon_out[0][0], loss.item())

        loss.backward()
        grad_sign = delta.grad.data.sign()
        delta.data = delta.data + batch_multiply(0.01, grad_sign)
        # Clipping is equivalent to projecting back onto the l_\infty ball
        # This technique is known as projected gradient descent (PGD)
        delta.data.clamp_(-epsilon, epsilon)
        delta.data = clamp(x.data + delta.data,0.,1.) - x.data
        delta.grad.data.zero_()
        # if out != target:
            # print(t, out[0][0], loss.item())
            # break
    if args.comet:
        if not args.mnist:
            clean_image = (image)[0].detach().cpu().numpy().transpose(1,2,0)
            adv_image = (x + delta)[0].detach().cpu().numpy().transpose(1,2,0)
            delta_image = (delta)[0].detach().cpu().numpy().transpose(1,2,0)
        else:
            clean_image = (image)[0].detach()
            adv_image = (x + delta)[0].detach()
            recon_image = (x)[0].detach()
            delta_image = (delta)[0].detach().cpu()
        plot_image_to_comet(args,clean_image,"clean.png")
        plot_image_to_comet(args,adv_image,"Adv.png")
        plot_image_to_comet(args,delta_image,"delta.png")
        plot_image_to_comet(args,recon_image,"recon.png")
    return out, delta

def single_white_box_generator(args, image, target, model, G):
    epsilon = 0.5
    # Create noise vector
    x = image
    opt = optim.SGD(G.parameters(), lr=1e-2)

    print("Target is %d" %(target))
    for t in range(args.PGD_steps):
        delta, _ = G(x)
        delta = delta.view(delta.size(0), 1, 28, 28)
        delta.data.clamp_(-epsilon, epsilon)
        delta.data = clamp(x.data + delta.data,0.,1.) - x.data
        pred = model(x.detach() + delta)
        out = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
        loss = -nn.CrossEntropyLoss(reduction="sum")(pred, target)
        if args.comet:
            args.experiment.log_metric("Whitebox CE loss",loss,step=t)
        if t % 5 == 0:
            print(t, out[0][0], loss.item())
        opt.zero_grad()
        loss.backward()
        for p in G.parameters():
            p.grad.data.sign_()
        # Clipping is equivalent to projecting back onto the l_\infty ball
        # This technique is known as projected gradient descent (PGD)
        # delta.data.clamp_(-epsilon, epsilon)
        # delta.data = clamp(x.data + delta.data,0.,1.) - x.data
        opt.step()
        if out != target:
            print(t, out[0][0], loss.item())
            break
    if args.comet:
        if not args.mnist:
            clean_image = (image)[0].detach().cpu().numpy().transpose(1,2,0)
            adv_image = (x + delta)[0].detach().cpu().numpy().transpose(1,2,0)
            delta_image = (delta)[0].detach().cpu().numpy().transpose(1,2,0)
        else:
            clean_image = (image)[0].detach()
            adv_image = (x + delta)[0].detach()
            delta_image = (delta)[0].detach()
        plot_image_to_comet(args,clean_image,"clean.png")
        plot_image_to_comet(args,adv_image,"Adv.png")
        plot_image_to_comet(args,delta_image,"delta.png")
    return out, delta

def white_box_generator(args, train_loader, test_loader, model, G):
    epsilon = args.epsilon
    # Create noise vector
    opt = optim.SGD(G.parameters(), lr=1e-2)
    ''' Training Phase '''
    for epoch in range(0,args.attack_epochs):
        train_itr = tqdm(enumerate(train_loader),\
                total=len(train_loader.dataset)/args.batch_size)
        correct = 0
        for batch_idx, (data, target) in train_itr:
            x, target = data.to(args.device), target.to(args.device)
            for t in range(args.PGD_steps):
                delta, _ = G(x)
                delta = delta.view(delta.size(0), 1, 28, 28)
                # Clipping is equivalent to projecting back onto the l_\infty ball
                # This technique is known as projected gradient descent (PGD)
                delta.data.clamp_(-epsilon, epsilon)
                delta.data = clamp(x.data + delta.data,0.,1.) - x.data
                pred = model(x.detach() + delta)
                out = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
                loss = -nn.CrossEntropyLoss(reduction="sum")(pred, target)
                if args.comet:
                    args.experiment.log_metric("Whitebox CE loss",loss,step=t)
                opt.zero_grad()
                loss.backward()
                for p in G.parameters():
                    p.grad.data.sign_()
                opt.step()
            correct += out.eq(target.unsqueeze(1).data).sum()

        print('\nTrain: Epoch:{} Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'\
                .format(epoch,\
                    loss, correct, len(train_loader.dataset),
                    100. * correct / len(train_loader.dataset)))

    ''' Testing Phase '''
    test_itr = tqdm(enumerate(test_loader),\
            total=len(test_loader.dataset)/args.batch_size)
    correct_test = 0
    for batch_idx, (data, target) in test_itr:
        x, target = data.to(args.device), target.to(args.device)
        for t in range(args.PGD_steps):
            delta, _ = G(x)
            delta = delta.view(delta.size(0), 1, 28, 28)
            # Clipping is equivalent to projecting back onto the l_\infty ball
            # This technique is known as projected gradient descent (PGD)
            delta.data.clamp_(-epsilon, epsilon)
            delta.data = clamp(x.data + delta.data,0.,1.) - x.data
            pred = model(x.detach() + delta)
            out = pred.max(1, keepdim=True)[1] # get the index of the max log-probability

        correct_test += out.eq(target.unsqueeze(1).data).sum()

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'\
            .format(correct_test, len(test_loader.dataset),\
                100. * correct_test / len(test_loader.dataset)))

    if args.comet:
        if not args.mnist:
            clean_image = (image)[0].detach().cpu().numpy().transpose(1,2,0)
            adv_image = (x + delta)[0].detach().cpu().numpy().transpose(1,2,0)
            delta_image = (delta)[0].detach().cpu().numpy().transpose(1,2,0)
        else:
            clean_image = (image)[0].detach()
            adv_image = (x + delta)[0].detach()
            delta_image = (delta)[0].detach()
        plot_image_to_comet(args,clean_image,"clean.png")
        plot_image_to_comet(args,adv_image,"Adv.png")
        plot_image_to_comet(args,delta_image,"delta.png")
    return out, delta

def soft_reward(pred, targ):
    """
    BlackBox adversarial soft reward. Highest reward when `pred` for `targ`
    class is low. Use this reward to reinforce action gradients.

    Computed as: 1 - (targ pred).
    Args:
        pred: model log prediction vector, to be normalized below
        targ: true class integer, we want to decrease probability of this class
    """
    # pred = F.softmax(pred, dim=1)
    pred_prob = torch.exp(pred)
    gather = pred[:,targ] # gather target predictions
    ones = torch.ones_like(gather)
    r = ones - gather
    # r = gather.mean() # old line of code
    r = r.mean()

    return r

def hard_reward(pred, targ):
    """
    BlackBox adversarial 0/1 reward.
    1 if predict something other than target, 0 if predict target. This reward
    should make it much harder to optimize a black box attacker.
    """
    pred = F.softmax(pred, dim=1)
    out = pred.max(1, keepdim=True)[1] # get the index of the max log-prob

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
    policy_loss = (-log_prob) * f.detach()
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
    policy_loss = (-log_prob) * ac - f_cv
    cv_loss = policy_loss
    return policy_loss
