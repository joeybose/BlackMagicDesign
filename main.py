from comet_ml import Experiment
import matplotlib.pyplot as plt
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

def white_box_untargeted(args, image, model, normalize):
    source_class = 341 # pig class
    epsilon = 2./255
    # Create noise vector
    delta = tensor_to_cuda(torch.zeros_like(image,requires_grad=True))
    # Optimize noise vector (only) to fool model
    opt = optim.SGD([delta], lr=1e-1)
    pig_tensor = image
    target = tensor_to_cuda(torch.LongTensor([source_class]))
    for t in range(30):
        pred = model(normalize(pig_tensor + delta))
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

    if args.comet:
        clean_image = (pig_tensor)[0].detach().cpu().numpy().transpose(1,2,0)
        adv_image = (pig_tensor + delta)[0].detach().cpu().numpy().transpose(1,2,0)
        delta_image = (delta)[0].detach().cpu().numpy().transpose(1,2,0)
        plot_image_to_comet(args,clean_image,"pig.png")
        plot_image_to_comet(args,adv_image,"Adv_pig.png")
        plot_image_to_comet(args,delta_image,"delta_pig.png")
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

def train_mnist_vae(args):
    model = to_cuda(MnistVAE())
    trainloader, testloader = load_mnist()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(100):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(trainloader):
            img, _ = data
            img = img.view(img.size(0), -1)
            img = Variable(img)
            if torch.cuda.is_available():
                img = img.cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(img)
            loss = vae_loss_function(recon_batch, img, mu, logvar)
            loss.backward()
            train_loss += loss.data[0]
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(img),
                    len(trainloader.dataset), 100. * batch_idx / len(trainloader),
                    loss.data[0] / len(img)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(trainloader.dataset)))
        if args.comet:
            args.experiment.log_metric("MNIST VAE loss",train_loss,step=epoch)
        if epoch % 10 == 0:
            save = to_img(recon_batch.cpu().data)
            save_image(save, 'results/image_{}.png'.format(epoch))
            if args.comet:
                args.experiment.log_image('results/image_{}.png'.format(epoch),overwrite=False)
    ipdb.set_trace()
    model.module.save("mnist_enc.pt","mnist_dec.pt")

def train_black(args, data, unk_model, model, cv):
    """
    Main training loop for black box attack
    """
    estimator = reinforce
    opt = optim.SGD(model.parameters(), lr=5e-3)
    data = normalize(data) # pig for testing
    target = 341 # pig class for testing
    epsilon = 2./255
    # Loop data. For now, just loop same image
    for i in range(30):
        # Get gradients for delta model
        delta = model(data) # perturbation
        x_prime = data + delta # attack sample
        pred = unk_model(x_prime) # target model prediction
        cont_var = cv(x_prime) # control variate prediction
        f = loss_func(pred, target) # target loss
        f_cv = loss_func(cont_var, target) # cont var loss
        out = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
        # Gradient from gradient estimator
        policy_loss = estimator(x_prime, f, cont_var)
        opt.zero_grad()
        policy_loss.backward()
        opt.step()
        delta.data.clamp_(-epsilon, epsilon)
        if args.comet:
            args.experiment.log_metric("Blackbox CE loss",f,step=i)
            args.experiment.log_metric("Blackbox Policy loss",policy_loss,step=i)
        if i % 5 == 0:
            print(i, out[0][0], f.item())
        # TODO: constrain delta
        # Optimize control variate arguments
    if args.comet:
        clean_image = (pig_tensor)[0].detach().cpu().numpy().transpose(1,2,0)
        adv_image = (pig_tensor + delta)[0].detach().cpu().numpy().transpose(1,2,0)
        delta_image = (delta)[0].detach().cpu().numpy().transpose(1,2,0)
        ipdb.set_trace()
        plot_image_to_comet(args,clean_image,"BB_pig.png")
        plot_image_to_comet(args,adv_image,"BB_Adv_pig.png")
        plot_image_to_comet(args,delta_image,"BB_delta_pig.png")

def main(args):
    # Normalize image for ImageNet
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Load data
    data = get_data(args)

    if args.train_vae:
        train_mnist_vae(args)

    # The unknown model to attack
    ipdb.set_trace()
    unk_model = to_cuda(load_unk_model(args))

    # Try Whitebox Untargeted first
    if args.debug:
        ipdb.set_trace()

    # Test white box
    if args.white:
        pred, delta = white_box_untargeted(args,data, unk_model, normalize)

    # Attack model
    model = to_cuda(models.BlackAttack(args.input_size, args.latent_size))

    # Control Variate
    cv = to_cuda(models.FC(args.input_size, classes))

    # Launch training
    train_black(args, data, unk_model, model, cv)


if __name__ == '__main__':
    """
    Process command-line arguments, then call main()
    """
    parser = argparse.ArgumentParser(description='BlackBox')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--steps', type=int, default=30, metavar='N',
                        help='max gradient steps (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--comet", action="store_true", default=False,
            help='Use comet for logging')
    parser.add_argument("--comet_username", type=str, default="joeybose",
            help='Username for comet logging')
    parser.add_argument("--comet_apikey", type=str,\
            default="Ht9lkWvTm58fRo9ccgpabq5zV",help='Api for comet logging')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--test', default=False, action='store_true',
                        help='just test model and print accuracy')
    parser.add_argument('--train_vae', default=False, action='store_true',
                        help='Train VAE')
    parser.add_argument('--mnist', default=False, action='store_true',
                        help='Use MNIST as Dataset')
    parser.add_argument('--white', default=False, action='store_true',
                        help='White Box test')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug')
    parser.add_argument('--model_path', type=str, default="mnist_cnn.pt",
                        help='where to save/load')
    parser.add_argument('--namestr', type=str, default='BMD', \
            help='additional info in output filename to help identify experiments')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    args.device = torch.device("cuda" if use_cuda else "cpu")
    if args.comet:
        experiment = Experiment(api_key=args.comet_apikey,\
                project_name="black-magic-design",\
                workspace=args.comet_username)
        experiment.set_name(args.namestr)
        args.experiment = experiment

    main(args)
