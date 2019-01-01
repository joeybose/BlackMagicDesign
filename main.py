import json, os
import argparse
import ipdb
from PIL import Image
from comet_ml import Experiment
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models.vgg import VGG
import torchvision.models.densenet as densenet
import torchvision.models.alexnet as alexnet
from torchvision.utils import save_image
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import utils, models, attacks

def train_mnist_vae(args):
    model = to_cuda(MnistVAE())
    if os.path.exists("mnist_enc.pt") and os.path.exists("mnist_dec.pt"):
        model.module.load("mnist_enc.pt","mnist_dec.pt")
        encoder = model.module.encoder
        decoder = model.module.decoder
    else:
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
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'\
                        .format(
                        epoch,
                        batch_idx * len(img),
                        len(trainloader.dataset),
                        100. * batch_idx / len(trainloader),
                        loss.data[0] / len(img)))

            print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss / len(trainloader.dataset)))
            if args.comet:
                args.experiment.log_metric(\
                        "MNIST VAE loss",train_loss,step=epoch)
            if epoch % 10 == 0:
                save = to_img(recon_batch.cpu().data)
                save_image(save, 'results/image_{}.png'.format(epoch))
                if args.comet:
                    args.experiment.log_image(\
                        'results/image_{}.png'.format(epoch),overwrite=False)
        model.module.save("mnist_enc.pt","mnist_dec.pt")

    return encoder, decoder

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
        out = pred.max(1, keepdim=True)[1] # get the index of the max log-prob
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
        adv_image=(pig_tensor+delta)[0].detach().cpu().numpy().transpose(1,2,0)
        delta_image = (delta)[0].detach().cpu().numpy().transpose(1,2,0)
        plot_image_to_comet(args,clean_image,"BB_pig.png")
        plot_image_to_comet(args,adv_image,"BB_Adv_pig.png")
        plot_image_to_comet(args,delta_image,"BB_delta_pig.png")

def main(args):
    if args.mnist:
        # Normalize image for MNIST
        # normalize = Normalize(mean=(0.1307,), std=(0.3081,))
        normalize = None
    else:
        # Normalize image for ImageNet
        normalize=Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

    # Load data
    data,target = utils.get_data(args)

    # The unknown model to attack
    unk_model = utils.load_unk_model(args)

    # Try Whitebox Untargeted first
    if args.debug:
        ipdb.set_trace()

    if args.train_vae:
        encoder,decoder = train_mnist_vae(args)

    # Test white box
    if args.white:
        pred, delta = attacks.whitebox_untargeted(\
                                    args, data, target, unk_model, normalize)
        pred, delta = attacks.whitebox_pgd(\
                                    args, data, target, unk_model, normalize)

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
    parser.add_argument('--PGD_steps', type=int, default=100, metavar='N',
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
            help='additional info in output filename to describe experiments')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    # Check if settings file
    if not os.path.isfile("settings.json"):
        with open('settings.json') as f:
            data = json.load(f)
        args.comet_apikey = data["comet"]["api_key"]

        # No set_trace ;)
        if data["ipdb"] == "False":
            ipdb.set_trace = lambda: None

    args.device = torch.device("cuda" if use_cuda else "cpu")
    if args.comet:
        experiment = Experiment(api_key=args.comet_apikey,\
                project_name="black-magic-design",\
                workspace=args.comet_username)
        experiment.set_name(args.namestr)
        args.experiment = experiment

    main(args)
