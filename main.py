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
from utils import to_cuda
import flows
from torch.nn.utils import clip_grad_norm_

def train_mnist_vae(args):
    model = to_cuda(MnistVAE())
    cond = True if (os.path.exists("saved_models/mnist_enc.pt") and os.path.exists("saved_models/mnist_dec.pt") \
            and os.path.exists("saved_models/mnist_vae.pt")) else False
    if cond:
        model.load_state_dict(torch.load("saved_models/mnist_vae.pt"))
        # model.module.load("mnist_enc.pt","mnist_dec.pt")
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
        model.module.save("saved_models/mnist_enc.pt","saved_models/mnist_dec.pt")
        torch.save(model.state_dict(),"saved_models/mnist_vae.pt")
        encoder = model.module.encoder
        decoder = model.module.decoder

    return encoder, decoder, model

def train_mnist_ae(args):
    model = to_cuda(Mnistautoencoder())
    cond = True if (os.path.exists("saved_models/mnist_aenc.pt") and os.path.exists("saved_models/mnist_adec.pt") \
            and os.path.exists("mnist_ae.pt")) else False
    if cond:
        model.load_state_dict(torch.load("saved_models/mnist_ae.pt"))
        # model.module.load("mnist_enc.pt","mnist_dec.pt")
        encoder = model.module.encoder
        decoder = model.module.decoder
    else:
        trainloader, testloader = load_mnist()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,\
                weight_decay=1e-5)
        num_epochs = 50
        for epoch in range(num_epochs):
            for data in trainloader:
                img, _ = data
                img = Variable(img).cuda()
                # ===================forward=====================
                output = model(img)
                loss = criterion(output, img)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
	    # ===================log========================
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch+1, num_epochs, loss.data[0]))
            if epoch % 10 == 0:
                pic = to_img(output.cpu().data)
                save_image(pic, './results/ae_image_{}.png'.format(epoch))
                if args.comet:
                    args.experiment.log_image('results/ae_image_{}.png'.format(epoch),overwrite=False)

            if args.comet:
                args.experiment.log_metric("MNIST AE loss",loss,step=epoch)
        model.module.save("mnist_aenc.pt","saved_models/mnist_adec.pt")
        torch.save(model.state_dict(),"saved_models/mnist_ae.pt")
        encoder = model.module.encoder
        decoder = model.module.decoder
    return encoder, decoder, model

def main(args):
    if args.mnist:
        # Normalize image for MNIST
        # normalize = Normalize(mean=(0.1307,), std=(0.3081,))
        normalize = None
        args.input_size = 784
    elif args.cifar:
        normalize=utils.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
        args.input_size = 32*32*3
    else:
        # Normalize image for ImageNet
        normalize=utils.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        args.input_size = 150528

    # Load data
    if args.single_data:
        data,target = utils.get_single_data(args)
    else:
        train_loader,test_loader = utils.get_data(args)


    # The unknown model to attack
    unk_model = utils.load_unk_model(args)

    # Try Whitebox Untargeted first
    if args.debug:
        ipdb.set_trace()

    if args.train_vae:
        encoder, decoder, vae = train_mnist_vae(args)
    else:
        encoder, decoder, vae = None, None, None

    if args.train_ae:
        encoder, decoder, ae = train_mnist_ae(args)
    else:
        encoder, decoder, ae = None, None, None

    # Add A Flow
    norm_flow = None
    if args.use_flow:
        # norm_flow = flows.NormalizingFlow(30, args.latent).to(args.device)
        norm_flow = flows.Planar
    # Test white box
    if args.white:
        # Choose Attack Function
        if args.no_pgd_optim:
            white_attack_func = attacks.L2_white_box_generator
        else:
            white_attack_func = attacks.PGD_white_box_generator

        # Choose Dataset
        if args.mnist:
            G = models.Generator(input_size=784).to(args.device)
        elif args.cifar:
            if args.vanilla_G:
                G = models.DCGAN().to(args.device)
                G = nn.DataParallel(G.generator)
            else:
                G = models.ConvGenerator(models.Bottleneck,[6,12,24,16],growth_rate=12,\
                                     flows=norm_flow,use_flow=args.use_flow,\
                                     deterministic=args.deterministic_G).to(args.device)
                G = nn.DataParallel(G)
            nc,h,w = 3,32,32

        # Test on a single data point or entire dataset
        if args.single_data:
            # pred, delta = attacks.single_white_box_generator(args, data, target, unk_model, G)
            # pred, delta = attacks.white_box_untargeted(args, data, target, unk_model)
            pred, delta = attacks.whitebox_pgd(args, data, target, unk_model,\
                    nc, h, w)
        else:
            pred, delta = white_attack_func(args, train_loader,\
                    test_loader, unk_model, G, nc, h, w)

    # Blackbox Attack model
    ipdb.set_trace()
    model = models.GaussianPolicy(args.input_size, 400,
        args.latent_size,decode=False).to(args.device)

    # Control Variate
    cv = to_cuda(models.FC(args.input_size, args.classes))

    # Launch training
    if args.single_data:
        pred, delta = attacks.single_blackbox_attack(args, 'lax', data, target, unk_model, model, cv)
        pred, delta = attacks.single_blackbox_attack(args, 'reinforce', data, target, unk_model, model, cv)

if __name__ == '__main__':
    """
    Process command-line arguments, then call main()
    """
    parser = argparse.ArgumentParser(description='BlackBox')
    # Hparams
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--latent_dim', type=int, default=20, metavar='N',
                        help='Latent dim for VAE')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--latent_size', type=int, default=50, metavar='N',
                        help='Size of latent distribution (default: 50)')
    parser.add_argument('--estimator', default='reinforce', const='reinforce',
                    nargs='?', choices=['reinforce', 'lax'],
                    help='Grad estimator for noise (default: %(default)s)')
    parser.add_argument('--reward', default='soft', const='soft',
                    nargs='?', choices=['soft', 'hard'],
                    help='Reward for grad estimator (default: %(default)s)')
    parser.add_argument('--flow_type', default='planar', const='soft',
                    nargs='?', choices=['planar', 'radial'],
                    help='Type of Normalizing Flow (default: %(default)s)')
    # Training
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--PGD_steps', type=int, default=40, metavar='N',
                        help='max gradient steps (default: 30)')
    parser.add_argument('--max_iter', type=int, default=20, metavar='N',
                        help='max gradient steps (default: 30)')
    parser.add_argument('--epsilon', type=float, default=0.5, metavar='M',
			help='Epsilon for Delta (default: 0.1)')
    parser.add_argument('--LAMBDA', type=float, default=0.1, metavar='M',
			help='Lambda for L2 lagrange penalty (default: 0.1)')
    parser.add_argument('--bb_steps', type=int, default=2000, metavar='N',
                        help='Max black box steps per sample(default: 1000)')
    parser.add_argument('--attack_epochs', type=int, default=50, metavar='N',
                        help='Max numbe of epochs to train G')
    parser.add_argument('--num_flows', type=int, default=30, metavar='N',
                        help='Number of Flows')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--input_size', type=int, default=784, metavar='S',
                        help='Input size for MNIST is default')
    parser.add_argument('--batch_size', type=int, default=256, metavar='S',
                        help='Batch size')
    parser.add_argument('--test_batch_size', type=int, default=512, metavar='S',
                        help='Test Batch size')
    parser.add_argument('--test', default=False, action='store_true',
                        help='just test model and print accuracy')
    parser.add_argument('--clip_grad', default=True, action='store_true',
                        help='Clip grad norm')
    parser.add_argument('--train_vae', default=False, action='store_true',
                        help='Train VAE')
    parser.add_argument('--train_ae', default=False, action='store_true',
                        help='Train AE')
    parser.add_argument('--mnist', default=False, action='store_true',
                        help='Use MNIST as Dataset')
    parser.add_argument('--cifar', default=False, action='store_true',
                        help='Use CIFAR-10 as Dataset')
    parser.add_argument('--white', default=False, action='store_true',
                        help='White Box test')
    parser.add_argument('--use_flow', default=False, action='store_true',
                        help='Add A NF to Generator')
    parser.add_argument('--inf_loss', default=False, action='store_true',
                        help='L-infinity penalty')
    parser.add_argument('--carlini_loss', default=False, action='store_true',
                        help='Use CW loss function')
    parser.add_argument('--no_pgd_optim', default=False, action='store_true',
                        help='Use Lagrangian objective instead of PGD')
    parser.add_argument('--vanilla_G', default=False, action='store_true',
                        help='Vanilla G White Box')
    parser.add_argument('--deterministic_G', default=False, action='store_true',
                        help='Deterministic Latent State')
    parser.add_argument('--single_data', default=False, action='store_true',
                        help='Test on a single data')
    parser.add_argument('--resample_test', default=False, action='store_true',
			    help='Load model and test resampling capability')
    parset.add_argument('--resample_iterations', type=int, default=100, metavar='N',
                        help='How many times to resample (default: 100)')
    # Bells
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--comet", action="store_true", default=False,
            help='Use comet for logging')
    parser.add_argument("--comet_username", type=str, default="joeybose",
            help='Username for comet logging')
    parser.add_argument("--comet_apikey", type=str,\
            default="Ht9lkWvTm58fRo9ccgpabq5zV",help='Api for comet logging')
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
        ipdb.set_trace()
        args.comet_apikey = data["api_key"]
        args.comet_username = data["username"]

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
