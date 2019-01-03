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
from torch.nn.utils import clip_grad_norm_

def train_mnist_vae(args):
    model = to_cuda(MnistVAE())
    cond = True if (os.path.exists("mnist_enc.pt") and os.path.exists("mnist_dec.pt") \
            and os.path.exists("mnist_vae.pt")) else False
    if cond:
        model.load_state_dict(torch.load("mnist_vae.pt"))
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
        model.module.save("mnist_enc.pt","mnist_dec.pt")
        torch.save(model.state_dict(),"mnist_vae.pt")
        encoder = model.module.encoder
        decoder = model.module.decoder

    return encoder, decoder, model

def train_mnist_ae(args):
    model = to_cuda(Mnistautoencoder())
    cond = True if (os.path.exists("mnist_aenc.pt") and os.path.exists("mnist_adec.pt") \
            and os.path.exists("mnist_ae.pt")) else False
    if cond:
        model.load_state_dict(torch.load("mnist_ae.pt"))
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
        model.module.save("mnist_aenc.pt","mnist_adec.pt")
        torch.save(model.state_dict(),"mnist_ae.pt")
        encoder = model.module.encoder
        decoder = model.module.decoder
    return encoder, decoder, model

def train_black(args, data, target, unk_model, model, cv):
    """
    Main training loop for black box attack
    """
    # Settings
    if args.estimator == "reinforce":
        estimator = attacks.reinforce_new
    elif args.estimator == "lax":
        estimator = attacks.lax_black
    if args.reward == "soft":
        reward = attacks.soft_reward

    # Freeze unknown model
    unk_model = utils.freeze_model(unk_model)

    # Original target
    target_int = int(target)
    print("Original target class is: ", target_int)
    pred = unk_model(data) # target model prediction
    # pred_prob = F.softmax(pred, dim=1)
    pred_prob = torch.exp(pred)
    pred_prob = float(pred_prob[0][target_int])
    print("Original model prediction for this class is: ", pred_prob)
    out = pred.max(1, keepdim=True)[1] # get the index of the max log-prob
    print("Current model prediction for this class is: ",\
            out[0][0].cpu().numpy())
    if out!= target:
        print("Breaking because wrong classification to begin with")
    epsilon = args.epsilon
    print("Epsilon: ", epsilon)

    print("++++BlackBox Attack start++++")
    opt = optim.SGD(model.parameters(), lr=1e-2)
    cv_opt = optim.SGD(cv.parameters(), lr=1e-2)
    # TODO: normalize?
    # normalize = utils.Normalize()
    # data = normalize(data)
    # Loop data. For now, just loop same image
    for i in range(args.bb_steps):
        # Get prediction
        # Get gradients for delta model
        delta, mu, logvar, log_prob_a = model(data) # perturbation
        # TODO: Best way to deal with delta?
        norm_pre = torch.norm(delta, float('inf'))
        # Delta constraint
        delta.data.clamp_(-epsilon, epsilon)
        delta.data = torch.clamp(data.data + delta.data,0.,1.) - data.data
        x_prime = data + delta # attack sample
        pred = unk_model(x_prime).detach() # target model prediction
        out = pred.max(1, keepdim=True)[1] # get the index of the max log-prob

        # Print some info
        def print_info():
            norm = torch.norm(delta, float('inf'))
            pred_prob = torch.exp(pred)
            pred_prob = float(pred_prob[0][target_int])
            print("[{:1.0f}] Target pred: {:1.4f} | delta norm pre-clamp:{:1.4f} | delta norm post-clamp: {:1.4f} | Curr Class:{:d}"\
                    .format(i, pred_prob, norm_pre, norm, out[0][0]))

        # Break if attack successful
        if not bool(out.squeeze(1) == target):
            print("Attack successful after {} steps".format(i))
            print_info()
            break

        # Monitor training
        cont_var = cv(x_prime) # control variate prediction
        f = reward(pred, target) # target loss
        f_cv = reward(cont_var, target) # cont var loss

        # Gradient from gradient estimator
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Old method:
        # policy_loss = estimator(log_prob=-1*log_prob_a, f=f, f_cv=f_cv).sum()
        # loss = policy_loss + KLD
        # opt.zero_grad()
        # loss.backward()

        # New method:
        opt.zero_grad()
        # Backprop KL gradient, retain since we'll backprop policy grad later
        KLD.backward(retain_graph=True)
        # Get gradient wrt to log prob
        d_log_prob_a = estimator(log_prob=-1*log_prob_a, f=f, f_cv=f_cv,
                                    param=[mu,logvar], cv=cv, cv_opt=cv_opt)
        # Backprop to all leaf nodes
        if d_log_prob_a is not None:
            log_prob_a.backward(d_log_prob_a)

        # Max inner product of infty norm constraint
        for p in model.parameters():
            p.grad.data.sign_()
        opt.step()
        if args.comet:
            args.experiment.log_metric("Blackbox CE loss",f,step=i)
            args.experiment.log_metric("Blackbox Policy loss",policy_loss,step=i)

        if i % 100 == 0:
            print_info()
        # Optimize control variate arguments
    if i == args.bb_steps - 1:
        print("Attack failed within max steps of {}".format(args.bb_steps))
        print_info()
    if args.comet:
        clean_image = (data)[0].detach().cpu()
        adv_image=(x_prime)[0].detach().cpu()
        delta_image = (delta)[0].detach().cpu()
        utils.plot_image_to_comet(args,clean_image,"BB_clean.png")
        utils.plot_image_to_comet(args,adv_image,"BB_Adv.png")
        utils.plot_image_to_comet(args,delta_image,"BB_delta.png")

def main(args):
    if args.mnist:
        # Normalize image for MNIST
        # normalize = Normalize(mean=(0.1307,), std=(0.3081,))
        normalize = None
    else:
        # Normalize image for ImageNet
        normalize=utils.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

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
    # Test white box
    if args.white:
        G = models.Generator(input_size=784).to(args.device)
        if args.single_data:
            pred, delta = attacks.single_white_box_generator(args, data, target, unk_model, G)
        else:
            pred, delta = attacks.white_box_generator(args, train_loader,\
                    test_loader, unk_model, G)

    # Attack model
    model = to_cuda(models.GaussianPolicy(args.input_size, 400, args.latent_size))
    # model = to_cuda(models.BlackAttack(args.input_size, args.latent_size))

    # Control Variate
    cv = to_cuda(models.FC(args.input_size, args.classes))

    # Launch training
    if args.single_data:
        train_black(args, data, target, unk_model, model, cv)


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
    # Training
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--PGD_steps', type=int, default=100, metavar='N',
                        help='max gradient steps (default: 30)')
    parser.add_argument('--epsilon', type=float, default=0.5, metavar='M',
			help='Epsilon for Delta (default: 0.1)')
    parser.add_argument('--bb_steps', type=int, default=1000, metavar='N',
                        help='Max black box steps per sample(default: 1000)')
    parser.add_argument('--attack_epochs', type=int, default=3, metavar='N',
                        help='Max numbe of epochs to train G')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--input_size', type=int, default=784, metavar='S',
                        help='Input size for MNIST is default')
    parser.add_argument('--batch_size', type=int, default=1024, metavar='S',
                        help='Batch size')
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
    parser.add_argument('--white', default=False, action='store_true',
                        help='White Box test')
    parser.add_argument('--single_data', default=False, action='store_true',
                        help='Test on a single data')
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
