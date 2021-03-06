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
import utils, attack_models, graph_attacks
from utils import to_cuda
import flows
from torch.nn.utils import clip_grad_norm_
from attack_models import GCNModelVAE

def main(args):
    # Load data
    features, labels, train_mask, val_mask, test_mask, data = utils.get_data(args)
    n_nodes = data.graph.number_of_nodes()
    # n_nodes = 40

    # The unknown model to attack
    unk_model = utils.load_unk_model(args,data,features,labels)

    # Try Whitebox Untargeted first
    if args.debug:
        ipdb.set_trace()

    # Add A Flow
    norm_flow = None
    if args.use_flow:
        # norm_flow = flows.NormalizingFlow(30, args.latent).to(args.device)
        norm_flow = flows.Planar
    # Test white box
    if args.white:
        # Choose Attack Function
        if args.no_pgd_optim:
            white_attack_func = graph_attacks.L2_white_box_generator
        else:
            white_attack_func = graph_attacks.PGD_white_box_generator

        G = GCNModelVAE(args.attack_adj,n_nodes,args.in_feats,\
                        args.n_hidden,args.n_hidden,args.dropout,\
                        deterministic=args.deterministic_G)
        G = G.cuda()
        # Attack !!!
        white_attack_func(args, features, labels, train_mask, \
                          val_mask, test_mask, data, unk_model, G)

    # # Blackbox Attack model
    # ipdb.set_trace()
    # model = models.GaussianPolicy(args.input_size, 400,
        # args.latent_size,decode=False).to(args.device)

    # # Control Variate
    # cv = to_cuda(models.FC(args.input_size, args.classes))

    # # Launch training
    # if args.single_data:
        # pred, delta = attacks.single_blackbox_attack(args, 'lax', data, target, unk_model, model, cv)
        # pred, delta = attacks.single_blackbox_attack(args, 'reinforce', data, target, unk_model, model, cv)

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
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument('--flow_type', default='planar', const='soft',
                    nargs='?', choices=['planar', 'radial'],
                    help='Type of Normalizing Flow (default: %(default)s)')
    # Training
    parser.add_argument("--dataset", type=str, default="cora",
            help='The input dataset. Can be cora, citeseer, pubmed, syn(synthetic dataset) or reddit')
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
    parser.add_argument('--attack_epochs', type=int, default=10, metavar='N',
                        help='Max numbe of epochs to train G')
    parser.add_argument('--num_flows', type=int, default=30, metavar='N',
                        help='Number of Flows')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--input_size', type=int, default=784, metavar='S',
                        help='Input size for MNIST is default')
    parser.add_argument('--batch_size', type=int, default=256, metavar='S',
                        help='Batch size')
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument('--white', default=False, action='store_true',
                        help='White Box test')
    parser.add_argument('--use_flow', default=False, action='store_true',
                        help='Add A NF to Generator')
    parser.add_argument('--carlini_loss', default=False, action='store_true',
                        help='Use CW loss function')
    parser.add_argument('--attack_adj', default=False, action='store_true',
                        help='Modify the Adjacency Matrix of the Classifier')
    parser.add_argument('--influencer_attack', default=False, action='store_true',
                        help='Influencer attack like Zuegner et. al')
    parser.add_argument('--single_node_attack', default=False, action='store_true',
                        help='Attack on a single node like Zuegner et. al')
    parser.add_argument('--no_pgd_optim', default=False, action='store_true',
                        help='Use Lagrangian objective instead of PGD')
    parser.add_argument('--deterministic_G', default=False, action='store_true',
                        help='Deterministic Latent State')
    parser.add_argument('--resample_test', default=False, action='store_true',
			    help='Load model and test resampling capability')
    parser.add_argument('--resample_iterations', type=int, default=100, metavar='N',
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
