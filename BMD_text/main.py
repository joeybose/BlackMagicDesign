from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json, os
import argparse
import ipdb
from PIL import Image
from comet_ml import Experiment
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import transforms
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import utils, models, text_attacks
import flows
from torch.nn.utils import clip_grad_norm_
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
from torch.nn.modules.loss import NLLLoss,MultiLabelSoftMarginLoss,MultiLabelMarginLoss,BCELoss
from attack_models import Seq2Seq, JSDistance, Seq2SeqCAE, Baseline_LSTM
import dataHelper

def main(args):
    # Load data
    if args.single_data:
        data,target = utils.get_single_data(args)
    else:
        train_loader,test_loader = utils.get_data(args)

    # The unknown model to attack, specified in args.model
    unk_model = utils.load_unk_model(args,train_loader,test_loader)

    # Try Whitebox Untargeted first
    if args.debug:
        ipdb.set_trace()

    # TODO: do we need this alphabet?
    ntokens = len(args.alphabet)
    inv_alphabet = {v: k for k, v in args.alphabet.items()}
    args.inv_alph = inv_alphabet

    # Load model which will produce the attack
    if args.convolution_enc:
        G = Seq2SeqCAE(emsize=args.emsize,
                                 glove_weights=args.embeddings,
                                 train_emb=args.train_emb,
                                 nhidden=args.nhidden,
                                 ntokens=ntokens,
                                 nlayers=args.nlayers,
                                 noise_radius=args.noise_radius,
                                 hidden_init=args.hidden_init,
                                 dropout=args.dropout,
                                 conv_layer=args.arch_conv_filters,
                                 conv_windows=args.arch_conv_windows,
                                 conv_strides=args.arch_conv_strides)
    else:
        G = Seq2Seq(emsize=args.emsize,
                              glove_weights=args.embeddings,
                              train_emb=args.train_emb,
                              nhidden=args.nhidden,
                              ntokens=ntokens,
                              nlayers=args.nlayers,
                              noise_radius=args.noise_radius,
                              hidden_init=args.hidden_init,
                              dropout=args.dropout)

    # Efficient compute
    G = G.to(args.device)
    G = nn.DataParallel(G)

    # Maybe Add A Flow
    norm_flow = None
    if args.use_flow:
        # norm_flow = flows.NormalizingFlow(30, args.latent).to(args.device)
        norm_flow = flows.Planar

    # Test white box
    if args.white:
        # Choose Attack Function
        if args.no_pgd_optim:
            white_attack_func = text_attacks.L2_white_box_generator
        else:
            white_attack_func = text_attacks.PGD_white_box_generator

        # Test on a single data point or entire dataset
        if args.single_data:
            # pred, delta = attacks.single_white_box_generator(args, data, target, unk_model, G)
            # pred, delta = attacks.white_box_untargeted(args, data, target, unk_model)
            text_attacks.whitebox_pgd(args, data, target, unk_model)
        else:
            white_attack_func(args, train_loader,\
                    test_loader, unk_model, G)

    # Blackbox Attack model
    if args.debug:
        ipdb.set_trace()
    model = models.GaussianPolicy(args.input_size, 400,
        args.latent_size,decode=False).to(args.device)

    # Control Variate
    cv = models.FC(args.input_size, args.classes).to(args.device)

    # Launch training
    if args.single_data:
        pred, delta = text_attacks.single_blackbox_attack(args, 'lax', data, target, unk_model, model, cv)
        pred, delta = text_attacks.single_blackbox_attack(args, 'reinforce', data, target, unk_model, model, cv)

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
    parser.add_argument('--LAMBDA', type=float, default=100, metavar='M',
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
    parser.add_argument('--white', default=False, action='store_true',
                        help='White Box test')
    parser.add_argument('--use_flow', default=False, action='store_true',
                        help='Add A NF to Generator')
    parser.add_argument('--carlini_loss', default=False, action='store_true',
                        help='Use CW loss function')
    parser.add_argument('--no_pgd_optim', default=False, action='store_true',
                        help='Use Lagrangian objective instead of PGD')
    parser.add_argument('--vanilla_G', default=False, action='store_true',
                        help='Vanilla G White Box')
    parser.add_argument('--single_data', default=False, action='store_true',
                        help='Test on a single data')
    # Imported Model Params
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--nhidden', type=int, default=300,
                        help='number of hidden units per layer in LSTM')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--noise_radius', type=float, default=0.2,
                        help='stdev of noise for autoencoder (regularizer)')
    parser.add_argument('--noise_anneal', type=float, default=0.995,
                        help='anneal noise_radius exponentially by this every 100 iterations')
    parser.add_argument('--hidden_init', action='store_true',
                        help="initialize decoder hidden state with encoder's")
    parser.add_argument('--arch_i', type=str, default='300-300',
                        help='inverter architecture (MLP)')
    parser.add_argument('--arch_g', type=str, default='300-300',
                        help='generator architecture (MLP)')
    parser.add_argument('--arch_d', type=str, default='300-300',
                        help='critic/discriminator architecture (MLP)')
    parser.add_argument('--arch_conv_filters', type=str, default='500-700-1000',
                        help='encoder filter sizes for different convolutional layers')
    parser.add_argument('--arch_conv_strides', type=str, default='1-2-2',
                        help='encoder strides for different convolutional layers')
    parser.add_argument('--arch_conv_windows', type=str, default='3-3-3',
                        help='encoder window sizes for different convolutional layers')
    parser.add_argument('--z_size', type=int, default=100,
                        help='dimension of random noise z to feed into generator')
    parser.add_argument('--temp', type=float, default=1,
                        help='softmax temperature (lower --> more discrete)')
    parser.add_argument('--enc_grad_norm', type=bool, default=True,
                        help='norm code gradient from critic->encoder')
    parser.add_argument('--train_emb', type=bool, default=True,
                        help='Train Glove Embeddings')
    parser.add_argument('--gan_toenc', type=float, default=-0.01,
                        help='weight factor passing gradient from gan to encoder')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--useJS', type=bool, default=True,
                        help='use Jenson Shannon distance')
    parser.add_argument('--perturb_z', type=bool, default=True,
                        help='perturb noise space z instead of hidden c')
    parser.add_argument('--max_seq_len', type=int, default=200,
                    help='max_seq_len')
    parser.add_argument('--gamma', type=float, default=0.95,
                    help='Discount Factor')
    parser.add_argument('--model', type=str, default="lstm_emb_input",
                    help='classification model name')
    parser.add_argument('--hidden_dim', type=int, default=128,
                    help='hidden_dim')
    parser.add_argument('--burn_in', type=int, default=500,
                    help='Train VAE burnin')
    parser.add_argument('--beta', type=float, default=0.,
                    help='Entropy reg')
    parser.add_argument('--embedding_training', type=bool, default=False,
                    help='embedding_training')
    parser.add_argument('--convolution_enc', action='store_true', default=False,
                        help='use convolutions in encoder')
    parser.add_argument('--seqgan_reward', action='store_true', default=False,
                        help='use seq gan reward')
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
    parser.add_argument('--model_path', type=str, default="saved_models/lstm.pt",
                        help='where to save/load')
    parser.add_argument('--namestr', type=str, default='BMD Text', \
            help='additional info in output filename to describe experiments')
    parser.add_argument('--dataset', type=str, default="imdb",help='dataset')
    parser.add_argument('--clip', type=float, default=1, help='gradient clipping, max norm')
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
