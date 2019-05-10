from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json, os
import datetime
import argparse
from types import MethodType
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
        train_loader,test_loader = utils.get_data(args, args.prepared_data)

    # The unknown model to attack, specified in args.model
    unk_model = utils.load_unk_model(args,train_loader,test_loader)

    # Try Whitebox Untargeted first
    if args.debug:
        ipdb.set_trace()

    # TODO: do we need this alphabet?
    ntokens = args.vocab_size
    # ntokens = len(args.alphabet)
    # inv_alphabet = {v: k for k, v in args.alphabet.items()}
    # args.inv_alph = inv_alphabet

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

    # # Blackbox Attack model
    # model = models.GaussianPolicy(args.input_size, 400,
        # args.latent_size,decode=False).to(args.device)

    # # Control Variate
    # cv = models.FC(args.input_size, args.classes).to(args.device)

    # # Launch training
    # if args.single_data:
        # pred, delta = text_attacks.single_blackbox_attack(args, 'lax', data, target, unk_model, model, cv)
        # pred, delta = text_attacks.single_blackbox_attack(args, 'reinforce', data, target, unk_model, model, cv)

if __name__ == '__main__':
    """
    Process command-line arguments, then call main()
    """
    parser = argparse.ArgumentParser(description='BlackBox')
    # Hparams
    padd = parser.add_argument
    padd('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    padd('--latent_dim', type=int, default=20, metavar='N',
                        help='Latent dim for VAE')
    padd('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    padd('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    padd('--latent_size', type=int, default=50, metavar='N',
                        help='Size of latent distribution (default: 50)')
    padd('--estimator', default='reinforce', const='reinforce',
                    nargs='?', choices=['reinforce', 'lax'],
                    help='Grad estimator for noise (default: %(default)s)')
    padd('--reward', default='soft', const='soft',
                    nargs='?', choices=['soft', 'hard'],
                    help='Reward for grad estimator (default: %(default)s)')
    padd('--flow_type', default='planar', const='soft',
                    nargs='?', choices=['planar', 'radial'],
                    help='Type of Normalizing Flow (default: %(default)s)')
    # Training
    padd('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    padd('--PGD_steps', type=int, default=40, metavar='N',
                        help='max gradient steps (default: 30)')
    padd('--max_iter', type=int, default=20, metavar='N',
                        help='max gradient steps (default: 30)')
    padd('--max_batches', type=int, default=None, metavar='N',
                        help='max number of batches per epoch, used for debugging (default: None)')
    padd('--epsilon', type=float, default=0.5, metavar='M',
			help='Epsilon for Delta (default: 0.1)')
    padd('--LAMBDA', type=float, default=100, metavar='M',
			help='Lambda for L2 lagrange penalty (default: 0.1)')
    padd('--nn_temp', type=float, default=1.0, metavar='M',
                   help='Starting diff. nearest neighbour temp (default: 1.0)')
    padd('--bb_steps', type=int, default=2000, metavar='N',
                        help='Max black box steps per sample(default: 1000)')
    padd('--attack_epochs', type=int, default=10, metavar='N',
                        help='Max numbe of epochs to train G')
    padd('--num_flows', type=int, default=30, metavar='N',
                        help='Number of Flows')
    padd('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    padd('--batch_size', type=int, default=256, metavar='S',
                        help='Batch size')
    padd('--embedding_dim', type=int, default=300,
                    help='embedding_dim')
    padd('--embedding_type', type=str, default="non-static",
                    help='embedding_type')
    padd('--test_batch_size', type=int, default=128, metavar='N',
                        help='Test Batch size. 256 requires 12GB GPU memory')
    padd('--test', default=False, action='store_true',
                        help='just test model and print accuracy')
    padd('--clip_grad', default=True, action='store_true',
                        help='Clip grad norm')
    padd('--train_vae', default=False, action='store_true',
                        help='Train VAE')
    padd('--train_ae', default=False, action='store_true',
                        help='Train AE')
    padd('--white', default=False, action='store_true',
                        help='White Box test')
    padd('--use_flow', default=False, action='store_true',
                        help='Add A NF to Generator')
    padd('--carlini_loss', default=False, action='store_true',
                        help='Use CW loss function')
    padd('--no_pgd_optim', default=False, action='store_true',
                        help='Use Lagrangian objective instead of PGD')
    padd('--vanilla_G', default=False, action='store_true',
                        help='Vanilla G White Box')
    padd('--single_data', default=False, action='store_true',
                        help='Test on a single data')
    padd('--prepared_data',default='dataloader/prepared_data.pickle',
                        help='Test on a single data')

    # Imported Model Params
    padd('--emsize', type=int, default=300,
                        help='size of word embeddings')
    padd('--nhidden', type=int, default=300,
                        help='number of hidden units per layer in LSTM')
    padd('--nlayers', type=int, default=2,
                        help='number of layers')
    padd('--noise_radius', type=float, default=0.2,
                        help='stdev of noise for autoencoder (regularizer)')
    padd('--noise_anneal', type=float, default=0.995,
                        help='anneal noise_radius exponentially by this every 100 iterations')
    padd('--hidden_init', action='store_true',
                        help="initialize decoder hidden state with encoder's")
    padd('--arch_i', type=str, default='300-300',
                        help='inverter architecture (MLP)')
    padd('--arch_g', type=str, default='300-300',
                        help='generator architecture (MLP)')
    padd('--arch_d', type=str, default='300-300',
                        help='critic/discriminator architecture (MLP)')
    padd('--arch_conv_filters', type=str, default='500-700-1000',
                        help='encoder filter sizes for different convolutional layers')
    padd('--arch_conv_strides', type=str, default='1-2-2',
                        help='encoder strides for different convolutional layers')
    padd('--arch_conv_windows', type=str, default='3-3-3',
                        help='encoder window sizes for different convolutional layers')
    padd('--z_size', type=int, default=100,
                        help='dimension of random noise z to feed into generator')
    padd('--temp', type=float, default=1,
                        help='softmax temperature (lower --> more discrete)')
    padd('--enc_grad_norm', type=bool, default=True,
                        help='norm code gradient from critic->encoder')
    padd('--train_emb', type=bool, default=True,
                        help='Train Glove Embeddings')
    padd('--gan_toenc', type=float, default=-0.01,
                        help='weight factor passing gradient from gan to encoder')
    padd('--dropout', type=float, default=0.0,
                        help='dropout applied to layers (0 = no dropout)')
    padd('--useJS', type=bool, default=True,
                        help='use Jenson Shannon distance')
    padd('--perturb_z', type=bool, default=True,
                        help='perturb noise space z instead of hidden c')
    padd('--max_seq_len', type=int, default=200,
                    help='max_seq_len')
    padd('--gamma', type=float, default=0.95,
                    help='Discount Factor')
    padd('--model', type=str, default="lstm_arch",
                    help='classification model name')
    padd('--distance_func', type=str, default="cosine",
                    help='NN distance function')
    padd('--hidden_dim', type=int, default=128,
                    help='hidden_dim')
    padd('--burn_in', type=int, default=500,
                    help='Train VAE burnin')
    padd('--beta', type=float, default=0.,
                    help='Entropy reg')
    padd('--embedding_training', type=bool, default=False,
                    help='embedding_training')
    padd('--convolution_enc', action='store_true', default=False,
                        help='use convolutions in encoder')
    padd('--seqgan_reward', action='store_true', default=False,
                        help='use seq gan reward')
    padd('--train_classifier', action='store_true', default=False,
                        help='Train Classifier from scratch')
    padd('--diff_nn', action='store_true', default=False,
                        help='Backprop through Nearest Neighbors')
    # Bells
    padd('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    padd('--no_parallel', action='store_true', default=False,
                        help="Don't use multiple GPUs")
    padd('--save_adv_samples', action='store_true', default=False,
                            help='Write adversarial samples to disk')
    padd('--nearest_neigh_all', action='store_true', default=False,
                          help='Evaluate near. neig. for whole evaluation set')
    padd("--comet", action="store_true", default=False,
            help='Use comet for logging')
    padd("--comet_username", type=str, default="joeybose",
            help='Username for comet logging')
    padd("--comet_apikey", type=str,\
            default="Ht9lkWvTm58fRo9ccgpabq5zV",help='Api for comet logging')
    padd('--debug', default=False, action='store_true',
                        help='Debug')
    padd('--debug_neighbour', default=False, action='store_true',
                        help='Debug nearest neighbour training')
    padd('--model_path', type=str, default="saved_models/lstm_torchtext2.pt",\
                        help='where to save/load')
    padd('--no_load_embedding', action='store_false', default=True,
                    help='load Glove embeddings')
    padd('--namestr', type=str, default='BMD Text', \
            help='additional info in output filename to describe experiments')
    padd('--dataset', type=str, default="imdb",help='dataset')
    padd('--clip', type=float, default=1, help='gradient clipping, max norm')
    padd('--use_glove', type=str, default="true",
                    help='gpu number')
    args = parser.parse_args()
    args.classes = 2
    args.sample_file = "temp/adv_samples.txt"
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    # Check if settings file
    if os.path.isfile("settings.json"):
        with open('settings.json') as f:
            data = json.load(f)
        args.comet_apikey = data["apikey"]
        args.comet_username = data["username"]

    # Prep file to save samples
    if args.save_adv_samples:
        now = datetime.datetime.now()
        if os.path.exists(args.sample_file):
            os.remove(args.sample_file)
        with open(args.sample_file, 'w') as f:
            f.write("Adversarial samples starting:\n{}\n".format(now))

    # No set_trace ;)
    if args.debug is False:
        ipdb.set_trace = lambda: None

    args.device = torch.device("cuda" if use_cuda else "cpu")
    if args.comet:
        experiment = Experiment(api_key=args.comet_apikey,\
                project_name="black-magic-design",\
                workspace=args.comet_username)
        experiment.set_name(args.namestr)
        def log_text(self, msg):
            # Change line breaks for html breaks
            msg = msg.replace('\n','<br>')
            self.log_html("<p>{}</p>".format(msg))
        experiment.log_text = MethodType(log_text, experiment)
        args.experiment = experiment

    main(args)
