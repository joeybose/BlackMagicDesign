import os
import argparse
from solver import Solver
from dataloader import get_loader
from torch.backends import cudnn
import sys
sys.path.append("..")
import utils, models
from cnn_models import *
import ipdb

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directory if it doesn't exist.
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.adversary_dir):
        os.makedirs(config.adversary_dir)

    # Data loader.
    train_loader, test_loader = utils.get_data(config,normalize=True,root='../data')

    # Solver for training and generating natural adversary.
    solver = Solver(train_loader, test_loader, config)

    if config.mode == 'train':
        solver.train()

    elif config.mode == 'generate':
        solver.generate_adversary(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--z_dim', type=int, default=100, help='dimension of z')
    parser.add_argument('--image_dim', type=int, default=3, help='dimension of images')
    parser.add_argument('--conv_dim', type=int, default=64)
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_i', type=float, default=0.1, help='weight for divergence of z')

    # Training configuration.
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'classify', 'generate'])
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'lsun'])
    parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=1000, help='number of training steps')
    parser.add_argument('--dis_iters', type=int, default=5, help='number of discriminator steps')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for generator')
    parser.add_argument('--i_lr', type=float, default=0.0001, help='learning rate for inverter')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for discriminator')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--mnist', default=False, action='store_true',
                        help='Use MNIST as Dataset')
    parser.add_argument('--cifar', default=False, action='store_true',
                        help='Use CIFAR-10 as Dataset')
    parser.add_argument('--test_batch_size', type=int, default=256, metavar='S',
                        help='Test Batch size')
    parser.add_argument('--namestr', type=str, default='NAG', \
            help='additional info in output filename to describe experiments')

    # Testing configuration.
    parser.add_argument('--classifier', type=str, default='lenet')
    parser.add_argument('--resume_iters', type=int, default=None, help='load model from this step')
    parser.add_argument('--cla_iters', type=int, default=500, choices='load classifier from this step')
    parser.add_argument('--search', type=str, default='iterative', choices=['iterative', 'recursive'])
    parser.add_argument('--n_samples', type=int, default=5000, help='number of adversary samples to generate')
    parser.add_argument('--step', type=float, default=0.01, help='delta r for search step size')

    # Directories.
    parser.add_argument('--cla_dir', type=str, default='./classifier')
    parser.add_argument('--model_dir', type=str, default='./models')
    parser.add_argument('--sample_dir', type=str, default='./samples')
    parser.add_argument('--adversary_dir', type=str, default='./adversaries')

    # Step size.
    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--model_step', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=10)

    config = parser.parse_args()
    print(config)
    main(config)
