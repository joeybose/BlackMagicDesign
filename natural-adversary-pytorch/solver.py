import torch
from torchvision.utils import save_image
import torchvision.models as models
import os
import time
import ipdb
import sys
sys.path.append("..")
import utils, models
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np

from model import Generator, DCGAN
from model import CifarDiscriminator, Discriminator
from model import Inverter
from classifier import LeNet

class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        # Data loader.
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.config = config

        # Model configurations.
        self.z_dim = config.z_dim
        self.image_dim = config.image_dim
        self.conv_dim = config.conv_dim
        self.lambda_gp = config.lambda_gp
        self.lambda_i = config.lambda_i

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.dis_iters = config.dis_iters # discriminator iteration
        self.resume_iters = config.resume_iters
        self.g_lr = config.g_lr
        self.i_lr = config.i_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Testing configurations.
        self.classifier = config.classifier
        self.cla_iters = config.cla_iters
        self.search = config.search
        self.n_samples = config.n_samples
        self.step = config.step

        # Directories.
        self.model_dir = config.model_dir
        self.sample_dir = config.sample_dir
        self.cla_dir = config.cla_dir
        self.adversary_dir = config.adversary_dir

        # Step size.
        self.log_step = config.log_step
        self.model_step = config.model_step
        self.sample_step = config.sample_step

        # Build the model.
        self.build_model()

    def build_model(self):
        self.G = DCGAN(self.config.batch_size,nz=100)
        self.I = Inverter(self.z_dim, self.image_dim, self.conv_dim)
        self.D = CifarDiscriminator()
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.i_optimizer = torch.optim.Adam(self.I.parameters(), self.i_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        self.G.to(self.device)
        self.I.to(self.device)
        self.D.to(self.device)

    def load_model(self, resume_iters):
        print('Loading the trained model from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_dir, '{}_G.ckpt'.format(resume_iters))
        I_path = os.path.join(self.model_dir, '{}_I.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_dir, '{}_D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.I.load_state_dict(torch.load(I_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.i_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx)-1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        gradient_penalty = torch.mean((dydx_l2norm - 1.)**2)
        return gradient_penalty

    def train(self):
        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.load_model(self.resume_iters)

        # Fixed inputs for sampling.
        fixed_noise = torch.randn(self.batch_size, self.z_dim,1,1).to(self.device)
        fixed_images = next(iter(self.train_loader))[0].to(self.device)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            for j, (images, _) in enumerate(self.train_loader):

                # =================== Process input data. =================== #
                noise = torch.randn(len(images), self.z_dim,1,1).to(self.device)
                images = images.to(self.device)

                # ================ Train the discriminator. ================= #
                # Compute loss (Wasserstein-1 distance).
                d_real_loss = self.D(images)
                fake_images = self.G(noise)
                d_fake_loss = self.D(fake_images)

                # Compute loss for gradient penalty.
                alpha = torch.rand(images.size(0), 1, 1, 1).to(self.device)
                x_hat = alpha * fake_images + (1-alpha) * images
                d_gp_loss = self.gradient_penalty(self.D(x_hat), x_hat)

                d_loss = torch.mean(d_fake_loss) - torch.mean(d_real_loss) + self.lambda_gp * d_gp_loss

                # Backprop and optimize.
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # =================== Train the inverter. =================== #
                # Compute loss.
                reconst_images = self.G(self.I(images))
                reconst_noise = self.I(self.G(noise))

                i_loss = torch.mean((images-reconst_images)**2) \
                         + self.lambda_i * torch.mean((noise-reconst_noise)**2)

                # Backprop and optimize.
                self.reset_grad()
                i_loss.backward()
                self.i_optimizer.step()

                # ================== Train the generator. =================== #
                # Compute loss.
                g_loss = -torch.mean(self.D(self.G(noise)))

                # Backprop and optimize.
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

            # Print out training information.
            if (i+1) % self.log_step == 0:
                elapsed_time = time.time() - start_time
                print('Elapsed time [{:.4f}], Iteration [{}/{}], '
                      'D Loss: {:.4f}, I Loss: {:.4f}, G Loss: {:.4f}'.format(
                    elapsed_time, i+1, self.num_iters,
                    d_loss.item(), i_loss.item(), g_loss.item()
                ))

            # Save sampled images.
            if (i+1) % self.sample_step == 0:
                # Generate from noise.
                output = self.G(fixed_noise)
                sample_path = os.path.join(self.sample_dir, '{}_samples.jpg'.format(i+1))
                save_image(output.data.cpu(), sample_path)
                print('Saved generated images into {}...'.format(sample_path))

                # Reconstruct images.
                reconst_images = self.G(self.I(fixed_images))
                comparison = torch.zeros((fixed_images.size(0) * 2,
                                          fixed_images.size(1),
                                          fixed_images.size(2),
                                          fixed_images.size(3)),
                                         dtype=torch.float).to(self.device)
                for k in range(fixed_images.size(0)):
                    comparison[2*k] = fixed_images[k]
                    comparison[2*k+1] = reconst_images[k]

                sample_path = os.path.join(self.sample_dir, '{}_reconstructions.jpg'.format(i+1))
                save_image(comparison.data.cpu(), sample_path)

            # Save model checkpoints.
            if (i+1) % self.model_step == 0:
                G_path = os.path.join(self.model_dir, '{}_G.ckpt'.format(i+1))
                I_path = os.path.join(self.model_dir, '{}_I.ckpt'.format(i+1))
                D_path = os.path.join(self.model_dir, '{}_D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.I.state_dict(), I_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_dir))


    def load_unk_model(self, args,retrain=False):
        """
        Load an unknown model. Used for convenience to easily swap unk model
        """
        if args.mnist:
            if os.path.exists("../saved_models/mnist_cnn.pt"):
                model = Net().to(args.device)
                model.load_state_dict(torch.load("../saved_models/mnist_cnn.pt"))
                model.eval()
            else:
                model = main_mnist(args)
        if args.cifar:
            if os.path.exists("../saved_models/cifar_VGG16.pt") and retrain == False:
                model = VGG().to(args.device)
                model = nn.DataParallel(model)
                model.load_state_dict(torch.load("../saved_models/cifar_VGG16.pt"))
            else:
                model = utils.main_cifar(args,normalize=False)
        else:
            # load pre-trained ResNet50
            model = resnet50(pretrained=True).to(args.device)
        return model

    def generate_adversary(self,args):
        # Load the trained models.
        self.load_model(self.resume_iters)

        # Choose search method.
        print('Search method:', self.search)

        # Choose the classifier to generate adversary examples against.
        print('Classifier:', self.classifier)
        C = load_unk_model(args,retrain=False)
        # if self.classifier == 'lenet':
            # C = LeNet().to(self.device)
            # cla_path = os.path.join(self.cla_dir, self.classifier, '{}_lenet.ckpt'.format(self.cla_iters))
            # C.load_state_dict(torch.load(cla_path, map_location=lambda storage, loc: storage))

        # Generate adversary examples.
        for j, (images, labels) in enumerate(self.test_loader):
            for i in range(32):
                x = images[i].unsqueeze(0).to(self.device)
                y = labels[i].to(self.device)

                adversary = self.iterative_search(self.G, self.I, C, x, y,
                                                  n_samples=self.n_samples, step=self.step)
                adversary_path = os.path.join(self.adversary_dir,
                                 '{}_{}_{}.jpg'.format(self.classifier, j+1, i+1))
                self.save_adversary(adversary, adversary_path)
                print('Saved natural adversary example...'.format(adversary_path))

    def save_adversary(self, adversary, filename):
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))

        ax[0].imshow(adversary['x'],
                     interpolation='none', cmap=plt.get_cmap('gray'))
        ax[0].text(1, 5, str(adversary['y']), color='white', fontsize=20)
        ax[0].axis('off')

        ax[1].imshow(adversary['x_adv'],
                     interpolation='none', cmap=plt.get_cmap('gray'))
        ax[1].text(1, 5, str(adversary['y_adv']), color='white', fontsize=20)
        ax[1].axis('off')

        fig.savefig(filename)
        plt.close()

    def iterative_search(self, G, I, C, x, y, y_target=None, z=None,
                         n_samples=5000, step=0.01, l=0., h=10., ord=2):
        """
        :param G: function of generator
        :param I: function of inverter
        :param C: function of classifier
        :param x: input instance
        :param y: label
        :param y_target: target label for adversary
        :param z: latent vector corresponding to x
        :param n_samples: number of samples in each search iteration
        :param step: delta r for search step size
        :param l: lower bound of search range
        :param h: upper bound of search range
        :param ord: indicating norm order
        :return: adversary for x against the classifier (d_adv is delta z between z and z_adv)
        """

        x_adv, y_adv, z_adv, d_adv = None, None, None, None
        h = l + step

        if z is None:
            z = I(x)

        while True:
            delta_z = np.random.randn(n_samples, z.size(1))
            d = np.random.rand(n_samples) * (h - l) + l        # random values between the search range [l, h)
            norm_p = np.linalg.norm(delta_z, ord=ord, axis=1)  # L2 norm of delta z along axis=1
            d_norm = np.divide(d, norm_p).reshape(-1, 1)       # rescale/normalize d
            delta_z = np.multiply(delta_z, d_norm)             # random noise vectors of norms within (r, r + delta r]
            delta_z = torch.from_numpy(delta_z).float().to(self.device)
            z_tilde = z + delta_z
            x_tilde = G(z_tilde)
            y_tilde = torch.argmax(C(x_tilde), dim=1)

            if y_target is None:
                indices_adv = np.where(y_tilde != y)[0]
            else:
                indices_adv = np.where(y_tilde == y_target)[0]

            # No candidate generated.
            if len(indices_adv) == 0:
                print('No candidate generated, increasing search range...')
                l = h
                h = l + step

            # Certain candidates generated.
            else:
                # Choose the data index with the least perturbation.
                idx_adv = indices_adv[np.argmin(d[indices_adv])]

                # For debugging.
                if y_target is None:
                    assert (y_tilde[idx_adv] != y)

                else:
                    assert (y_tilde[idx_adv] == y_target)

                # Save natural adversary example.
                if d_adv is None or d[idx_adv] < d_adv:
                    x_adv = x_tilde[idx_adv]
                    y_adv = y_tilde[idx_adv]
                    z_adv = z_tilde[idx_adv]
                    d_adv = d[idx_adv]

                    if y_target is None:
                        print("Untarget y=%d y_adv=%d d_adv=%.4f l=%.4f h=%.4f" % (y, y_adv, d_adv, l, h))
                    else:
                        print("Targeted y=%d y_adv=%d d_adv=%.4f l=%.4f h=%.4f" % (y, y_adv, d_adv, l, h))

                    break

        adversary = {'x': x.squeeze(0).squeeze(0).data.cpu().numpy(),
                     'y': y.data.cpu().numpy(),
                     'z': z.squeeze(0).data.cpu().numpy(),
                     'x_adv': x_adv.squeeze(0).data.cpu().numpy(),
                     'y_adv': y_adv.data.cpu().numpy(),
                     'z_adv': z_adv.data.cpu().numpy(),
                     'd_adv': d_adv}

        return adversary
