import torch
import math
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal
import ipdb
from torch.distributions import Normal
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h, self.g)
        return h

"""
Taken from: https://github.com/vmasrani/gae_in_pytorch/blob/master/layers.py
"""
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init_range = math.sqrt(6.0 / (self.in_features + self.out_features))
        self.weight.data.uniform_(-init_range, init_range)
        if self.bias is not None:
            self.bias.data.uniform_(-init_range, init_range)

    def forward(self, x, adj):
        support = SparseMM()(x, self.weight)
        # support = torch.mm(x, self.weight)
        output = SparseMM()(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.sigmoid = nn.Sigmoid()
        self.fudge = 1e-7

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = (self.sigmoid(torch.mm(z, z.t())) + self.fudge) * (1 - 2 * self.fudge)
        return adj

class GAE(nn.Module):
    """Graph Auto Encoder (see: https://arxiv.org/abs/1611.07308)"""

    def __init__(self, data, n_hidden, n_latent, dropout, subsampling=False):
        super(GAE, self).__init__()

        # Data
        self.x = data['features']
        self.adj_norm = data['adj_norm']
        self.adj_labels = data['adj_labels']
        self.obs = self.adj_labels.view(1, -1)

        # Dimensions
        N, D = data['features'].shape
        self.n_samples = N
        self.n_edges = self.adj_labels.sum()
        self.n_subsample = 2 * self.n_edges
        self.input_dim = D
        self.n_hidden = n_hidden
        self.n_latent = n_latent

        # Parameters
        self.pos_weight = float(N * N - self.n_edges) / self.n_edges
        self.norm = float(N * N) / ((N * N - self.n_edges) * 2)
        self.subsampling = subsampling

        # Layers
        self.dropout = dropout
        self.encoder = GCNEncoder(self.input_dim, self.n_hidden, self.n_latent, self.dropout)
        self.decoder = InnerProductDecoder(self.dropout)

    def model(self):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)

        # Setup hyperparameters for prior p(z)
        z_mu    = torch.zeros([self.n_samples, self.n_latent])
        z_sigma = torch.ones([self.n_samples, self.n_latent])

        # sample from prior
        z = pyro.sample("latent", dist.Normal(z_mu, z_sigma).to_event(2))

        # decode the latent code z
        z_adj = self.decoder(z).view(1, -1)

        # Score against data
        pyro.sample('obs', WeightedBernoulli(z_adj, weight=self.pos_weight).to_event(2), obs=self.obs)

    def guide(self):
        # register PyTorch model 'encoder' w/ pyro
        pyro.module("encoder", self.encoder)

        # Use the encoder to get the parameters use to define q(z|x)
        z_mu, z_sigma = self.encoder(self.x, self.adj_norm)

        # Sample the latent code z
        pyro.sample("latent", dist.Normal(z_mu, z_sigma).to_event(2))

    def get_embeddings(self):
        z_mu, _ = self.encoder.eval()(self.x, self.adj_norm)
        # Put encoder back into training mode
        self.encoder.train()
        return z_mu

# Initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)

