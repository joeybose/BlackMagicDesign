import numpy as np
import torch
import torchvision
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.models import resnet50
import torchvision.utils as vutils
from torchvision.utils import save_image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from random import randint
from PIL import Image
import os
from attack_models import GCN
import ipdb

def reduce_sum(x, keepdim=True):
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    return x.squeeze().sum()

def L2_dist(x, y):
    return reduce_sum((x - y)**2)

def to_cuda(model):
    cuda_stat = torch.cuda.is_available()
    if cuda_stat:
        model = torch.nn.DataParallel(model,\
                device_ids=range(torch.cuda.device_count())).cuda()
    return model

def tensor_to_cuda(x):
    cuda_stat = torch.cuda.is_available()
    if cuda_stat:
        x = x.cuda()
    return x

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        # correct = torch.sum(indices == labels.cuda())
        results = (indices == labels.cuda())
        correct = torch.sum(results)
        correct_indices = (results != 0).nonzero()
        return correct.item() * 1.0 / len(labels), correct_indices

def get_data(args):
    """
    Data loader. For now, just a test sample
    """
    args.syn_train_ratio = 0.1
    args.syn_val_ratio = 0.1
    args.syn_test_ratio = 0.8
    data = load_data(args)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.ByteTensor(data.train_mask)
    val_mask = torch.ByteTensor(data.val_mask)
    test_mask = torch.ByteTensor(data.test_mask)
    args.in_feats = features.shape[1]
    args.classes = data.num_labels
    args.n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (args.n_edges, args.classes,
              train_mask.sum().item(),
              val_mask.sum().item(),
              test_mask.sum().item()))

    train_mask = train_mask.cuda()
    val_mask = val_mask.cuda()
    test_mask = test_mask.cuda()
    stop_number = int(np.round(len(labels)*0.1))
    attacker_mask = torch.ByteTensor(sample_mask(range(stop_number), labels.shape[0]))
    target_mask = torch.ByteTensor(sample_mask(range(stop_number), labels.shape[0]))
    return features, labels, train_mask, val_mask, test_mask, data

def load_unk_model(args,data,features,labels):
    """
    Load an unknown model. Used for convenience to easily swap unk model
    """
    g = data.graph
    # add self loop
    if args.self_loop:
        g.remove_edges_from(g.selfloop_edges())
        g.add_edges_from(zip(g.nodes(), g.nodes()))
    g = DGLGraph(g)
    n_edges = g.number_of_edges()
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(g,
                args.in_feats,
                args.n_hidden,
                args.classes,
                args.n_layers,
                F.relu,
                args.dropout).cuda()
    load_file = 'saved_models/'+args.dataset+'_graph_classifier.pt'
    model.load_state_dict(torch.load('saved_models/graph_classifier.pt'))
    # model.load_state_dict(torch.load(load_file))
    model.eval()
    return model

def train_classifier(args, model, device,features, labels, train_mask, val_mask, test_mask, data):
    model.train()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))
    torch.save(model.state_dict(),'saved_models/graph_classifier.pt')

def freeze_model(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False
    return model


