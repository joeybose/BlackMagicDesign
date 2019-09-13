from comet_ml import Experiment
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
from nag_model import *
import time
import ipdb
from torchvision import models
from torch import optim
from torchvision.utils import save_image
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import os
import sys
sys.path.append("..")
import utils, models
from cnn_models import *
import flows
from torch.nn.utils import clip_grad_norm_


#Some constants for training, change them accordingly
size=32
lr = 1e-3
epochs = 20
LAMBDA = 0.1

def load_unk_model(args,retrain=False):
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
        if os.path.exists("./saved_models/cifar_VGG16.pt") and retrain == False:
            # model = DenseNet121().to(args.device)
            model = VGG().to(args.device)
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load("./saved_models/cifar_VGG16.pt"))
            model.eval()
        else:
            model = utils.main_cifar(args,normalize=False)
    else:
        # load pre-trained ResNet50
        model = resnet50(pretrained=True).to(args.device)
    return model

def do_validation(args,epoch,val_loader,unk_model,model):
    total_fool = 0
    print ("{}".format("############### VALIDATION PHASE STARTED ################"))
    model.eval()
    for j, (images_val,target_val) in enumerate(val_loader):
        with torch.no_grad():
            images_val = images_val.to(args.device)
            target_val = target_val.to(args.device)
            z_ref_val = make_z(args, (len(images_val),\
                                100, 1, 1), minval=-1., maxval=1.)
            perturbations_val = model(z_ref_val).detach()
            random_adv_batch_val = add_perturbation(images_val.detach(), perturbations_val)
            v_val, topk_val = scores(unk_model,images_val.detach())
            v_adv_val, topk_adv_val = scores(unk_model,random_adv_batch_val.detach())
            correct = torch.eq(target_val.view(-1,1),topk_adv_val.view(-1,1)).sum()
            nfool = len(images_val) - correct
            total_fool = total_fool + nfool
    foolr = 100*float(total_fool)/(len(val_loader)*args.test_batch_size)
    print("{} {} {}".format("Fooling rate",foolr,total_fool))
    print ("{}".format("############### VALIDATION PHASE ENDED ################"))
    args.experiment.log_metric("Val Adv Fool Rate",foolr,step=epoch)

def add_perturbation(inp, perturbation):
    # 10/256
    epsilon = 0.04
    perturbation.data.clamp_(-epsilon, epsilon)
    adv_inputs = inp + perturbation
    adv_inputs = torch.clamp(adv_inputs, 0, 1.0)
    return adv_inputs

def log_loss(prob_vec, adv_prob_vec, top_prob):
    size = prob_vec.size()[0]
    for i in range(size):
        if i==0:
            loss = adv_prob_vec[i][top_prob[i][0]]
        else:
            loss = loss + adv_prob_vec[i][top_prob[i][0]]

        mean = (loss/size)
        gen_loss = - torch.log(1 - mean)
        return gen_loss, mean

def scores(net, inp_val):
    out1 = net(inp_val)
    out = F.softmax(out1)
    _, topk = torch.topk(out, 1)
    return out, topk

def make_z(args, shape, minval, maxval):
    z = minval + torch.rand(shape) * (maxval - 1)
    return z.to(args.device)

def save_checkpoint(state, epoch):
    ckpt_dir = 'home/vkv/NAG/ckpt/'
    print("[*] Saving model to {}".format(ckpt_dir))

    filename = 'NAG' + '_ckpt.pth.tar'
    ckpt_path = os.path.join(ckpt_dir, filename)
    torch.save(state, ckpt_path)

normalize=utils.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])

parser = argparse.ArgumentParser(description='BlackBox')
parser.add_argument("--comet_username", type=str, default="joeybose",
        help='Username for comet logging')
parser.add_argument("--comet_apikey", type=str,\
        default="Ht9lkWvTm58fRo9ccgpabq5zV",help='Api for comet logging')
parser.add_argument('--model_path', type=str, default="mnist_cnn.pt",
                    help='where to save/load')
parser.add_argument('--mnist', default=False, action='store_true',
                    help='Use MNIST as Dataset')
parser.add_argument('--cifar', default=True, action='store_true',
                    help='Use CIFAR-10 as Dataset')
parser.add_argument('--test_batch_size', type=int, default=256, metavar='S',
                    help='Test Batch size')
parser.add_argument('--batch_size', type=int, default=256, metavar='S',
                    help='Batch size')
parser.add_argument('--namestr', type=str, default='NAG', \
        help='additional info in output filename to describe experiments')
args = parser.parse_args()
args.input_size = 32*32*3
args.device = torch.device("cuda")

with open('../settings.json') as f:
    data = json.load(f)
args.comet_apikey = data["apikey"]
args.comet_username = data["username"]
experiment = Experiment(api_key=args.comet_apikey,\
        project_name="black-magic-design",\
        workspace=args.comet_username)
experiment.set_name(args.namestr)
args.experiment = experiment

train_loader,val_loader = utils.get_data(args,normalize=False)
model = DCGAN(args.batch_size,nz=100).to(args.device)

# The unknown model to attack
unk_model = load_unk_model(args,retrain=False)
optimizer = optim.Adam(model.parameters())
pdist = nn.PairwiseDistance(p=2)
val_counter = 0
for epoch in range(epochs):
        total_fool = 0
        for i, (images, target) in enumerate(train_loader):
            images = images.to(args.device)
            target = target.to(args.device)
            z_ref = make_z(args, (len(images), 100,1,1), minval=-1., maxval=1.)
            model.train()
            model.zero_grad()
            perturbations = model(z_ref)
            random_adv_batch = add_perturbation(images, perturbations)
            perm_idx = torch.randperm(len(images))
            random_adv_batch2 = random_adv_batch[perm_idx]
            v, topk = scores(unk_model,images)
            v_adv, topk_adv = scores(unk_model,random_adv_batch)
            v_adv2, _ = scores(unk_model,random_adv_batch2)
            outputs = []
            def hook(module, input, output):
                outputs.append(output)
            unk_model.module.features[10].register_forward_hook(hook)
            f1 = unk_model(random_adv_batch)
            f2 = unk_model(random_adv_batch2)
            f1_res4f = outputs[0]
            f2_res4f = outputs[1]
            feature_loss = -1*utils.L2_norm_dist(f1_res4f, f2_res4f)
            q1_loss, meanq1 = log_loss(v, v_adv, topk)
            q1_loss = q1_loss + LAMBDA*feature_loss
            q1_loss.backward()
            optimizer.step()
            if i!=0 and i%100==0:
                clean_image = (images)[0:16].detach()
                adv_image = (random_adv_batch)[0:16].detach()
                delta_image = (perturbations)[0:16].detach().cpu()
                utils.plot_image_to_comet(args,clean_image,"clean.png",normalize=True)
                utils.plot_image_to_comet(args,adv_image,"Adv.png",normalize=True)
                utils.plot_image_to_comet(args,delta_image,"delta.png",normalize=True)
            print ("{} {} {} {} {} {} {} {} {} {}".format("Epoch",epoch,"Iteration",\
                              i,"Log loss",q1_loss.item(),"Mean",meanq1.item(),"Feature loss",feature_loss.item()))
            correct = torch.eq(target.view(-1,1),topk_adv.view(-1,1)).sum()
            nfool = len(images) - correct
            total_fool = total_fool + nfool
            if i!=0 and i%100==0:
                val_counter += 1
                do_validation(args,val_counter,val_loader,unk_model,model)
        foolr = 100*float(total_fool)/(len(train_loader)*args.batch_size)
        print("{} {} {}".format("Train Fooling rate",foolr,total_fool))
        args.experiment.log_metric("Train Adv Fool Rate",foolr,step=epoch)



