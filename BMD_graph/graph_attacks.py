from PIL import Image
from torchvision import transforms
import torch
from attack_models import *
from torch import nn, optim
from torchvision.models import resnet50
from torchvision.models.vgg import VGG
import torchvision.models.densenet as densenet
import torchvision.models.alexnet as alexnet
from torchvision.utils import save_image
import torch.nn.functional as F
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
from advertorch.utils import clamp
from torch import optim
from torch.autograd import Variable
import itertools
import json
import os
import numpy as np
import argparse
from tqdm import tqdm
from utils import *
import ipdb
from advertorch.attacks import LinfPGDAttack

def PGD_test_model(args,epoch,test_loader,model,G,nc=1,h=28,w=28):
    ''' Testing Phase '''
    epsilon = args.epsilon
    test_itr = tqdm(enumerate(test_loader),\
            total=len(test_loader.dataset)/args.test_batch_size)
    correct_test = 0
    for batch_idx, (data, target) in test_itr:
        x, target = data.to(args.device), target.to(args.device)
        # for t in range(args.PGD_steps):
        if not args.vanilla_G:
            delta, kl_div  = G(x)
        else:
            delta = G(x)
        delta = delta.view(delta.size(0), nc, h, w)
        # Clipping is equivalent to projecting back onto the l_\infty ball
        # This technique is known as projected gradient descent (PGD)
        delta.data.clamp_(-epsilon, epsilon)
        delta.data = torch.clamp(x.data + delta.data,-1.,1.) - x.data
        pred = model(x.detach() + delta)
        out = pred.max(1, keepdim=True)[1] # get the index of the max log-probability

        correct_test += out.eq(target.unsqueeze(1).data).sum()

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'\
            .format(correct_test, len(test_loader.dataset),\
                100. * correct_test / len(test_loader.dataset)))
    if args.comet:
        if not args.mnist:
            index = np.random.choice(len(x) - 64, 1)[0]
            clean_image = (x)[index:index+64].detach()#.permute(-1,1,2,0)
            adv_image = (x + delta)[index:index+64].detach()#.permute(-1,1,2,0)
            delta_image = (delta)[index:index+64].detach()#.permute(-1,1,2,0)
        else:
            clean_image = (x)[0].detach()
            adv_image = (x + delta)[0].detach()
            delta_image = (delta)[0].detach()
        plot_image_to_comet(args,clean_image,"clean.png",normalize=True)
        plot_image_to_comet(args,adv_image,"Adv.png",normalize=True)
        plot_image_to_comet(args,delta_image,"delta.png",normalize=True)

def L2_test_model(args,epoch,features,labels,adj_mat,test_mask,\
                  data,model,G,target_mask=None,attack_mask=None,return_correct=False):

    ''' Testing Phase '''
    correct_test = 0
    if args.attack_adj:
        x = adj_mat
    else:
        x = features

    with torch.no_grad():
        delta, kl_div  = G(features,adj_mat)
        if args.influencer_attack:
            delta = delta*attack_mask

    adv_inputs = features.detach() + delta
    logits = model(adv_inputs)
    if args.influencer_attack:
        pred = logits[target_mask]
        masked_labels = labels[target_mask]
    else:
        pred = logits[test_mask]
        masked_labels = labels[test_mask]
    _, indices = torch.max(pred, dim=1)
    correct_test = torch.sum(indices.data.cpu() == masked_labels)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'\
            .format(correct_test, len(masked_labels),\
                100. * correct_test / len(masked_labels)))
    if args.comet:
        args.experiment.log_metric("Test Adv Accuracy",\
                100.*correct_test/len(masked_labels),step=epoch)
    if return_correct:
        return correct_test

def carlini_wagner_loss(args, output, target, scale_const=1):
    # compute the probability of the label class versus the maximum other
    target_onehot = torch.zeros(target.size() + (args.classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    confidence = 0
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    # if targeted:
        # # if targeted, optimize for making the other class most likely
        # loss1 = torch.clamp(other - real + confidence, min=0.)  # equiv to max(..., 0.)
    # else:
        # if non-targeted, optimize for making this class least likely.
    loss1 = torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.mean(scale_const * loss1)

    return loss

def PGD_white_box_generator(args, train_loader, test_loader, model, G,\
        nc=1,h=28,w=28):
    epsilon = args.epsilon
    opt = optim.Adam(G.parameters(),lr=1e-4)
    if args.carlini_loss:
        misclassify_loss_func = carlini_wagner_loss
    else:
        misclassify_loss_func = CE_loss_func
    ''' Training Phase '''
    for epoch in range(0,args.attack_epochs):
        train_itr = tqdm(enumerate(train_loader),\
                total=len(train_loader.dataset)/args.batch_size)
        correct = 0
        PGD_test_model(args,epoch,test_loader,model,G,nc,h,w)
        for batch_idx, (data, target) in train_itr:
            x, target = data.to(args.device), target.to(args.device)
            for t in range(args.PGD_steps):
                if not args.vanilla_G:
                    delta, kl_div  = G(x)
                else:
                    delta = G(x)
                delta = delta.view(delta.size(0), nc, h, w)
                # Clipping is equivalent to projecting back onto the l_\infty ball
                # This technique is known as projected gradient descent (PGD)
                delta.data.clamp_(-epsilon, epsilon)
                delta.data = torch.clamp(x.data + delta.data,-1.,1.) - x.data
                pred = model(x.detach() + delta)
                out = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
                loss = misclassify_loss_func(args,pred,target) + kl_div.sum()
                if args.comet:
                    args.experiment.log_metric("Whitebox CE loss",loss,step=t)
                opt.zero_grad()
                loss.backward()
                for p in G.parameters():
                    p.grad.data.sign_()
                opt.step()
            correct += out.eq(target.unsqueeze(1).data).sum()

        if args.comet:
            args.experiment.log_metric("Whitebox CE loss",loss,step=epoch)
            args.experiment.log_metric("Adv Accuracy",\
                    100.*correct/len(train_loader.dataset),step=epoch)

        print('\nTrain: Epoch:{} Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'\
                .format(epoch,\
                    loss, correct, len(train_loader.dataset),
                    100. * correct / len(train_loader.dataset)))

    return out, delta

def get_node_neighbors(g,node,all_ids,min_neighbors=False,target_mask=None):
    predecessors = g.predecessors(node)
    successors = g.successors(node)
    predecessors = set(predecessors.cpu().numpy()[0:5])
    successors = set(successors.cpu().numpy()[0:5])
    neighbors = predecessors.union(successors)
    if min_neighbors:
        if len(neighbors) < 5:
            num_neigh_samples = 5 - len(neighbors)
            if target_mask is not None:
                neigh_set = all_ids - set(target_ids).union(neighbors)
            else:
                neigh_set = all_ids - set(neighbors)
            neigh_array = np.asarray(list(neigh_set))
            sampled_ids = np.random.choice(neigh_array,num_neigh_samples,replace=False)
            neighbors = itertools.chain.from_iterable([neighbors,sampled_ids])
    return list(neighbors)

def create_single_node_masks(g,labels,correct_indices,top_nodes,bot_nodes,\
                      num_target_samples=20,num_attack_samples=200):
    top_nodes = top_nodes.squeeze().cpu().numpy()
    bot_nodes = bot_nodes.squeeze().cpu().numpy()
    all_ids = set(range(len(labels)))
    target_set = set(correct_indices) - set(top_nodes)
    target_set = target_set - set(bot_nodes)
    target_ids = np.random.choice(list(target_set),num_target_samples,replace=False)
    target_ids = list(target_ids) + list(top_nodes) + list(bot_nodes)
    attacker_ids = [get_node_neighbors(g,int(node),all_ids,True) for node in target_ids]
    return target_ids, attacker_ids

def create_node_masks(g,labels,correct_indices,top_nodes,bot_nodes,\
                      num_target_samples=20,num_attack_samples=200):
    top_nodes = top_nodes.squeeze().cpu().numpy()
    bot_nodes = bot_nodes.squeeze().cpu().numpy()
    all_ids = set(range(len(labels)))
    target_set = set(correct_indices) - set(top_nodes)
    target_set = target_set - set(bot_nodes)
    target_ids = np.random.choice(list(target_set),num_target_samples,replace=False)
    target_ids = list(target_ids) + list(top_nodes) + list(bot_nodes)
    attacker_nodes = [get_node_neighbors(g,int(node),all_ids) for node in target_ids]
    attacker_nodes = list(itertools.chain.from_iterable(attacker_nodes))
    num_attack_samples = num_attack_samples - len(attacker_nodes)
    attack_set = all_ids - set(target_ids).union(attacker_nodes)
    attack_array = np.asarray(list(attack_set))
    attack_ids = np.random.choice(attack_array,num_attack_samples,replace=False)
    attack_ids = list(itertools.chain.from_iterable([attacker_nodes,attack_ids]))
    target_mask = torch.ByteTensor(sample_mask(target_ids, labels.shape[0]))
    attack_mask = torch.ByteTensor(sample_mask(attack_ids, labels.shape[0]))
    return target_mask.cuda(), attack_mask.cuda()

def L2_white_box_generator(args, features, labels, train_mask, val_mask, test_mask, data, model, G):
    epsilon = args.epsilon
    opt = optim.Adam(G.parameters())
    if args.carlini_loss:
        misclassify_loss_func = carlini_wagner_loss
    else:
        misclassify_loss_func = CE_loss_func

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
    adj_mat = g.adjacency_matrix().cuda()
    features = features.cuda()

    if args.attack_adj:
        x = adj_mat
    else:
        x = features

    ''' Create Masks '''
    target_mask, attack_mask = None,None
    if args.influencer_attack:
        all_mask = torch.ByteTensor(sample_mask(range(len(labels)), labels.shape[0]))
        acc, correct_indices, top_nodes, bot_nodes = evaluate(model,
                                                              features.cuda(),
                                                              labels, test_mask)
        target_mask, attack_mask = create_node_masks(g,labels,correct_indices.squeeze().cpu().numpy(),\
                          top_nodes,bot_nodes)
        attack_mask = attack_mask.unsqueeze(1).float()
        attack_mask = attack_mask.repeat(1,features.shape[1])

    ''' Training Phase '''
    for epoch in range(0,args.attack_epochs):
        correct = 0
        delta, kl_div  = G(features,adj_mat)
        kl_div = kl_div.sum()

        ''' Projection Step '''
        if args.influencer_attack:
            delta = delta*attack_mask

        adv_inputs = features.detach() + delta

        if args.attack_adj:
            model.g = adv_inputs

        logits = model(adv_inputs)

        if args.influencer_attack:
            pred = logits[target_mask]
            masked_labels = labels[target_mask]
        else:
            pred = logits[train_mask]
            masked_labels = labels[train_mask]

        _, indices = torch.max(pred, dim=1)
        correct = torch.sum(indices.data.cpu() == masked_labels)
        loss_misclassify = misclassify_loss_func(args,pred,masked_labels.cuda()) / len(masked_labels)
        loss_perturb = L2_dist(features,adv_inputs) / len(features)
        loss = loss_misclassify + args.LAMBDA * loss_perturb + kl_div
        opt.zero_grad()
        loss.backward()
        opt.step()

        if args.comet:
            args.experiment.log_metric("Whitebox Total loss",loss,step=epoch)
            args.experiment.log_metric("Whitebox L2 loss",loss_perturb,step=epoch)
            args.experiment.log_metric("Whitebox Misclassification loss",\
                    loss_misclassify,step=epoch)
            args.experiment.log_metric("Adv Accuracy",\
                    100.*correct/len(masked_labels),step=epoch)

        print('\nTrain: Epoch:{} Total Loss: {:.4f}, Delta: {:.4f}, KL: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'\
                .format(epoch,\
                    loss, loss_perturb, kl_div, correct, len(masked_labels),
                    100. * correct / len(masked_labels)))
        L2_test_model(args,epoch,features,labels,adj_mat,test_mask,data,model,G,target_mask,attack_mask)

    if args.single_node_attack and args.influencer_attack:
        single_target_ids, single_attack_ids = create_single_node_masks(g,labels,correct_indices.squeeze().cpu().numpy(),\
                          top_nodes,bot_nodes)
        single_correct_test = 0
        for target_node, attacker_nodes in zip(single_target_ids,single_attack_ids):
            single_target_mask = torch.ByteTensor(sample_mask(target_node,labels.shape[0])).cuda()
            single_attack_mask = torch.ByteTensor(sample_mask(attacker_nodes, labels.shape[0]))
            single_attack_mask = single_attack_mask.unsqueeze(1).float()
            single_attack_mask = single_attack_mask.repeat(1,features.shape[1]).cuda()
            single_correct_test += L2_test_model(args,epoch,features,labels,adj_mat,test_mask,data,model,\
                          G,single_target_mask,single_attack_mask,return_correct=True)
        single_acc = single_correct_test.data.cpu().numpy() / len(single_target_ids)
        print("Single Accuracy is %f " %(single_acc))

def soft_reward(pred, targ):
    """
    BlackBox adversarial soft reward. Highest reward when `pred` for `targ`
    class is low. Use this reward to reinforce action gradients.

    Computed as: 1 - (targ pred).
    Args:
        pred: model log prediction vector, to be normalized below
        targ: true class integer, we want to decrease probability of this class
    """
    # pred = F.softmax(pred, dim=1)
    pred_prob = torch.exp(pred)
    gather = pred[:,targ] # gather target predictions
    ones = torch.ones_like(gather)
    r = ones - gather
    r = r.mean()

    return r

def hard_reward(pred, targ):
    """
    BlackBox adversarial 0/1 reward.
    1 if predict something other than target, 0 if predict target. This reward
    should make it much harder to optimize a black box attacker.
    """
    pred = F.softmax(pred, dim=1)
    out = pred.max(1, keepdim=True)[1] # get the index of the max log-prob

def CE_loss_func(args,pred, targ):
    """
    Want to maximize CE, so return negative since optimizer -> gradient descent
    Args:
        pred: model prediction
        targ: true class, we want to decrease probability of this class
    """
    loss = -nn.CrossEntropyLoss(reduction="sum")(pred, targ)
    loss = loss / len(targ)

    return loss

def linf_constraint(grad):
    """
    Constrain delta to l_infty ball
    """
    return torch.sign(grad)

def reinforce(log_prob, f, **kwargs):
    """
    Based on
    https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
    """
    policy_loss = (-log_prob) * f.detach()
    return policy_loss

def reinforce_new(log_prob, f, **kwargs):
    policy_loss = (-log_prob) * f.detach()
    d_loss = torch.autograd.grad([policy_loss.mean()], [log_prob],
                                        create_graph=True,retain_graph=True)[0]
    return d_loss.detach()

def lax_black(log_prob, f, f_cv, param, cv, cv_opt):
    """
    Returns policy loss equivalent to:
    (f(x) - c(x))*grad(log(policy)) + grad(c(x))
    The l_infty constraint should appear elsewhere
    Args:
        f: unknown function
        f_cv: control variate

    Checkout https://github.com/duvenaud/relax/blob/master/pytorch_toy.py
    """
    log_prob = (-1)*log_prob
    # Gradients of log_prob wrt to Gaussian params
    d_params_probs = torch.autograd.grad([log_prob.sum()],param,
                                    create_graph=True, retain_graph=True)

    # Gradients of cont var wrt to Gaussian params
    d_params_cont = torch.autograd.grad([f_cv], param,
                                    create_graph=True, retain_graph=True)


    # Difference between f and control variate
    ac = f - f_cv

    # Scale gradient, negative cv gradient since reward
    d_log_prob = []
    for p, c in zip(d_params_probs, d_params_cont):
        d_log_prob.append(ac*p - c)

    # Backprop param gradients
    for p, g in zip(param, d_log_prob):
        p.backward(g.detach(), retain_graph=True)

    # Optimize control variate to minimize variance
    var = sum([v**2 for v in d_log_prob])
    d_var = torch.autograd.grad([var.mean()], cv.parameters(),
                                    create_graph=True, retain_graph=True)

    # Set gradients to control variate params
    for p, g in zip(cv.parameters(), d_var):
        p.grad = g

    cv_opt.step()

    return None


