import datetime
from PIL import Image
from torchvision import transforms
from torch import autograd
import torch
from models import *
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
from torch.distributions import Categorical
import json
import os
import numpy as np
import argparse
from tqdm import tqdm
from utils import *
import ipdb
from advertorch.attacks import LinfPGDAttack
from tools.nearest import DiffNearestNeighbours

def whitebox_pgd(args, image, target, model, normalize=None):
    adversary = LinfPGDAttack(
	model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
	nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
	targeted=False)
    adv_image = adversary.perturb(image, target)
    print("Target is %d" %(target))
    pred = model(adv_image)
    out = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
    print("Adv Target is %d" %(out))
    clean_image = (image)[0].detach()
    adv_image = adv_image[0].detach()
    if args.comet:
        plot_image_to_comet(args,clean_image,"clean.png")
        plot_image_to_comet(args,adv_image,"Adv.png")
    return pred, clamp(clean_image - adv_image,0.,1.)

def white_box_untargeted(args, image, target, model, enc=None, dec=None, \
        vae=None, ae= None, normalize=None):
    epsilon = 0.3
    # Create noise vector
    delta = torch.zeros_like(image,requires_grad=True).to(args.device)
    # Optimize noise vector (only) to fool model
    x = image

    use_vae = True if (vae is not None) else False
    use_ae = True if (ae is not None) else False
    small_changes = 0
    print("Target is %d" %(target))
    for t in range(args.PGD_steps):
        if normalize is not None:
            if use_vae:
                x = x.view(x.size(0), -1).unsqueeze(0)
                z, mu, logvar = vae(x)
                z = z.clamp(0, 1)
                x = z.view(z.size(0), 1, 28, 28)
            elif use_ae:
                x = ae(x)
            pred = model(normalize(x + delta))
        else:
            if use_vae:
                x = x.view(x.size(0), -1).unsqueeze(0)
                z, mu, logvar = vae(x)
                z = z.clamp(0, 1)
                x = z.view(z.size(0), 1, 28, 28)
            elif use_ae:
                x = ae(x)
            pred = model(x.detach() + delta)
            recon_pred = model(x.detach())
        out = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
        recon_out = recon_pred.max(1, keepdim=True)[1] # get the index of the max log-probability
        loss = nn.CrossEntropyLoss(reduction="sum")(pred, target)
        recon_image = (x)[0].detach()
        if args.comet:
            args.experiment.log_metric("Whitebox CE loss",loss,step=t)
            plot_image_to_comet(args,recon_image,"recon.png")
        if t % 5 == 0:
            print(t, out[0][0], recon_out[0][0], loss.item())

        loss.backward()
        grad_sign = delta.grad.data.sign()
        delta.data = delta.data + batch_multiply(0.01, grad_sign)
        # Clipping is equivalent to projecting back onto the l_\infty ball
        # This technique is known as projected gradient descent (PGD)
        delta.data.clamp_(-epsilon, epsilon)
        delta.data = clamp(x.data + delta.data,0.,1.) - x.data
        delta.grad.data.zero_()
        # if out != target:
            # print(t, out[0][0], loss.item())
            # break
    if args.comet:
        if not args.mnist:
            clean_image = (image)[0].detach().cpu().numpy().transpose(1,2,0)
            adv_image = (x + delta)[0].detach().cpu().numpy().transpose(1,2,0)
            delta_image = (delta)[0].detach().cpu().numpy().transpose(1,2,0)
        else:
            clean_image = (image)[0].detach()
            adv_image = (x + delta)[0].detach()
            recon_image = (x)[0].detach()
            delta_image = (delta)[0].detach().cpu()
        plot_image_to_comet(args,clean_image,"clean.png")
        plot_image_to_comet(args,adv_image,"Adv.png")
        plot_image_to_comet(args,delta_image,"delta.png")
        plot_image_to_comet(args,recon_image,"recon.png")
    return out, delta

def single_white_box_generator(args, image, target, model, G):
    epsilon = 0.5
    # Create noise vector
    x = image
    opt = optim.SGD(G.parameters(), lr=1e-2)

    print("Target is %d" %(target))
    for t in range(args.PGD_steps):
        delta, kl_div = G(x)
        delta = delta.view(delta.size(0), 1, 28, 28)
        delta.data.clamp_(-epsilon, epsilon)
        delta.data = clamp(x.data + delta.data,0.,1.) - x.data
        pred = model(x.detach() + delta)
        out = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
        loss = -nn.CrossEntropyLoss(reduction="sum")(pred, target)
        if args.comet:
            args.experiment.log_metric("Whitebox CE loss",loss,step=t)
        if t % 5 == 0:
            print(t, out[0][0], loss.item())
        opt.zero_grad()
        loss.backward()
        for p in G.parameters():
            p.grad.data.sign_()
        # Clipping is equivalent to projecting back onto the l_\infty ball
        # This technique is known as projected gradient descent (PGD)
        # delta.data.clamp_(-epsilon, epsilon)
        # delta.data = clamp(x.data + delta.data,0.,1.) - x.data
        opt.step()
        if out != target:
            print(t, out[0][0], loss.item())
            break
    if args.comet:
        if not args.mnist:
            clean_image = (image)[0].detach().cpu().numpy().transpose(1,2,0)
            adv_image = (x + delta)[0].detach().cpu().numpy().transpose(1,2,0)
            delta_image = (delta)[0].detach().cpu().numpy().transpose(1,2,0)
        else:
            clean_image = (image)[0].detach()
            adv_image = (x + delta)[0].detach()
            delta_image = (delta)[0].detach()
        plot_image_to_comet(args,clean_image,"clean.png")
        plot_image_to_comet(args,adv_image,"Adv.png")
        plot_image_to_comet(args,delta_image,"delta.png")
    return out, delta

def PGD_test_model(args,epoch,test_loader,model,G,nc=1,h=28,w=28):
    ''' Testing Phase '''
    epsilon = args.epsilon
    test_itr = tqdm(enumerate(test_loader),\
            total=len(test_loader)/args.test_batch_size)
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
            .format(correct_test, len(test_loader),\
                100. * correct_test / len(test_loader)))
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

def L2_test_model(args,epoch,test_loader,model,G):
    model.eval()
    ''' Testing Phase '''
    test_itr = tqdm(enumerate(test_loader),\
            total=len(test_loader)/args.batch_size)
    correct_test = 0
    for batch_idx, batch in enumerate(test_itr):
        x, target = batch[1]['text'].cuda(), batch[1]['labels'].cuda()
        # # output: batch x seq_len x ntokens
        # with torch.no_grad():
        input_embeddings = model.get_embeds(x)
        delta_embeddings, kl_div = G(x)
        adv_embeddings = input_embeddings.detach() + delta_embeddings
        preds = model(adv_embeddings,use_embed=True)
        prob, idx = torch.max(F.log_softmax(preds,dim=1), 1)
        # _ = decode_to_natural_lang(x[0],args)
        # _ = decode_to_natural_lang(adv_out[0],args)
        correct_test += idx.eq(target.data).sum()

    print('\nAdversarial noise Test set: Accuracy: {}/{} ({:.0f}%)\n'\
            .format(correct_test, len(test_loader.dataset),\
                100. * correct_test / len(test_loader.dataset)))
    model.train()
    if args.comet:
        args.experiment.log_metric("Test Adv Accuracy",\
                100.*correct_test/len(test_loader.dataset),step=epoch)
    # if args.comet:
        # file_base = "adv_images/" + args.namestr + "/"
        # if not os.path.exists(file_base):
            # os.makedirs(file_base)
        # plot_image_to_comet(args,clean_image,file_base+"clean.png",normalize=True)
        # plot_image_to_comet(args,adv_image,file_base+"Adv.png",normalize=True)
        # plot_image_to_comet(args,delta_image,file_base+"delta.png",normalize=True)

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
                total=len(train_loader)/args.batch_size)
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
                    100.*correct/len(train_loader),step=epoch)

        print('\nTrain: Epoch:{} Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'\
                .format(epoch,\
                    loss, correct, len(train_loader),
                    100. * correct / len(train_loader)))

    return out, delta

def train_ae(args, train_loader, G):
    opt = optim.Adam(G.parameters())
    criterion_ce = nn.CrossEntropyLoss()

    ''' Training Phase '''
    train_itr = tqdm(enumerate(train_loader),\
            total=len(train_loader)/args.batch_size)
    correct = 0
    ntokens = len(args.alphabet)

    # Only 1 Epoch because it already overfits
    ipdb.set_trace()
    for batch_idx, batch in enumerate(train_itr):
        if batch_idx > args.burn_in:
            break
        x, target = batch[1].text, batch[1].label
        iter_count = 0
        opt.zero_grad()

        # output: batch x seq_len x ntokens
        if not args.vanilla_G:
            masked_output, masked_target, kl_div  = G(x)
            kl_div = kl_div.sum() / len(x)
        else:
            output = G(x)

        # output_size: batch_size, maxlen, self.ntokens
        # flattened_output = output.view(-1, ntokens)

        # masked_output = flattened_output.masked_select(output_mask).view(-1, ntokens)
        loss = criterion_ce(masked_output, masked_target)
        loss.backward()
        # # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
        torch.nn.utils.clip_grad_norm_(G.parameters(), args.clip)
        opt.step()
        accuracy = None
        if batch_idx % 10 == 0:
            # accuracy
            probs = F.softmax(masked_output,dim=1)
            max_vals, max_indices = torch.max(probs, 1)
            accuracy = torch.mean(max_indices.eq(masked_target).float()).item()
            print("Batch %d Loss %f Accuracy %f" %(batch_idx,loss.item(),accuracy))
        if args.comet:
            args.experiment.log_metric("VAE Misclassification loss",\
                    loss,step=batch_idx)
            args.experiment.log_metric("VAE Accuracy",\
                    accuracy,step=batch_idx)

def L2_white_box_generator(args, train_loader, test_loader, model, G):
    """
    Args:
        args         : ArgumentParser args
        train_loader : (torchtext obj) data to train
        test_loader  : (torchtext obj) data to test
        model        : the model you are attacking
        G            : the model which generates adversarial samples for `model`
    """
    epsilon = args.epsilon
    opt = optim.Adam(G.parameters())
    criterion_ce = nn.CrossEntropyLoss()
    if args.carlini_loss:
        misclassify_loss_func = carlini_wagner_loss
    else:
        misclassify_loss_func = CE_loss_func

    ''' Burn in VAE '''
    if args.train_ae:
        print("Doing Burn in VAE")
        train_ae(args, train_loader, G)
    utils.evaluate(model,test_loader)

    if args.diff_nn:
        # Differentiable nearest neigh auxiliary loss
        diff_nearest_func = DiffNearestNeighbours(args.embeddings, args.device,
                                                  args.nn_temp,100,
                                                  args.distance_func)
        if str(args.device) == 'cuda' and not args.no_parallel:
            diff_nearest_func = nn.DataParallel(diff_nearest_func)

    ''' Training Phase '''
    for epoch in range(0,args.attack_epochs):
        # neig_eg, test_accuracies = utils.evaluate_neighbours(test_loader,
                                                        # model, G, args, epoch)

        print(datetime.now())
        train_itr = tqdm(enumerate(train_loader),\
                total=len(train_loader.dataset)/args.batch_size)
        correct = 0
        ntokens = len(args.alphabet)
        # L2_test_model(args,epoch,test_loader,model,G)
        for batch_idx, batch in enumerate(train_itr):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            x, y = batch[1]['text'].cuda(), batch[1]['labels'].cuda()
            num_unperturbed = 10
            iter_count = 0
            loss_misclassify = 10
            loss_perturb = 10
            iter_count = 0

            ''' Get input embeddings from Model '''
            x_embeddings = model.get_embeds(x)

            while loss_misclassify > 0 and loss_perturb > 1:
                opt.zero_grad()
                input_embeddings = x_embeddings.detach()

                # output: batch x seq_len x ntokens
                if not args.vanilla_G:
                    delta_embeddings, kl_div = G(x)
                    kl_div = kl_div.mean()
                else:
                    output = G(x)

                adv_embeddings = input_embeddings + delta_embeddings

                if args.diff_nn:
                    # Differentiable nearest neighbour
                    adv_embeddings = diff_nearest_func(adv_embeddings)

                # Losses
                # TODO: still need L2?
                l2_dist = L2_dist(input_embeddings,adv_embeddings)
                loss_perturb =  l2_dist / len(input_embeddings)

                # Evaluate target model with adversarial samples
                preds = model(adv_embeddings,use_embed=True)
                loss_misclassify = misclassify_loss_func(args,preds,y)

                loss = loss_misclassify + args.LAMBDA * loss_perturb + kl_div
                loss.backward()
                # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
                torch.nn.utils.clip_grad_norm_(G.parameters(), args.clip)
                opt.step()
                out = torch.max(F.log_softmax(preds,dim=1), 1)[1]
                correct = out.eq(y.data).sum()

                iter_count = iter_count + 1
                if iter_count > args.max_iter:
                    break
            correct += out.eq(y.data).sum()

        # Get examples of nearest neighbour text and scores
        neig_eg, train_accuracies = utils.evaluate_neighbours(test_loader,
                                                        model, G, args, epoch, mode='Train')
        neig_eg, test_accuracies = utils.evaluate_neighbours(test_loader,
                                                        model, G, args, epoch)
        if args.comet:
            args.experiment.log_text(neig_eg)
            args.experiment.log_metric("Whitebox Total loss",loss,step=epoch)
            args.experiment.log_metric("Whitebox Recon loss",loss_perturb,step=epoch)
            args.experiment.log_metric("Whitebox Misclassification loss",\
                    loss_misclassify,step=epoch)
            args.experiment.log_metric("Adv Accuracy",\
                    100.*correct/len(train_loader.dataset),step=epoch)

            # Log orig accuracy, perturbed emb acc and perturbed tok acc
            for k, v in test_accuracies.items():
                args.experiment.log_metric(k, v,step=epoch)
        print("Misclassification Loss: %f Perturb Loss %f" %(\
                                                loss_misclassify,loss_perturb))
        print('\nTrain: Epoch:{} Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'\
                .format(epoch,\
                    loss, correct, len(train_loader.dataset),
                    100. * correct / len(train_loader.dataset)))


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

def reinforce_seq_loss(cumulative_rewards, fake_logits, fake_sentence, baseline, args):
    # cumulative rewards : bs x seq_len
    # fake logits        : bs x seq_len x vocab_size  (distribution @ every timestep)
    # fake sentence      : bs x seq_len               (indices for the words)
    # baseline           : bs x seq_len               (baseline coming from critic)
    # assert cumulative_rewards.shape == baseline.shape == fake_sentence.shape
    assert cumulative_rewards.shape == fake_sentence.shape

    bs, seq_len, vocab_size = fake_logits.shape
    # advantages = cumulative_rewards

    # use a baseline in regular mode
    # if args.use_baseline:
        # advantages -= baseline
    # if args.adv_clip > 0:
        # advantages = torch.clamp(advantages, -args.adv_clip, args.adv_clip)
    cumulative_rewards.detach()

    loss = 0.
    for t in range(seq_len):
        dist = Categorical(logits=fake_logits[:, t])
        log_prob = dist.log_prob(fake_sentence[:, t])
        ment_reg = args.beta * dist.entropy()
        loss += 1*log_prob *cumulative_rewards[:, t] + ment_reg
    # return -loss.sum() / bs average loss over batches
    return loss.sum() / bs # average loss over batches

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


