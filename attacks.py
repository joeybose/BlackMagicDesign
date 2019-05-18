from PIL import Image
from torchvision import transforms
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
import json
import os
import numpy as np
import argparse
from tqdm import tqdm
from utils import *
import ipdb
from advertorch.attacks import LinfPGDAttack

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

def PGD_generate_multiple_samples(args,epoch,test_loader,model,G,nc=1,h=28,w=28):
    epsilon = args.epsilon
    test_itr = tqdm(enumerate(test_loader),\
            total=len(test_loader.dataset)/args.test_batch_size)
    correct_test = 0
    correct_batch_avg_list = []
    for batch_idx, (data, target) in test_itr:
        x, target = data.to(args.device), target.to(args.device)
        correct_batch_avg = 0
        for t in range(10):
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
            correct_batch_avg = out.eq(target.unsqueeze(1).data).sum()

        correct_batch_avg = correct_batch_avg / (10*len(x))
        correct_batch_avg_list.append(correct_batch_avg)
        correct_test += out.eq(target.unsqueeze(1).data).sum()
    batch_avg = sum(correct_batch_avg) / len(correct_batch_avg)
    print('\nTest set: Accuracy: {}/{} ({:.0f}%) | Multiple Samples Accuracy{:.0f}\n'\
            .format(correct_test, len(test_loader.dataset),\
                100. * correct_test / len(test_loader.dataset), batch_avg))
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

def L2_test_model(args,epoch,test_loader,model,G,nc=1,h=28,w=28,mode="NotTest"):
    ''' Testing Phase '''
    test_itr = tqdm(enumerate(test_loader),\
            total=len(test_loader.dataset)/args.batch_size)
    correct_test = 0
    # Empty list to hold resampling results. Since we loop batches, results
    # accumulate in appropriate list index, where index is the sampling number
    resample_adv = [[] for i in range(args.resample_iterations)]
    for batch_idx, (data, target) in test_itr:
        x, target = data.to(args.device), target.to(args.device)
        if not args.vanilla_G:
            delta, kl_div  = G(x)
        else:
            delta = G(x)
        delta = delta.view(delta.size(0), nc, h, w)
        adv_inputs = x + delta
        adv_inputs = torch.clamp(adv_inputs, -1.0, 1.0)
        pred = model(adv_inputs.detach())
        out = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct_adv_tensor = out.eq(target.unsqueeze(1).data)
        correct_test += out.eq(target.unsqueeze(1).data).sum()

	# Resample failed examples
        if mode == 'Test' and args.resample_test:
            ipdb.set_trace()
            re_x = x.detach()
            for j in range(args.resample_iterations):
                delta, kl_div = G(re_x)
                adv_inputs = re_x + delta.detach()
		adv_inputs = torch.clamp(adv_inputs, -1.0, 1.0)
		pred = model(adv_inputs.detach())
                out = pred.max(1, keepdim=True)[1] # get the index of the max log-probability

                # From previous correct adv tensor,get indices for correctly pred
                # Since we care about those on which attack failed
                idx = corr_adv_tensor > 0
                # fail_mask = (corr_adv_tensor-1)*(-1)
                # fail_count = fail_mask.sum()
                correct_failed_adv = out.eq(target.unsqueeze(1).data)
                failed_only = correct_failed_adv[idx]
                resample_adv[j].extend(failed_only.cpu().numpy().tolist())

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'\
            .format(correct_test, len(test_loader.dataset),\
                100. * correct_test.cpu().numpy() / len(test_loader.dataset)))

    if args.comet:
        test_acc = 100. * correct_test / len(test_loader.dataset)
        args.experiment.log_metric("Test Adv Accuracy",test_acc,step=epoch)
        if not args.mnist:
            index = np.random.choice(len(x) - 64, 1)[0]
            clean_image = (x)[index:index+64].detach()
            adv_image = (x + delta)[index:index+64].detach()
            delta_image = (delta)[index:index+64].detach()
        else:
            clean_image = (x)[0].detach()
            adv_image = (x + delta)[0].detach()
            delta_image = (delta)[0].detach()
        file_base = "adv_images/" + args.namestr + "/"
        if not os.path.exists(file_base):
            os.makedirs(file_base)
        plot_image_to_comet(args,clean_image,file_base+"clean.png",normalize=True)
        plot_image_to_comet(args,adv_image,file_base+"Adv.png",normalize=True)
        plot_image_to_comet(args,delta_image,file_base+"delta.png",normalize=True)

	# Log resampling stuff
	if mode =='Test' and args.resample_test:
	    cumulative = 0
	    size_test = len(resample_adv[0])
	    for j in range(len(resample_adv)):
		fooled = len(resample_adv[j]) - sum(resample_adv[j])
		percent_fooled = fooled / len(resample_adv[j])
		cumulative += fooled
		cum_per_fooled = cumulative / size_test
		results += '| {:0.2f} |'.format(percent_fooled)
		if args.comet:
		    args.experiment.log_metric("Resampling perc fooled",
							percent_fooled,step=j)
		    args.experiment.log_metric("Resampling perc cumulative fooled",
							cum_per_fooled,step=j)

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
        # PGD_generate_multiple_samples(args,epoch,test_loader,model,G,nc,h,w)
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

def L2_white_box_generator(args, train_loader, test_loader, model, G,\
        nc=1,h=28,w=28):
    epsilon = args.epsilon
    opt = optim.Adam(G.parameters())
    mode = "Train"
    if args.carlini_loss:
        misclassify_loss_func = carlini_wagner_loss
    else:
        misclassify_loss_func = CE_loss_func

    ''' Training Phase '''
    for epoch in range(0,args.attack_epochs):
        train_itr = tqdm(enumerate(train_loader),\
                total=len(train_loader.dataset)/args.batch_size)
        correct = 0
        if epoch == (args.attack_epochs - 1):
            mode = "Test"
        ipdb.set_trace()
        L2_test_model(args,epoch,test_loader,model,G,nc,h,w,mode=mode)
        for batch_idx, (data, target) in train_itr:
            x, target = data.to(args.device), target.to(args.device)
            num_unperturbed = 10
            iter_count = 0
            loss_perturb = 20
            loss_misclassify = 10
            while loss_misclassify > 0 and loss_perturb > 0:
                if not args.vanilla_G:
                    delta, kl_div  = G(x)
                    kl_div = kl_div.sum() / len(x)
                else:
                    delta = G(x)
                delta = delta.view(delta.size(0), nc, h, w)
                adv_inputs = x.detach() + delta
                adv_inputs = torch.clamp(adv_inputs, -1.0, 1.0)
                pred = model(adv_inputs)
                out = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
                loss_misclassify = misclassify_loss_func(args,pred,target)
                if args.inf_loss:
                    loss_perturb = Linf_dist(x,adv_inputs) / len(x)
                else:
                    loss_perturb = L2_dist(x,adv_inputs) / len(x)
                loss = loss_misclassify + args.LAMBDA * loss_perturb + kl_div
                opt.zero_grad()
                loss.backward()
                opt.step()
                iter_count = iter_count + 1
                num_unperturbed = out.eq(target.unsqueeze(1).data).sum()
                if iter_count > args.max_iter:
                    break
            correct += out.eq(target.unsqueeze(1).data).sum()

        if args.comet:
            acc = 100.*correct.cpu().numpy()/len(train_loader.dataset)
            args.experiment.log_metric("Whitebox Total loss",loss,step=epoch)
            args.experiment.log_metric("Whitebox Perturb loss",loss_perturb,step=epoch)
            args.experiment.log_metric("Whitebox Misclassification loss",\
                    loss_misclassify,step=epoch)
            args.experiment.log_metric("Train Adv Accuracy",acc,step=epoch)

        print('\nTrain: Epoch:{} Loss: {:.4f}, Misclassification Loss \
                :{:.4f}, Perturbation Loss {:.4f} Accuracy: {}/{} ({:.0f}%)\n'\
            .format(epoch,\
                loss, loss_misclassify, loss_perturb, correct, len(train_loader.dataset),
                100. * correct.cpu().numpy() / len(train_loader.dataset)))

    return out, delta

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


