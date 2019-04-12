# -*- coding: utf-8 -*-
import torch
import dataHelper
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
import numpy as np
from functools import wraps
import torch.optim as optim
import time
import sys
import logging
import os
import models
import ipdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
    return var

def reduce_sum(x, keepdim=True):
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    return x.squeeze().sum()

def L2_dist(x, y):
    return reduce_sum((x - y)**2)

def decode_to_natural_lang(sequence,args):
    sentence_list = []
    for token in sequence:
        word = args.inv_alph.get(token.item())
        sentence_list.append(word)
    sentence = ' '.join(sentence_list)
    print(sentence)
    return sentence

def evaluation(model,test_iter,from_torchtext=True):
    model.eval()
    accuracy=[]
#    batch= next(iter(test_iter))
    correct_test = 0
    for index,batch in enumerate( test_iter):
        text = batch.text[0] if from_torchtext else batch.text
        predicted = model(text)
        prob, idx = torch.max(F.log_softmax(predicted,dim=1), 1)
        percision=(idx== batch.label).float().mean()
        correct_test += idx.eq(batch.label.data).sum()

        if torch.cuda.is_available():
            accuracy.append(percision.data.item() )
        else:
            accuracy.append(percision.data.numpy()[0] )
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'\
            .format(correct_test, 25000,\
                100. * correct_test / 25000))
    model.train()
    return np.mean(accuracy)

def train_unk_model(args,model,train_itr,test_itr):
    loss_fun = F.cross_entropy
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=2e-5, weight_decay=1e-3)
    optimizer.zero_grad()
    for i in range(10):
        accuracy = []
        from_torchtext = False
        for epoch,batch in enumerate(train_itr):
            start= time.time()

            text = batch.text[0] if from_torchtext else batch.text
            predicted = model(text)

            loss= loss_fun(predicted,batch.label)

            # optimizer.zero_grad()
            loss.backward()
            clip_gradient(optimizer, 1e-1)
            prob, idx = torch.max(F.log_softmax(predicted,dim=1), 1)
            percision=(idx== batch.label).float().mean()

            if torch.cuda.is_available():
                accuracy.append(percision.data.item() )
            else:
                accuracy.append(percision.data.numpy()[0] )
            optimizer.step()
            if epoch% 1000==0:
                if  torch.cuda.is_available():
                    print("%d iteration %d epoch with loss : %.5f in %.4f seconds" % (i,epoch,loss.cpu().item(),time.time()-start))
                else:
                    print("%d iteration %d epoch with loss : %.5f in %.4f seconds" % (i,epoch,loss.data.numpy()[0],time.time()-start))

        percision=evaluation(model,test_itr,False)
        train_acc =  np.mean(accuracy)
        print("%d iteration with Train Acc %.4f" % (i,train_acc))
        print("%d iteration with Test Acc %.4f" % (i,percision))
        fn = args.model+args.namestr+'.pt'
        torch.save(model.state_dict(), fn)
    return model

def load_unk_model(args,train_itr,test_itr):
    """
    Load an unknown model. Used for convenience to easily swap unk model
    """
    args.lstm_layers=2
    model=models.setup(args)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    # model = train_unk_model(args,model,train_itr,test_itr)
    # model.eval()
    return model

def get_cumulative_rewards(disc_logits, orig_targets, args, is_already_reward=False):
    adv_targets = (orig_targets == 0).long() # Flips the bit
    # disc_logits : bs x seq_len x classes
    disc_logits = disc_logits.permute(1,0,2)
    disc_logits = F.softmax(disc_logits,dim=2)
    logit = []
    for i in range(0,args.batch_size):
        logit.append(torch.index_select(disc_logits[i],1,orig_targets[i]))

    adv_logits = torch.cat(logit,1).t()
    assert len(adv_logits.size()) == 2
    if is_already_reward:
        rewards = adv_logits
    else:
        rewards = F.sigmoid(adv_logits + 1e-7)
        rewards = torch.log(rewards + 1e-7)

    bs, seq_len = rewards.size()
    cumulative_rewards = torch.zeros_like(rewards)
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            cumulative_rewards[:, t] = rewards[:, t]
            # if in SEQGAN mode, make sure reward only comes from the last timestep
            if args.seqgan_reward: rewards = rewards * 0.
        else:
            cumulative_rewards[:, t] = rewards[:, t] + args.gamma * cumulative_rewards[:, t+1].clone()

    return cumulative_rewards

def get_data(args):
    # if from_torchtext:
        # train_iter, test_iter = utils.loadData(opt)
    # else:
    import dataHelper as helper
    train_iter, test_iter = dataHelper.loadData(args)
    return train_iter, test_iter

def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print( "%s runed %.2f seconds"% (func.__name__,delta))
        return ret
    return _deco

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None and param.requires_grad:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def loadData(opt):
    if not opt.from_torchtext:
        import dataHelper as helper
        return helper.loadData(opt)
    device = 0 if  torch.cuda.is_available()  else -1

    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True,fix_length=opt.max_seq_len)
    LABEL = data.Field(sequential=False)
    if opt.dataset=="imdb":
        train, test = datasets.IMDB.splits(TEXT, LABEL)
    elif opt.dataset=="sst":
        train, val, test = datasets.SST.splits( TEXT, LABEL, fine_grained=True, train_subtrees=True,
                                               filter_pred=lambda ex: ex.label != 'neutral')
    elif opt.dataset=="trec":
        train, test = datasets.TREC.splits(TEXT, LABEL, fine_grained=True)
    else:
        print("does not support this datset")

    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train)
    # print vocab information
    print('len(TEXT.vocab)', len(TEXT.vocab))
    print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

    train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=opt.batch_size,device=device,repeat=False,shuffle=True)

    opt.label_size= len(LABEL.vocab)
    opt.vocab_size = len(TEXT.vocab)
    opt.embedding_dim= TEXT.vocab.vectors.size()[1]
    opt.embeddings = TEXT.vocab.vectors

    return train_iter, test_iter


def evaluation(model,test_iter,from_torchtext=True):
    model.eval()
    accuracy=[]
#    batch= next(iter(test_iter))
    for index,batch in enumerate( test_iter):
        text = batch.text[0] if from_torchtext else batch.text
        predicted = model(text)
        prob, idx = torch.max(predicted, 1)
        percision=(idx== batch.label).float().mean()

        if torch.cuda.is_available():
            accuracy.append(percision.data.item() )
        else:
            accuracy.append(percision.data.numpy()[0] )
    model.train()
    return np.mean(accuracy)



def getOptimizer(params,name="adam",lr=1,momentum=None,scheduler=None):

    name = name.lower().strip()

    if name=="adadelta":
        optimizer=torch.optim.Adadelta(params, lr=1.0*lr, rho=0.9, eps=1e-06, weight_decay=0).param_groups()
    elif name == "adagrad":
        optimizer=torch.optim.Adagrad(params, lr=0.01*lr, lr_decay=0, weight_decay=0)
    elif name == "sparseadam":
        optimizer=torch.optim.SparseAdam(params, lr=0.001*lr, betas=(0.9, 0.999), eps=1e-08)
    elif name =="adamax":
        optimizer=torch.optim.Adamax(params, lr=0.002*lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    elif name =="asgd":
        optimizer=torch.optim.ASGD(params, lr=0.01*lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    elif name == "lbfgs":
        optimizer=torch.optim.LBFGS(params, lr=1*lr, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
    elif name == "rmsprop":
        optimizer=torch.optim.RMSprop(params, lr=0.01*lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    elif name =="rprop":
        optimizer=torch.optim.Rprop(params, lr=0.01*lr, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
    elif name =="sgd":
        optimizer=torch.optim.SGD(params, lr=0.1*lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    elif name =="adam":
         optimizer=torch.optim.Adam(params, lr=0.1*lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    else:
        print("undefined optimizer, use adam in default")
        optimizer=torch.optim.Adam(params, lr=0.1*lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    if scheduler is not None:
        if scheduler == "lambdalr":
            lambda1 = lambda epoch: epoch // 30
            lambda2 = lambda epoch: 0.95 ** epoch
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        elif scheduler=="steplr":
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif scheduler =="multisteplr":
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        elif scheduler =="reducelronplateau":
            return  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        else:
            pass

    else:
        return optimizer


    return
def getLogger():
    import random
    random_str = str(random.randint(1,10000))

    now = int(time.time())
    timeArray = time.localtime(now)
    timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
    log_filename = "log/" +time.strftime("%Y%m%d", timeArray)

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    if not os.path.exists("log"):
        os.mkdir("log")
    if not os.path.exists(log_filename):
        os.mkdir(log_filename)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',datefmt='%a, %d %b %Y %H:%M:%S',filename=log_filename+'/qa'+timeStamp+"_"+ random_str+'.log',filemode='w')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    return logger
def is_writeable(path, check_parent=False):
    '''
    Check if a given path is writeable by the current user.
    :param path: The path to check
    :param check_parent: If the path to check does not exist, check for the
    ability to write to the parent directory instead
    :returns: True or False
    '''
    if os.access(path, os.F_OK) and os.access(path, os.W_OK):
    # The path exists and is writeable
        return True
    if os.access(path, os.F_OK) and not os.access(path, os.W_OK):
    # The path exists and is not writeable
        return False
    # The path does not exists or is not writeable
    if check_parent is False:
    # We're not allowed to check the parent directory of the provided path
        return False
    # Lets get the parent directory of the provided path
    parent_dir = os.path.dirname(path)
    if not os.access(parent_dir, os.F_OK):
    # Parent directory does not exit
        return False
    # Finally, return if we're allowed to write in the parent directory of the
    # provided path
    return os.access(parent_dir, os.W_OK)

def is_readable(path):
    '''
    Check if a given path is readable by the current user.
    :param path: The path to check
    :returns: True or False
    '''
    if os.access(path, os.F_OK) and os.access(path, os.R_OK):
    # The path exists and is readable
        return True
    # The path does not exist
    return False

