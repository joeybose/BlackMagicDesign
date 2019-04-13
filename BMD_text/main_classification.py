# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

from six.moves import cPickle

import opts
import models
import torch.nn as nn
import utils
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
from torch.nn.modules.loss import NLLLoss,MultiLabelSoftMarginLoss,MultiLabelMarginLoss,BCELoss
import dataHelper
import time,os
import ipdb


from_torchtext = False

opt = opts.parse_opt()
#opt.proxy="http://xxxx.xxxx.com:8080"


if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
    os.environ["CUDA_VISIBLE_DEVICES"] =opt.gpu
#opt.model ='lstm'
#opt.model ='capsule'
ipdb.set_trace()
if from_torchtext:
    train_iter, test_iter = utils.loadData(opt)
else:
    import dataHelper as helper
    train_iter, test_iter = dataHelper.loadData(opt)

opt.lstm_layers=2
print('Print loading models')
model2=models.setup(opt)
model2.load_state_dict(torch.load('saved_models/lstm_test.pt'))
model2.cuda()
percision=utils.evaluation(model2,test_iter,from_torchtext)
print("After iteration with model 2 Test Acc %.4f" % (percision))
ipdb.set_trace()
model=models.setup(opt)
# model.load_state_dict(torch.load('lstm_new.pt'))
if torch.cuda.is_available():
    model.cuda()
model.train()
print("# parameters:", sum(param.numel() for param in model.parameters() if param.requires_grad))
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=opt.learning_rate, weight_decay=1e-3)
optimizer.zero_grad()
loss_fun = F.cross_entropy

#batch = next(iter(train_iter))

#x=batch.text[0]

#x=batch.text[0] #64x200
# percision=utils.evaluation(model,test_iter,from_torchtext)
# print("Before iteration with Test Acc %.4f" % (percision))
for i in range(opt.max_epoch):
    accuracy = []
    for epoch,batch in enumerate(train_iter):
        start= time.time()

        text = batch.text[0] if from_torchtext else batch.text
        predicted = model(text)

        loss= loss_fun(predicted,batch.label)

        # optimizer.zero_grad()
        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
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

    percision=utils.evaluation(model,test_iter,from_torchtext)
    train_acc =  np.mean(accuracy)
    print("%d iteration with Train Acc %.4f" % (i,train_acc))
    print("%d iteration with Test Acc %.4f" % (i,percision))
    fn = opt.model+opt.namestr+'.pt'
    torch.save(model.state_dict(), fn)

# print('Print loading models')
# model2=models.setup(opt)
# model2.load_state_dict(torch.load('lstm_new2.pt'))
# model2.cuda()
# percision=utils.evaluation(model2,test_iter,from_torchtext)
# print("After iteration with model 2 Test Acc %.4f" % (percision))
# percision=utils.evaluation(model,test_iter,from_torchtext)
ipdb.set_trace()
# print("After iteration with model Test Acc %.4f" % (percision))
