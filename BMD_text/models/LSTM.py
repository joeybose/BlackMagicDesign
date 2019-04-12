# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import ipdb
#from memory_profiler import profile

class LSTMClassifier(nn.Module):
    # embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu
    def __init__(self,opt):
        self.opt=opt
        super(LSTMClassifier, self).__init__()
        self.name = "LSTMClassifier"
        self.hidden_dim = opt.hidden_dim
        self.batch_size = opt.batch_size
        self.use_gpu = torch.cuda.is_available()

        self.word_embeddings = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.word_embeddings.weight = nn.Parameter(opt.embeddings,requires_grad=opt.embedding_training)
#        self.word_embeddings.weight.data.copy_(torch.from_numpy(opt.embeddings))
        self.lstm = nn.LSTM(opt.embedding_dim, opt.hidden_dim)
        self.hidden2label = nn.Linear(opt.hidden_dim, opt.label_size)
        self.hidden = self.init_hidden()
        self.mean = opt.__dict__.get("lstm_mean",True)

    def init_hidden(self,batch_size=None):
        if batch_size is None:
            batch_size= self.batch_size

        if self.use_gpu:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1,batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, sentence, return_logits=False,use_embed=False):
        if not use_embed:
            embeds = self.word_embeddings(sentence) #64x200x300
        else:
            embeds = sentence

#        x = embeds.view(sentence.size()[1], self.batch_size, -1)
        x=embeds.permute(1,0,2) #200x64x300
        self.hidden= self.init_hidden(sentence.size()[0]) #1x64x128
        lstm_out, self.hidden = self.lstm(x, self.hidden) #200x64x128
        if self.mean=="mean":
            out = lstm_out.permute(1,0,2)
            final = torch.mean(out,1)
        else:
            final=lstm_out[-1]
        if return_logits:
            logits = self.hidden2label(lstm_out)
            y  = self.hidden2label(final)  #64x3
            return logits, y

        y  = self.hidden2label(final)  #64x3
        return y

class LSTMClassifierEmb(nn.Module):
    """
    An LSTM classifier to be attacked. For end-2-end white box attack, this
    model takes as input embeddings, as opposed to tokens. This way, gradients
    can flow from the classifier to the attacking model. The attack embeddings
    are the original embeddings plus some noise to confuse the model
    """
    # embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu
    def __init__(self,opt):
        self.opt=opt
        super(LSTMClassifierEmb, self).__init__()
        self.name = "LSTMClassifier"
        self.hidden_dim = opt.hidden_dim
        self.batch_size = opt.batch_size
        self.use_gpu = torch.cuda.is_available()

        self.word_embeddings = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.word_embeddings.weight = nn.Parameter(opt.embeddings,requires_grad=opt.embedding_training)
#        self.word_embeddings.weight.data.copy_(torch.from_numpy(opt.embeddings))
        self.lstm = nn.LSTM(opt.embedding_dim, opt.hidden_dim)
        self.hidden2label = nn.Linear(opt.hidden_dim, opt.label_size)
        self.hidden = self.init_hidden()
        self.mean = opt.__dict__.get("lstm_mean",True)

    def init_hidden(self,batch_size=None):
        if batch_size is None:
            batch_size= self.batch_size

        if self.use_gpu:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1,batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, embeds, return_logits=False):
        """
        Args:
            embeds: expect batch X 200 X 300
        """
        # embeds = self.word_embeddings(sentence) #64x200x300

#        x = embeds.view(sentence.size()[1], self.batch_size, -1)
        x=embeds.permute(1,0,2) #200x64x300
        self.hidden= self.init_hidden(sentence.size()[0]) #1x64x128
        lstm_out, self.hidden = self.lstm(x, self.hidden) #200x64x128
        if self.mean=="mean":
            out = lstm_out.permute(1,0,2)
            final = torch.mean(out,1)
        else:
            final=lstm_out[-1]
        if return_logits:
            logits = self.hidden2label(lstm_out)
            y  = self.hidden2label(final)  #64x3
            return logits, y

        y  = self.hidden2label(final)  #64x3
        return y
