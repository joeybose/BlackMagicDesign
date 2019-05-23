import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import torch.nn.functional as F
import dataHelper
import sys
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torchtext
import torch
import ipdb
import os, sys
import time
import random
from utils import Vocabulary

data_path = '.data/imdb/'
class lstm_arch(torch.nn.Module):
    def __init__(self,embed_dim,vocab_size,hid_dim,out_dim,dropout_prob):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size,embed_dim)   #prepare the lookup table for word embeddings
        self.rnn = torch.nn.LSTM(embed_dim,hid_dim,batch_first=True,bias=True,num_layers=2,bidirectional=True,dropout=dropout_prob)  #LSTM 2 layered and bidirectional
        self.fc = torch.nn.Linear(hid_dim*2,out_dim)          #fully connected layer for output
        self.dropout = torch.nn.Dropout(p = dropout_prob)

    def forward(self,feed_data):
        embed_out = self.dropout(self.embedding(feed_data))
        rnn_out,(rnn_hid,rnn_cell) = self.rnn(embed_out)
        hidden = self.dropout(torch.cat((rnn_hid[-2,:,:], rnn_hid[-1,:,:]), dim=1))
        return self.fc(hidden.squeeze(0))

def read_imdb(data_path=data_path, mode='train'):
    '''
    return: list of [review, label]
    '''
    data = []
    for label in ['pos', 'neg']:
        folder = os.path.join(data_path,'aclImdb', mode, label)
        for file_name in os.listdir(folder):
            with open(os.path.join(folder, file_name), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').strip().lower()
                data.append([review, 1 if label=='pos' else 0])
    random.shuffle(data)
    return data

def tokenize_imdb(data):
    '''
    return: list of [w1, w2,...,]
    '''
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]

    return [tokenizer(review) for review, _ in data]

def get_vocab_imdb(data):
    '''
    return: text.vocab.Vocabulary, each word appears at least 5 times.
    '''
    tokenized = tokenize_imdb(data)
    counter = collections.Counter([tk for st in tokenized for tk in st])
    return Vocabulary(counter, min_freq=5)

def preprocess_imdb(data, vocab, max_len):
    '''
    truncate or pad sentence to max_len
    return: X: list of [list of word index]
            y: list of label
    '''
    def pad(x):
        return x[:max_len] if len(x)>max_len else x+[0]*(max_len-len(x))

    tokenize = tokenize_imdb(data)
    X = np.array([pad(vocab.to_indices(x)) for x in tokenize])
    y = np.array([tag for _, tag in data])

    return X, y


def load_data_imdb(data, vocab, max_len, batch_size, shuffle=True):
    '''
    Create data iterator
    '''
    X, y = preprocess_imdb(data, vocab, max_len)
    dataset = ArrayDataset(X, y)
    data_iter = mxnet_dataloader.DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=4)
    return data_iter

def evaluate(model, iterator):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x, y = batch['text'].cuda(), batch['labels'].cuda()
            predictions = model(x).squeeze(1)
            loss = F.cross_entropy(predictions,y)
            prob, idx = torch.max(F.log_softmax(predictions,dim=1), 1)
            correct = idx.eq(y)
            acc = correct.sum().float() /len(correct)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        print('Evaluation loss: ',epoch_loss / len(iterator),'Eval accuracy: ' ,epoch_acc / len(iterator))

def train_model(model, iterator, optimizer,epoch):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for i, batch in enumerate(iterator):
        x, y = batch['text'].cuda(), batch['labels'].cuda()
        optimizer.zero_grad()
        predictions = model(x).squeeze(1)
        loss = F.cross_entropy(predictions,y)
        prob, idx = torch.max(F.log_softmax(predictions,dim=1), 1)
        correct = idx.eq(y)
        # rounded_predictions = torch.round(torch.sigmoid(predictions))
        # correct = (rounded_predictions == batch.label.float()).float()
        acc = correct.sum().float() /len(correct)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    print('Train loss: ',epoch_loss / len(iterator),'Train accuracy: ' ,epoch_acc / len(iterator))

class IMDBDataset(Dataset):
    """IMDB dataset."""
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        label = self.labels[idx]
        sample = {'text': x, 'labels': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == '__main__':
    train_data, test_data = read_imdb(data_path), read_imdb(data_path, 'test')
    batch_size = 64

    vocab = get_vocab_imdb(train_data)
    print('vocab_size:', len(vocab))
    MAX_LEN = 200
    train_data, train_labels = preprocess_imdb(train_data, vocab, MAX_LEN)
    test_data, test_labels = preprocess_imdb(test_data, vocab, MAX_LEN)
    trainset = IMDBDataset(train_data,train_labels)
    testset = IMDBDataset(test_data,test_labels)
    train_iter = DataLoader(trainset, batch_size=batch_size,
			    shuffle=True, num_workers=4)
    test_iter = DataLoader(testset, batch_size=batch_size,
			    shuffle=True, num_workers=4)

    fn = 'lstm_torchtext2.pt'
    model = lstm_arch(vocab_size=len(vocab),embed_dim=300,hid_dim=128,out_dim=2,dropout_prob=0.5)
    model.load_state_dict(torch.load(fn))
    model.cuda()
    ipdb.set_trace()
    evaluate(model,test_iter)
    glove_file = os.path.join( ".vector_cache","glove.6B.300d.txt")
    wordset = vocab._token_to_idx
    loaded_vectors,embedding_size = dataHelper.load_text_vec(wordset,glove_file)
    vectors = dataHelper.vectors_lookup(loaded_vectors,wordset,300)
    model.embedding.weight.data.copy_(torch.Tensor(vectors))
    model = model.cuda()
    # optimizer = torch.optim.Adam(model.parameters(),betas=(0.7,0.995),lr=0.005)  #optimizer for our model used adam here
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-4)
    for i in range(5):
        print('epoch %d :'%(i+1))
        train_model(model,train_iter,optimizer,1)
        evaluate(model,test_iter)
        print('\n')
    torch.save(model.state_dict(), fn)
