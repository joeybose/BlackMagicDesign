# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import collections
from datetime import datetime
import pickle
import dataHelper
import torch
from torch import nn
import torch.nn.functional as F
import collections
import sklearn
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
from torch.utils.data import Dataset, DataLoader
import numpy as np
from functools import wraps
import torch.optim as optim
import time
import sys
import logging
import os
import models
import ipdb
from dataHelper import*
from tools.nearest import NearestNeighbours

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNKNOWN_IDX = 0

class Vocabulary(object):
    """Indexing for text tokens.


    Build indices for the unknown token, reserved tokens, and input counter keys. Indexed tokens can
    be used by token embeddings.


    Parameters
    ----------
    counter : collections.Counter or None, default None
        Counts text token frequencies in the text data. Its keys will be indexed according to
        frequency thresholds such as `most_freq_count` and `min_freq`. Keys of `counter`,
        `unknown_token`, and values of `reserved_tokens` must be of the same hashable type.
        Examples: str, int, and tuple.
    most_freq_count : None or int, default None
        The maximum possible number of the most frequent tokens in the keys of `counter` that can be
        indexed. Note that this argument does not count any token from `reserved_tokens`. Suppose
        that there are different keys of `counter` whose frequency are the same, if indexing all of
        them will exceed this argument value, such keys will be indexed one by one according to
        their __cmp__() order until the frequency threshold is met. If this argument is None or
        larger than its largest possible value restricted by `counter` and `reserved_tokens`, this
        argument has no effect.
    min_freq : int, default 1
        The minimum frequency required for a token in the keys of `counter` to be indexed.
    unknown_token : hashable object, default '&lt;unk&gt;'
        The representation for any unknown token. In other words, any unknown token will be indexed
        as the same representation. Keys of `counter`, `unknown_token`, and values of
        `reserved_tokens` must be of the same hashable type. Examples: str, int, and tuple.
    reserved_tokens : list of hashable objects or None, default None
        A list of reserved tokens that will always be indexed, such as special symbols representing
        padding, beginning of sentence, and end of sentence. It cannot contain `unknown_token`, or
        duplicate reserved tokens. Keys of `counter`, `unknown_token`, and values of
        `reserved_tokens` must be of the same hashable type. Examples: str, int, and tuple.


    Attributes
    ----------
    unknown_token : hashable object
        The representation for any unknown token. In other words, any unknown token will be indexed
        as the same representation.
    reserved_tokens : list of strs or None
        A list of reserved tokens that will always be indexed.
    """

    def __init__(self, counter=None, most_freq_count=None, min_freq=1, unknown_token='',
                 reserved_tokens=None):

        # Sanity checks.
        assert min_freq > 0, '`min_freq` must be set to a positive value.'

        if reserved_tokens is not None:
            reserved_token_set = set(reserved_tokens)
            assert unknown_token not in reserved_token_set, \
                '`reserved_token` cannot contain `unknown_token`.'
            assert len(reserved_token_set) == len(reserved_tokens), \
                '`reserved_tokens` cannot contain duplicate reserved tokens.'

        self._index_unknown_and_reserved_tokens(unknown_token, reserved_tokens)

        if counter is not None:
            self._index_counter_keys(counter, unknown_token, reserved_tokens, most_freq_count,
                                     min_freq)

    def _index_unknown_and_reserved_tokens(self, unknown_token, reserved_tokens):
        """Indexes unknown and reserved tokens."""

        self._unknown_token = unknown_token
        # Thus, constants.UNKNOWN_IDX must be 0.
        self._idx_to_token = [unknown_token]

        if reserved_tokens is None:
            self._reserved_tokens = None
        else:
            self._reserved_tokens = reserved_tokens[:]
            self._idx_to_token.extend(reserved_tokens)

        self._token_to_idx = {token: idx for idx, token in enumerate(self._idx_to_token)}

    def _index_counter_keys(self, counter, unknown_token, reserved_tokens, most_freq_count,
                            min_freq):
        """Indexes keys of `counter`.


        Indexes keys of `counter` according to frequency thresholds such as `most_freq_count` and
        `min_freq`.
        """

        assert isinstance(counter, collections.Counter), \
            '`counter` must be an instance of collections.Counter.'

        unknown_and_reserved_tokens = set(reserved_tokens) if reserved_tokens is not None else set()
        unknown_and_reserved_tokens.add(unknown_token)

        token_freqs = sorted(counter.items(), key=lambda x: x[0])
        token_freqs.sort(key=lambda x: x[1], reverse=True)

        token_cap = len(unknown_and_reserved_tokens) + (
            len(counter) if most_freq_count is None else most_freq_count)

        for token, freq in token_freqs:
            if freq < min_freq or len(self._idx_to_token) == token_cap:
                break
            if token not in unknown_and_reserved_tokens:
                self._idx_to_token.append(token)
                self._token_to_idx[token] = len(self._idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    @property
    def token_to_idx(self):
        """
        dict mapping str to int: A dict mapping each token to its index integer.
        """
        return self._token_to_idx

    @property
    def idx_to_token(self):
        """
        list of strs:  A list of indexed tokens where the list indices and the token indices are aligned.
        """
        return self._idx_to_token

    @property
    def unknown_token(self):
        return self._unknown_token

    @property
    def reserved_tokens(self):
        return self._reserved_tokens

    def to_indices(self, tokens):
        """Converts tokens to indices according to the vocabulary.


        Parameters
        ----------
        tokens : str or list of strs
            A source token or tokens to be converted.


        Returns
        -------
        int or list of ints
            A token index or a list of token indices according to the vocabulary.
        """

        to_reduce = False
        if not isinstance(tokens, list):
            tokens = [tokens]
            to_reduce = True

        indices = [self.token_to_idx[token] if token in self.token_to_idx
                   else UNKNOWN_IDX for token in tokens]

        return indices[0] if to_reduce else indices


    def to_tokens(self, indices):
        """Converts token indices to tokens according to the vocabulary.


        Parameters
        ----------
        indices : int or list of ints
            A source token index or token indices to be converted.


        Returns
        -------
        str or list of strs
            A token or a list of tokens according to the vocabulary.
        """

        to_reduce = False
        if not isinstance(indices, list):
            indices = [indices]
            to_reduce = True

        max_idx = len(self.idx_to_token) - 1

        tokens = []
        for idx in indices:
            if not isinstance(idx, int) or idx > max_idx:
                raise ValueError('Token index %d in the provided `indices` is invalid.' % idx)
            else:
                tokens.append(self.idx_to_token[idx])

        return tokens[0] if to_reduce else tokens

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

def train_unk_model(args,model,train_itr,test_itr):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-4)
    for j in range(5):
        for i, batch in enumerate(iterator):
            x, y = batch['text'].cuda(), batch['labels'].cuda()
            optimizer.zero_grad()
            predictions = model(x).squeeze(1)
            loss = F.cross_entropy(predictions,y)
            prob, idx = torch.max(F.log_softmax(predictions,dim=1), 1)
            correct = idx.eq(y)
            acc = correct.sum().float() /len(correct)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        print('Train loss: ',epoch_loss / len(iterator),'Train accuracy: ' ,epoch_acc / len(iterator))
    fn = 'saved_models/'+args.model+args.namestr+'.pt'
    torch.save(model.state_dict(), fn)
    return model

def evaluate(model, iterator):
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x, y = batch['text'].cuda(), batch['labels'].cuda()
            predictions = model(x).squeeze(1)
            loss = F.cross_entropy(predictions,y)
            prob, idx = torch.max(F.log_softmax(predictions,dim=1), 1)
            correct = idx.eq(y)
            acc = correct.sum().float()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        print('No Adv Evaluation loss: ',epoch_loss / len(iterator.dataset),'Eval accuracy: ' ,epoch_acc / len(iterator.dataset))

def evaluate_neighbours(iterator, model, G, args, epoch, num_samples=None, mode='Test'):
    """
    Args:
        iterator: (dataloader iterator) to iterate batches
        model: target model
        G: adversarial model
        num_samples: (int) samples evaluated, if None, all samples evaluated
    """
    # Get target model predictions with:
    # - original input
    # - perturbed embeddings
    # - nearest neighbour tokens from perturbed embeddings
    t1 = datetime.now()
    correct_orig = []
    correct_adv_emb = []
    correct_adv_tok = []
    tokens_changed_perc = []
    G.eval()
    model.eval()
    # with torch.no_grad():

    # Nearest neigh function
    nearest = NearestNeighbours(args.embeddings, args.device)
    if str(args.device) == 'cuda' and not args.no_parallel:
        nearest = nn.DataParallel(nearest)

    for i, batch in enumerate(iterator):
        x = batch['text'].to(args.device)
        y = batch['labels'].to(args.device)
        if args.nearest_neigh_all == False:
            last_idx = 3
        else:
            last_idx = len(x)
        x, y = x[:last_idx], y[:last_idx]
        original_pred = model(x).squeeze(1)
        prob_orig, orig_idx = torch.max(F.softmax(original_pred,dim=1), 1)
        correct_orig.extend(orig_idx.eq(y).cpu().numpy().tolist())
        original_tokens, mask = decode_to_token(x, args.inv_alph, args)

        # Get input emb from target model
        input_embeddings = model.get_embeds(x).detach()

        # Attack model perturbation
        # with torch.no_grad():
        delta_embeddings, kl_div = G(x.detach())

        # Adversarial embeddings: combine input and perturb
        adv_embeddings = input_embeddings + delta_embeddings.detach()

        # Target predictions with adversarial embeddings
        adv_emb_preds = model(adv_embeddings,use_embed=True)
        prob_adv_emb, adv_emb_idx = torch.max(F.softmax(adv_emb_preds,dim=1), 1)
        correct_adv_emb.extend(adv_emb_idx.eq(y).cpu().numpy().tolist())

        # Target predictions with adversarial tokens
        adv_x = nearest(adv_embeddings, mask)

        adv_x = adv_x.type(x.dtype) # data type match
        adv_tok_preds = model(adv_x).squeeze(1)
        prob_adv_tok, adv_tok_idx = torch.max(F.softmax(adv_tok_preds,dim=1), 1)
        correct_adv_tok.extend(adv_tok_idx.eq(y).cpu().numpy().tolist())
        # Percent of changed tokens
        changed = 1 - (x.eq(adv_x)).sum().cpu().numpy() / x.numel()
        tokens_changed_perc.append(changed)
        if args.nearest_neigh_all == False:
            break


    accuracies = {
            "acc unperturbed" : sum(correct_orig)/len(correct_orig),
            "acc adv emb": sum(correct_adv_emb)/len(correct_adv_emb),
            "acc adv tok": sum(correct_adv_tok)/len(correct_adv_tok),
            "perc tok changed": sum(tokens_changed_perc)/len(tokens_changed_perc)
            }

    if args.comet:
        if mode == 'Train':
            args.experiment.log_metric("Train acc unperturbed",sum(correct_orig)/len(correct_orig),step=epoch)
            args.experiment.log_metric("Train acc adv emb",sum(correct_adv_emb)/len(correct_adv_emb),step=epoch)
            args.experiment.log_metric("Train acc adv tok",\
                    sum(tokens_changed_perc)/len(tokens_changed_perc),step=epoch)
            args.experiment.log_metric("Train perc tok changed",\
                    sum(tokens_changed_perc)/len(tokens_changed_perc),step=epoch)
        else:
            args.experiment.log_metric("Test acc unperturbed",sum(correct_orig)/len(correct_orig),step=epoch)
            args.experiment.log_metric("Test acc adv emb",sum(correct_adv_emb)/len(correct_adv_emb),step=epoch)
            args.experiment.log_metric("Test acc adv tok",\
                    sum(tokens_changed_perc)/len(tokens_changed_perc),step=epoch)
            args.experiment.log_metric("Test perc tok changed",\
                    sum(tokens_changed_perc)/len(tokens_changed_perc),step=epoch)
    t2 = datetime.now()
    # Save results to string, so easy to print and write to file
    # TODO: add number of flipped tokens
    results = '*'*80 + '\n'
    results += 'Epoch: {}\n'.format(epoch)

    results += '**Result on full' + mode + ' set**\n'
    time_diff = (t2-t1).total_seconds()/60.0
    results += 'Time to evaluate: {:.2f} minutes\n'.format(time_diff)
    results += 'Test set size: {}\n'.format(len(correct_orig))
    for k, v in accuracies.items():
        results += '{:s} : {:0.2f}\n'.format(k,v)

    results += '**Result on single ' + mode + ' set example**\n'
    results += 'Pred. on original input: {:0.2f} for class {:d}, correct class? {:d}\n'.format(\
                                                prob_orig[0], orig_idx[0], correct_orig[0])
    results += 'Pred. on perturbed embed.: {:0.2f} for class {:d}, correct class? {:d}\n'.format(\
                                        prob_adv_emb[0], adv_emb_idx[0], correct_adv_emb[0])
    results += 'Pred. on nearest neighbour tokens: {:0.2f} for class {:d}, correct class? {:d}\n'.format(\
                                        prob_adv_tok[0], adv_tok_idx[0], correct_adv_tok[0])
    results += 'Original example with class {}:\n'.format(y[0])
    original_tokens, mask = decode_to_token(x, args.inv_alph, args)
    results += original_tokens[0] + '\n'

    results += 'Adversarial example with predicted class {}:\n'.format(adv_tok_idx[0])
    # Get nearest neighbour indices/tokens
    nearest_tokens, _ = decode_to_token(adv_x, args.inv_alph, args,mask)
    results += nearest_tokens[0] + '\n'
    results += '*'*80 + '\n'
    print(results)

    # Append samples to file
    if args.save_adv_samples:
        with open(args.sample_file, 'a') as f:
            f.write(results)
        print("====Saved results to file===")
    G.train()
    model.train()
    return results, accuracies

def decode_to_token(x, idx_to_tok, args, mask=None):
    """
    Given a batch of embeddings, returns string tokens. Masked items will be
    removed from output. If no mask is provided, it will be created on-the-fly
    and returned.

    Args:
        x: (Tensor) size [batch, seq length, emb size]
        idx_to_tok: (list) return the string token given index where the index
                    matches to the embedding index
        mask: (Tensor) same shape as `x`
    """
    # Create mask
    if mask is None:
        mask = torch.where(x > 0, torch.tensor([1.]).to(args.device),
                                            torch.tensor([0.]).to(args.device))
    x_tokens = []
    # Loop sequences
    for j in range(x.shape[0]):
        seq_tokens = []
        for i, word in enumerate(x[j]):
            if float(mask[j,i]) > 0:
                seq_tokens.append(idx_to_tok[int(x[j,i])])
        sentence = ' '.join(seq_tokens[:])
        x_tokens.append(sentence)
    return x_tokens, mask

def load_unk_model(args,train_itr,test_itr):
    """
    Load an unknown model. Used for convenience to easily swap unk model
    """
    args.lstm_layers=2
    model=models.setup(args)
    if not args.train_classifier:
        model.load_state_dict(torch.load(args.model_path))
    else:
        model = train_unk_model(args,model,train_itr,test_itr)
    model.to(device)
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

def get_data(args, prepared_data):
    """
    Prepare the data or load some already prepared data such as the embeddings
    """

    # If data already prepared in pickle, load and return
    if os.path.isfile(prepared_data):
        print("Found data pickle, loading from {}".format(prepared_data))
        with open(prepared_data, 'rb') as p:
            d = pickle.load(p)
            args.vocab_size = d["vocab_size"]
            args.label_size = d["label_size"]
            args.embeddings = d["vectors"]
            args.inv_alph = d["inv_alph"]
            args.alphabet = d["wordset"]
            train_iter = d["train_iter"]
            test_iter = d["test_iter"]
        return train_iter, test_iter

    train_data, test_data = read_imdb(data_path), read_imdb(data_path, 'test')
    vocab = get_vocab_imdb(train_data)
    print('vocab_size:', len(vocab))
    args.vocab_size = len(vocab)
    args.label_size = 2
    MAX_LEN = args.max_seq_len
    train_data, train_labels = preprocess_imdb(train_data, vocab, MAX_LEN)
    test_data, test_labels = preprocess_imdb(test_data, vocab, MAX_LEN)
    trainset = IMDBDataset(train_data,train_labels)
    testset = IMDBDataset(test_data,test_labels)
    train_iter = DataLoader(trainset, batch_size=args.batch_size,
			    shuffle=True, num_workers=4)
    test_iter = DataLoader(testset, batch_size=args.test_batch_size,
			    shuffle=True, num_workers=4)

    glove_file = os.path.join( ".vector_cache","glove.6B.300d.txt")
    wordset = vocab._token_to_idx

    loaded_vectors,embedding_size =\
                            dataHelper.load_text_vec(wordset,glove_file)
    vectors = dataHelper.vectors_lookup(loaded_vectors,wordset,300)

    args.embeddings = vectors
    args.inv_alph = vocab._idx_to_token
    args.alphabet = wordset

    # Save prepared data for future fast load
    with open(prepared_data, 'wb') as p:
        d = {}
        d["vocab_size"] = args.vocab_size
        d["label_size"] = args.label_size
        d["vectors"] = args.embeddings
        d["inv_alph"] = args.inv_alph
        d["wordset"] = wordset
        d["train_iter"] = train_iter
        d["test_iter"] = test_iter
        pickle.dump(d, p, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved prepared data for future fast load to: {}".format(\
                                                                prepared_data))

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

