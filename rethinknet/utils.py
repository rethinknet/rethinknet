
from os.path import join
from functools import partial
import logging
import copy
import gzip
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MultiLabelBinarizer
import scipy.sparse as ss

from .models.rethinkNet import RethinkNet

label_counts = {
    'eurlex': 3993,
    'amazonCat': 13330,
    'wiki10': 30938,
    'deliciousLarge': 205443,
    'rcv1x': 2456,
    'amazon': 670091,
}

def load_data(filepath):
    with gzip.open(join(filepath, 'data.pkl.gz'), 'r') as f:
        ret = pickle.load(f, encoding='latin1')
    return ret

def load_extreme(filepath, ds_name, split_no):
    if 'mediamill' in ds_name:
        with open(join(filepath, '%s_data.txt' % ds_name), 'rb') as f:
            X, Y = load_svmlight_file(f, multilabel=True)
        with open(join(filepath, '%s_trSplit.txt' % ds_name), 'r') as f:
            trn_ind = np.array([int(line.split()[split_no])-1 for line in f.readlines()])
        with open(join(filepath, '%s_tstSplit.txt' % ds_name), 'r') as f:
            tst_ind = np.array([int(line.split()[split_no])-1 for line in f.readlines()])

        Y = MultiLabelBinarizer(sparse_output=True).fit_transform(Y)
        return X[trn_ind], Y[trn_ind], X[tst_ind], Y[tst_ind]
    else:
        with open(join(filepath, '%s_train.txt' % ds_name), 'rb') as f:
            trnX, trnY = load_svmlight_file(f, multilabel=True)
        with open(join(filepath, '%s_test.txt' % ds_name), 'rb') as f:
            tstX, tstY = load_svmlight_file(f, multilabel=True)
        Y = trnY + tstY
        if ds_name in label_counts:
            labeler = MultiLabelBinarizer(
                            classes=np.arange(label_counts[ds_name]),
                            sparse_output=True)
        else:
            print("no label counts for %s", ds_name)
            labeler = MultiLabelBinarizer(sparse_output=True)
        labeler.fit(Y)
        trnY = labeler.transform(trnY)
        tstY = labeler.transform(tstY)
    return trnX, trnY, tstX, tstY

def load_split(filepath, split_no):
    with gzip.open(join(filepath, 'split_%d.pkl.gz' % split_no), 'r') as f:
        split = pickle.load(f, encoding='latin1')
    return split

def get_model(model_name, model_param):
    if model_name == 'rethinkNet':
        return RethinkNet(**model_param)
    else:
        raise NotImplementedError('no such model %s' % model_name)


def pairwise_hamming(Z, Y):
    """
    Z and Y should be the same size 2-d matrix
    """
    return -np.abs(Z - Y).mean(axis=1)


def pairwise_f1(Z, Y):
    """
    Z and Y should be the same size 2-d matrix
    """
    # calculate F1 by sum(2*y_i*h_i) / (sum(y_i) + sum(h_i))
    Z = Z.astype(int)
    Y = Y.astype(int)
    up = 2*np.sum(Z & Y, axis=1).astype(float)
    down1 = np.sum(Z, axis=1)
    down2 = np.sum(Y, axis=1)

    down = (down1 + down2)
    down[down==0] = 1.
    up[down==0] = 1.

    #return up / (down1 + down2)
    #assert np.all(up / (down1 + down2) == up/down) == True
    return up / down

def pairwise_rankloss(Z, Y): #truth(Z), prediction(Y)
    """
    Z and Y should be the same size 2-d matrix
    """
    rankloss = ((Z==0) & (Y==1)).sum(axis=1) * ((Z==1) & (Y==0)).sum(axis=1)
    tie0 = 0.5 * ((Z==0) & (Y==0)).sum(axis=1) * ((Z==1) & (Y==0)).sum(axis=1)
    tie1 = 0.5 * ((Z==0) & (Y==1)).sum(axis=1) * ((Z==1) & (Y==1)).sum(axis=1)
    return -(rankloss + tie0 + tie1)

def pairwise_acc(Z, Y):
    f1 = 1.0 * ((Z>0) & (Y>0)).sum(axis=1)
    f2 = 1.0 * ((Z>0) | (Y>0)).sum(axis=1)
    f1[f2<=0] = 1.0
    f1[f2>0] /= f2[f2>0]
    return f1

def get_scoring_fn(scoring):
    from .calc_score import (
        pairwise_rankloss_full,
        pairwise_f1_full,
        pairwise_acc_full,
        pairwise_hamming_full,
    )
    if scoring == 'hamming':
        scoring_fn = pairwise_hamming
    elif scoring == 'f1':
        scoring_fn = pairwise_f1
    elif scoring == 'rankloss':
        scoring_fn = pairwise_rankloss
    elif scoring == 'acc':
        scoring_fn = pairwise_acc
    elif scoring == 'sparse_hamming':
        scoring_fn = pairwise_hamming_full
    elif scoring == 'sparse_f1':
        scoring_fn = pairwise_f1_full
    elif scoring == 'sparse_rankloss':
        scoring_fn = pairwise_rankloss_full
    elif scoring == 'sparse_acc':
        scoring_fn = pairwise_acc_full
    else:
        print("err:", [scoring])
    return scoring_fn
