from os.path import join
import threading
import itertools

from mkdir_p import mkdir_p
import numpy as np
from keras.layers import (
    Input,
    Dense,
    Dropout,
    LSTM,
    GRU,
    SimpleRNN,
    Activation,
    RepeatVector,
)
from keras.initializers import Identity
from keras.layers.merge import Concatenate, Add
from keras.layers.wrappers import TimeDistributed
from keras.regularizers import l2, l1
from keras.models import Model
from keras.optimizers import SGD, Nadam, RMSprop, Adadelta, Adam
from keras.constraints import Constraint
from keras import backend as K
import scipy.sparse as ss

import tensorflow as tf

from sklearn.model_selection import train_test_split
from bistiming import IterTimer, SimpleTimer

from .utils import get_random_state, weighted_binary_crossentropy, \
    get_rnn_unit, w_bin_xentropy
from ..calc_score import (
    reweight_pairwise_f1,
    reweight_pairwise_hamming,
    p_reweight_pairwise_f1,
    sparse_reweight_pairwise_f1,
    sparse_reweight_pairwise_rankloss,
    sparse_reweight_pairwise_acc,
    reweight_pairwise_rankloss,
)

def arch_001(input_shape, n_labels, weight_input_shape, l2w=1e-5, rnn_unit='lstm'):
    if l2w is None:
        regularizer = None
    else:
        regularizer = l2(l2w)

    inputs = Input(shape=input_shape[1:])
    x = RepeatVector(input_shape[0])(inputs)

    x = Dense(128, kernel_regularizer=regularizer, activation='relu')(x)

    x = get_rnn_unit(rnn_unit, 128, x, activation='sigmoid', l2w=regularizer,
                     recurrent_dropout=0.25)
    outputs = Dense(n_labels, activation='sigmoid')(x)

    weight_input = Input(shape=weight_input_shape)

    return Model(inputs=[inputs, weight_input], outputs=[outputs]), weight_input

def arch_006(input_shape, n_labels, weight_input_shape, l2w, rnn_unit='lstm'):
    if l2w is None:
        regularizer = None
    else:
        regularizer = l2(l2w)

    inputs = Input(shape=input_shape[1:])
    x = RepeatVector(input_shape[0])(inputs)

    x = get_rnn_unit(rnn_unit, 64, x, activation='sigmoid', l2w=regularizer,
                     recurrent_dropout=0.25)
    outputs = Dense(n_labels, activation='sigmoid')(x)

    weight_input = Input(shape=weight_input_shape)

    return Model(input=[inputs, weight_input], output=[outputs]), weight_input

def arch_007(input_shape, n_labels, weight_input_shape, l2w, rnn_unit='lstm'):
    regularizer = l2(l2w) if l2w is not None else None

    inputs = Input(shape=input_shape[1:])
    x = inputs

    w = Dense(128, kernel_regularizer=regularizer, activation='linear')(x)
    out = Dense(n_labels, activation='sigmoid')

    xs = []
    for i in range(input_shape[0]):
        if i > 0:
            u = Dense(128, kernel_regularizer=regularizer, use_bias=False,
                      activation='linear')(xs[-1])
            #u = Dropout(0.25)(u)
            x = Add()([w, u])
        else:
            x = w
        x = Activation('sigmoid')(x)
        xs.append(x)

    for i in range(input_shape[0]):
        xs[i] = out(xs[i])
        xs[i] = RepeatVector(1)(xs[i])

    if len(xs) > 1:
        outputs = Concatenate(axis=1)(xs)
    else:
        outputs = xs[0]

    weight_input = Input(shape=weight_input_shape)

    return Model(input=[inputs, weight_input], output=[outputs]), weight_input

def arch_008(input_shape, n_labels, weight_input_shape, l2w, rnn_unit='lstm'):
    regularizer = l2(l2w) if l2w is not None else None

    inputs = Input(shape=input_shape[1:])
    x = inputs

    w = Dense(128, kernel_regularizer=regularizer, activation='linear')(x)
    out = Dense(n_labels, activation='sigmoid')

    xs = []
    for i in range(input_shape[0]):
        if i > 0:
            u = Dense(128, kernel_regularizer=regularizer, use_bias=False,
                      activation='linear')(xs[-1])
            u = Dropout(0.25)(u)
            x = Add()([w, u])
        else:
            x = w
        x = Activation('sigmoid')(x)
        xs.append(x)

    for i in range(input_shape[0]):
        xs[i] = out(xs[i])
        xs[i] = RepeatVector(1)(xs[i])

    if len(xs) > 1:
        outputs = Concatenate(axis=1)(xs)
    else:
        outputs = xs[0]

    weight_input = Input(shape=weight_input_shape)

    return Model(input=[inputs, weight_input], output=[outputs]), weight_input

def arch_009(input_shape, n_labels, weight_input_shape, l2w, rnn_unit='lstm'):
    regularizer = l2(l2w) if l2w is not None else None

    inputs = Input(shape=input_shape[1:])
    x = inputs

    w = Dense(256, kernel_regularizer=regularizer, activation='linear')(x)
    out = Dense(n_labels, activation='sigmoid')

    xs = []
    for i in range(input_shape[0]):
        if i > 0:
            u = Dense(256, kernel_regularizer=regularizer, use_bias=False,
                      activation='linear')(xs[-1])
            u = Dropout(0.25)(u)
            x = Add()([w, u])
        else:
            x = w
        x = Activation('sigmoid')(x)
        xs.append(x)

    for i in range(input_shape[0]):
        xs[i] = out(xs[i])
        xs[i] = RepeatVector(1)(xs[i])

    if len(xs) > 1:
        outputs = Concatenate(axis=1)(xs)
    else:
        outputs = xs[0]

    weight_input = Input(shape=weight_input_shape)

    return Model(input=[inputs, weight_input], output=[outputs]), weight_input

def arch_010(input_shape, n_labels, weight_input_shape, l2w, rnn_unit='lstm'):
    if l2w is None:
        regularizer = None
    else:
        regularizer = l2(l2w)

    inputs = Input(shape=input_shape[1:])
    x = RepeatVector(input_shape[0])(inputs)

    x = get_rnn_unit(rnn_unit, 96, x, activation='sigmoid', l2w=regularizer,
                     recurrent_dropout=0.25)
    outputs = Dense(n_labels, activation='sigmoid')(x)

    weight_input = Input(shape=weight_input_shape)

    return Model(input=[inputs, weight_input], output=[outputs]), weight_input

def arch_011(input_shape, n_labels, weight_input_shape, l2w, rnn_unit='lstm'):
    regularizer = l2(l2w) if l2w is not None else None

    inputs = Input(shape=input_shape[1:])
    x = inputs

    w = Dense(192, kernel_regularizer=regularizer, activation='linear')(x)
    out = Dense(n_labels, activation='sigmoid')

    xs = []
    for i in range(input_shape[0]):
        if i > 0:
            u = Dense(192, kernel_regularizer=regularizer, use_bias=False,
                      activation='linear')(xs[-1])
            u = Dropout(0.25)(u)
            x = Add()([w, u])
        else:
            x = w
        x = Activation('sigmoid')(x)
        xs.append(x)

    for i in range(input_shape[0]):
        xs[i] = out(xs[i])
        xs[i] = RepeatVector(1)(xs[i])

    if len(xs) > 1:
        outputs = Concatenate(axis=1)(xs)
    else:
        outputs = xs[0]

    weight_input = Input(shape=weight_input_shape)

    return Model(input=[inputs, weight_input], output=[outputs]), weight_input

def arch_012(input_shape, n_labels, weight_input_shape, l2w, rnn_unit='lstm'):
    if l2w is None:
        regularizer = None
    else:
        regularizer = l2(l2w)

    n_features = input_shape[1]
    if rnn_unit == 'lstm':
        m = np.floor(np.roots([4, 4*(n_features+1) + n_labels, n_labels-200000])[1])
    elif rnn_unit == 'gru':
        m = np.floor(np.roots([3, 3*(n_features+1) + n_labels, n_labels-200000])[1])
    elif rnn_unit == 'simplernn':
        m = np.floor(np.roots([1, 1*(n_features+1) + n_labels, n_labels-200000])[1])
        #print(m*m+ m*((n_features+1) + n_labels) + n_labels)
    mem_units = int(m)

    inputs = Input(shape=input_shape[1:])
    x = RepeatVector(input_shape[0])(inputs)

    x = get_rnn_unit(rnn_unit, mem_units, x, activation='sigmoid', l2w=regularizer,
                     recurrent_dropout=0.25)
    outputs = Dense(n_labels, activation='sigmoid')(x)

    weight_input = Input(shape=weight_input_shape)

    return Model(input=[inputs, weight_input], output=[outputs]), weight_input

def arch_013(input_shape, n_labels, weight_input_shape, l2w, rnn_unit='lstm'):
    regularizer = l2(l2w) if l2w is not None else None

    b = input_shape[0]
    n_features = input_shape[1]
    m = np.floor(np.roots([b-1, (n_features+1) + n_labels, n_labels-200000])[1])
    #print((b-1)*m*m+ m*((n_features+1) + n_labels) + n_labels)
    mem_units = int(m)

    inputs = Input(shape=input_shape[1:])
    x = inputs

    w = Dense(mem_units, kernel_regularizer=regularizer, activation='linear')(x)
    out = Dense(n_labels, activation='sigmoid')

    xs = []
    for i in range(input_shape[0]):
        if i > 0:
            u = Dense(mem_units, kernel_regularizer=regularizer, use_bias=False,
                      activation='linear')(xs[-1])
            u = Dropout(0.25)(u)
            x = Add()([w, u])
        else:
            x = w
        x = Activation('sigmoid')(x)
        xs.append(x)

    for i in range(input_shape[0]):
        xs[i] = out(xs[i])
        xs[i] = RepeatVector(1)(xs[i])

    if len(xs) > 1:
        outputs = Concatenate(axis=1)(xs)
    else:
        outputs = xs[0]

    weight_input = Input(shape=weight_input_shape)

    return Model(input=[inputs, weight_input], output=[outputs]), weight_input

def arch_014(input_shape, n_labels, weight_input_shape, l2w, rnn_unit='lstm'):
    if l2w is None:
        regularizer = None
    else:
        regularizer = l2(l2w)

    inputs = Input(shape=input_shape[1:])
    x = RepeatVector(input_shape[0])(inputs)

    x = get_rnn_unit(rnn_unit, 64, x, activation='tanh', l2w=regularizer,
                     recurrent_dropout=0.25)
    outputs = Dense(n_labels, activation='sigmoid')(x)

    weight_input = Input(shape=weight_input_shape)

    return Model(input=[inputs, weight_input], output=[outputs]), weight_input

def arch_015(input_shape, n_labels, weight_input_shape, l2w, rnn_unit='lstm'):
    if l2w is None:
        regularizer = None
    else:
        regularizer = l2(l2w)

    inputs = Input(shape=input_shape[1:])
    x = RepeatVector(input_shape[0])(inputs)

    x = Dense(128, kernel_regularizer=regularizer, activation='relu')(x)

    x = get_rnn_unit(rnn_unit, 128, x, activation='sigmoid', l2w=regularizer,
                     recurrent_dropout=0.25)
    outputs = Dense(n_labels, activation='sigmoid')(x)

    weight_input = Input(shape=weight_input_shape)

    return Model(input=[inputs, weight_input], output=[outputs]), weight_input

def arch_016(input_shape, n_labels, weight_input_shape, l2w, rnn_unit='simplernn'):
    #IRNN
    if l2w is None:
        regularizer = None
    else:
        regularizer = l2(l2w)

    inputs = Input(shape=input_shape[1:])
    x = RepeatVector(input_shape[0])(inputs)

    x = get_rnn_unit(rnn_unit, 128, x, activation='relu', l2w=regularizer,
                     recurrent_dropout=0.25,
                     recurrent_initializer=Identity(1.))
    outputs = Dense(n_labels, activation='sigmoid')(x)

    weight_input = Input(shape=weight_input_shape)

    return Model(input=[inputs, weight_input], output=[outputs]), weight_input

def irnn_same_param(input_shape, n_labels, weight_input_shape, l2w, rnn_unit='simplernn'):
    #IRNN
    if l2w is None:
        regularizer = None
    else:
        regularizer = l2(l2w)

    n_features = input_shape[1]
    if rnn_unit == 'lstm':
        m = np.floor(np.roots([4, 4*(n_features+1) + n_labels, n_labels-200000])[1])
    elif rnn_unit == 'gru':
        m = np.floor(np.roots([3, 3*(n_features+1) + n_labels, n_labels-200000])[1])
    elif rnn_unit == 'simplernn':
        m = np.floor(np.roots([1, 1*(n_features+1) + n_labels, n_labels-200000])[1])
    mem_units = int(m)

    inputs = Input(shape=input_shape[1:])
    x = RepeatVector(input_shape[0])(inputs)

    x = get_rnn_unit(rnn_unit, mem_units, x, activation='relu', l2w=regularizer,
                     recurrent_dropout=0.25,
                     recurrent_initializer=Identity(1.))
    outputs = Dense(n_labels, activation='sigmoid')(x)

    weight_input = Input(shape=weight_input_shape)

    return Model(input=[inputs, weight_input], output=[outputs]), weight_input


class RethinkNet(object):

    def __init__(self, n_features, n_labels, scoring_fn, learning_rate=0.01,
            architecture="arch_001", b=3, batch_size=256, nb_epoches=1000,
            reweight='hw', rnn_unit='lstm', random_state=None, l2w=1e-5,
            tst_ds=None, decay=0., gamma=None, opt='nadam',
            **kwargs):
        self.random_state = get_random_state(random_state)
        self.batch_size = batch_size
        self.b = b
        self.scoring_fn = scoring_fn
        self.gamma = gamma

        if reweight in ['balanced', 'None']:
            self.reweight_scoring_fn = None
        elif reweight in ['hw', 'vw']:
            if 'pairwise_hamming' in self.scoring_fn.__str__():
                self.reweight_scoring_fn = reweight_pairwise_hamming
            elif 'pairwise_rankloss' in self.scoring_fn.__str__():
                self.reweight_scoring_fn = sparse_reweight_pairwise_rankloss
            elif 'pairwise_acc' in self.scoring_fn.__str__():
                self.reweight_scoring_fn = sparse_reweight_pairwise_acc
            elif 'pairwise_f1' in self.scoring_fn.__str__():
                self.reweight_scoring_fn = sparse_reweight_pairwise_f1

        self.tst_ds = tst_ds
        if tst_ds is not None:
            tstX, tstY = self.tst_ds
            tstX = ss.csr_matrix(tstX).astype('float32')
            tstY = ss.csr_matrix(tstY).astype(np.int8)
            self.tst_ds = tstX, tstY

        self.nb_epoches = nb_epoches
        self.reweight = reweight
        self.l2w = l2w

        self.n_labels = n_labels
        self.n_features = n_features
        self.input_shape = ((self.b, ) + (n_features, ))
        self.weight_input_shape = ((self.b, self.n_labels, ))
        model, weight_input = \
                globals()[architecture](self.input_shape, self.n_labels,
                        self.weight_input_shape, self.l2w, rnn_unit)
        model.summary()
        self.nb_params = int(model.count_params())

        print(opt)
        if opt is None:
            optimizer = Adam(lr=learning_rate, decay=decay)
        elif opt == 'nadam':
            optimizer = Nadam()

        self.loss = weighted_binary_crossentropy(weight_input)

        #model.compile(loss='binary_crossentropy', optimizer=optimizer)
        model.compile(loss=self.loss, optimizer=optimizer)
        self.model = model

        self.weight_dir = None

        print(decay, vars(self), architecture, rnn_unit)


    def _prep_X(self, X):
        X = X.toarray()
        return X

    def _prep_Y(self, Y):
        Y = Y.toarray()
        Y = np.repeat(Y[:, np.newaxis, :], self.b, axis=1)
        return Y

    def _prep_weight(self, trn_pred, trnY):
        weight = np.ones((trnY.shape[0], self.b, self.n_labels),
                         dtype='float32')
        i_start = 1
        if 'vw' in self.reweight:
            i_start = 0
        for i in range(i_start, self.b):
            if self.reweight == 'balanced':
                weight[:, i, :] = trnY.astype('float32') * (
                        1. / self.ones_weight - 1.)
                weight[:, i, :] += 1.
            elif self.reweight == 'None':
                pass
            elif self.reweight_scoring_fn in [
                    sparse_reweight_pairwise_acc,
                    sparse_reweight_pairwise_f1,
                    sparse_reweight_pairwise_rankloss]:
                trn_pre = trn_pred[i-1]
                if 'vw' in self.reweight:
                    trn_pre = trn_pred[i]
                weight[:, i, :]  = self.reweight_scoring_fn(
                                        trnY, trn_pre,
                                        use_true=('truth' in self.reweight))
            elif self.reweight_scoring_fn is not None:
                trn_pre = trn_pred[i-1]
                if 'vw' in self.reweight:
                    trn_pre = trn_pred[i]
                w = self.reweight_scoring_fn(
                        trnY,
                        trn_pre.toarray(),
                        use_true=('truth' in self.reweight))
                weight[:, i, :] = np.abs(w[:, :, 0] - w[:, :, 1])
            else:
                raise NotImplementedError()
            weight[:, i, :] *= weight[:, i, :].size / weight[:, i, :].sum()

        if self.gamma is not None:
            for i in range(self.b):
                weight[:, i, :] *= self.gamma**(self.b - i - 1)

        return weight.astype('float32')

    def fit(self, X, Y, with_validation=False, callbacks=[]):
        self.history = []
        nb_epoches = self.nb_epoches
        X = ss.csr_matrix(X).astype('float32')
        Y = ss.csr_matrix(Y).astype(np.int8)

        if with_validation:
            test_size = 0.33
            trnX, valX, trnY, valY = train_test_split(
                X, Y, test_size=test_size, random_state=self.random_state)
        else:
            trnX, trnY = X, Y

        if self.reweight == 'balanced':
            self.ones_weight = trnY.astype(np.int32).sum() / \
                               trnY.shape[0] / trnY.shape[1]

        trn_pred = []
        for i in range(self.b):
            trn_pred.append(
                ss.csr_matrix((trnX.shape[0], self.n_labels), dtype=np.int8))

        print(nb_epoches)
        predict_period = 10
        for epoch_i in range(0, nb_epoches, predict_period):
            input_generator = InputGenerator(
                self, trnX, trnY, trn_pred, shuffle=False,
                batch_size=self.batch_size, random_state=self.random_state)
            #input_generator.next()
            print(trnX.shape[0], self.batch_size)
            print(((trnX.shape[0] - 1) // self.batch_size) + 1)
            history = self.model.fit_generator(
                input_generator,
                steps_per_epoch=((trnX.shape[0] - 1) // self.batch_size) + 1,
                epochs=epoch_i + predict_period,
                max_q_size=32,
                workers=8,
                pickle_safe=True,
                initial_epoch=epoch_i,
                callbacks=callbacks)

            trn_scores = []
            val_scores = []
            tst_scores = []

            trn_pred = self.predict_chain(trnX)
            for j in range(self.b):
                trn_scores.append(np.mean(self.scoring_fn(trnY, trn_pred[j])))
            print("[epoch %6d] trn:" % (epoch_i + predict_period), trn_scores)

            if with_validation:
                val_pred = self.predict_chain(valX)
                for j in range(self.b):
                    val_scores.append(np.mean(self.scoring_fn(valY, val_pred[j])))
                print("[epoch %6d] val:" % (epoch_i + predict_period), val_scores)

            if self.tst_ds is not None:
                tstX, tstY = self.tst_ds
                tst_pred = self.predict_chain(tstX)
                for j in range(self.b):
                    tst_scores.append(np.mean(self.scoring_fn(tstY, tst_pred[j])))
                print("[epoch %6d] tst:" % (epoch_i + predict_period), tst_scores)

            self.history.append({
                'epoch_nb': epoch_i,
                'trn_scores': trn_scores,
                'val_scores': val_scores,
                'tst_scores': tst_scores,
            })

            if epoch_i % 100 == 90 and self.weight_dir:
                self.model.save_weights(self.weight_dir + '_epoch_%d' % epoch_i,
                                        overwrite=True)
        return

    def predict_chain(self, X):
        ret = [[] for i in range(self.b)]
        batches = range(X.shape[0] // self.batch_size
                        + ((X.shape[0] % self.batch_size) > 0))
        _ = np.ones((self.batch_size, self.b, self.n_labels))

        with IterTimer("Predicting training data", total=len(batches)) as timer:
            for bs in batches:
                timer.update(bs)
                if (bs+1) * self.batch_size > X.shape[0]:
                    batch_idx = np.arange(X.shape[0])[bs * self.batch_size: X.shape[0]]
                else:
                    batch_idx = np.arange(X.shape[0])[bs * self.batch_size: (bs+1) * self.batch_size]

                pred_chain = self.model.predict([self._prep_X(X[batch_idx]), _])
                pred_chain = pred_chain > 0.5

                for i in range(self.b):
                    ret[i].append(ss.csr_matrix(pred_chain[:, i, :], dtype=np.int8))
        for i in range(self.b):
            ret[i] = ss.vstack(ret[i])
        return ret


    def predict(self, X, method="last"):
        X = ss.csr_matrix(X)
        pred_chain = self.predict_chain(X)

        # last
        if method == 'last':
            pred = pred_chain[-1]

        # first
        elif method == 'first':
            pred = pred_chain[0]

        #print(pred)
        return pred

    def predict_topk(self, X, k=5):
        ret = np.zeros((self.b, X.shape[0], k), np.float32)
        batches = range(X.shape[0] // self.batch_size \
                        + ((X.shape[0] % self.batch_size) > 0))
        _ = np.ones((self.batch_size, self.b, self.n_labels))

        for bs in batches:
            if (bs+1) * self.batch_size > X.shape[0]:
                batch_idx = np.arange(X.shape[0])[bs * self.batch_size: X.shape[0]]
            else:
                batch_idx = np.arange(X.shape[0])[bs * self.batch_size: (bs+1) * self.batch_size]

            pred_chain = self.model.predict([self._prep_X(X[batch_idx]), _])

            for i in range(self.b):
                ind = np.argsort(pred_chain[:, i, :], axis=1)[:, -k:][:, ::-1]
                ret[i, batch_idx, :] = ind

        return ret


class InputGenerator(object):
    def __init__(self, model, X, Y=None, pred=None, shuffle=False, batch_size=256, random_state=None):
        self.model = model
        self.X = X
        self.Y = Y
        self.lock = threading.Lock()
        if random_state is None:
            self.random_state = np.random.RandomState()
        self.index_generator = self._flow_index(X.shape[0], batch_size, shuffle, random_state)
        self.dummy_weight = np.ones((batch_size, self.model.b, self.model.n_labels),
                                    dtype=float) 

        self.pred = pred

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def _flow_index(self, n, batch_size, shuffle, random_state):
        index = np.arange(n)
        for epoch_i in itertools.count():
            if shuffle:
                random_state.shuffle(index)
            for batch_start in range(0, n, batch_size):
                batch_end = min(batch_start + batch_size, n)
                yield epoch_i, index[batch_start: batch_end]

    def next(self):
        with self.lock:
            epoch_i, index_array = next(self.index_generator)
        batch_X = self.X[index_array]
        preped_X = self.model._prep_X(batch_X)

        if self.Y is None:
            return [preped_X, self.dummy_weight]
        else:
            batch_Y = self.Y[index_array]
            preped_Y = self.model._prep_Y(batch_Y)
            # if epoch_i % 5 == 4:
            #     pred = self.model.model.predict([preped_X, self.dummy_weight])
            #     pred = (pred > 0.5).astype(np.int8)
            #     pred = [ss.csr_matrix(pred[:, j, :]) for j in range(self.model.b)]
            #     for j in range(self.model.b):
            #         self.pred[j][index_array] = pred[j]
            # else:
            pred = [self.pred[j][index_array] for j in range(self.model.b)]

            if self.model.reweight_scoring_fn in [
                    #reweight_pairwise_hamming,
                    sparse_reweight_pairwise_acc,
                    sparse_reweight_pairwise_f1,
                    sparse_reweight_pairwise_rankloss]:
                lbl_weight = self.model._prep_weight(pred, batch_Y)
            else:
                lbl_weight = self.model._prep_weight(pred, preped_Y[:, 0, :])
            return [preped_X, lbl_weight], preped_Y
