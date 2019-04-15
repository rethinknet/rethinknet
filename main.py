
from os.path import join
import pickle
import gzip
from datetime import datetime

import numpy as np
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import scipy.sparse as ss
from bistiming import IterTimer, SimpleTimer
from keras.callbacks import TensorBoard
from keras import backend as K

from rethinknet.models.utils import get_random_state
from rethinknet.utils import load_data, load_extreme, get_model, get_scoring_fn, \
        load_split

def load_trn_tst(data_path, ds_name, split_no):
    X, Y = load_data(join(data_path, ds_name))
    split = load_split(join(data_path, ds_name), split_no)
    return X[split[0]], Y[split[0]], X[split[1]], Y[split[1]]

def nn_pred_save(model, X):
    first_pred = model.predict(X, 'first')
    last_pred = model.predict(X, 'last')

    return first_pred, None, last_pred


def main(data_path, ds_name, split_no, scoring, **kwargs):
    ret = {}

    scoring_fn = get_scoring_fn(scoring)

    trnX, trnY, tstX, tstY = load_trn_tst(data_path, ds_name, split_no)
    if kwargs['scaling'] == 'extreme_standard':
        scaler = StandardScaler(with_mean=False, copy=False)
    elif kwargs['scaling'] == 'maxabs':
        scaler = MaxAbsScaler()
    elif kwargs['scaling'] == 'standard':
        scaler = StandardScaler()
    elif kwargs['scaling'] == 'minmax':
        scaler = MinMaxScaler()
    elif kwargs['scaling'] == 'None':
        pass

    random_state = get_random_state(kwargs['random_state'])

    if kwargs['scaling'] != 'None':
        trnX = scaler.fit_transform(trnX)
        tstX = scaler.transform(tstX)

    print(trnX.shape, tstY.shape)
    ret['shapes'] = {
        'trnX': trnX.shape,
        'tstY': tstY.shape,
    }

    model_param = {
        'n_features': trnX.shape[1],
        'n_labels': trnY.shape[1],
        'scoring_fn': scoring_fn,
    }
    model_param.update(kwargs)

    print(model_param)

    model_name = "rethinkNet"

    trnX = ss.csr_matrix(trnX).astype('float32')
    trnY = ss.csr_matrix(trnY).astype(np.int)
    tstX = ss.csr_matrix(tstX).astype('float32')
    tstY = ss.csr_matrix(tstY).astype(np.int)

    from sklearn.model_selection import KFold
    k_fold = KFold(3)
    #params = [10**i for i in range(-8, 0, 1)]
    params = [10**-5]
    scores = []
    for c in params:
        temp = []
        for t, v in k_fold.split(trnX, trnY):
            model_param['l2w'] = c
            model = get_model(model_name, model_param)
            model.fit(trnX[t], trnY[t], with_validation=False)
            temp.append(
                np.mean(scoring_fn(model.predict(trnX[v]), trnY[v])))

            K.clear_session()

        scores.append(np.mean(temp))
    C = params[np.argmax(scores)]

    ret['validation_params'] = params
    ret['validation_scores'] = scores
    model_param['tst_ds'] = (tstX, tstY)
    model_param['log_name'] = None
    model_param['l2w'] = C
    model = get_model(model_name, model_param)
    model.fit(trnX, trnY, with_validation=False)


    first_pred, mean_pred, last_pred = nn_pred_save(model, trnX)
    print('[train] first', np.mean(scoring_fn(trnY, first_pred)))
    print('[train] last', np.mean(scoring_fn(trnY, last_pred)))
    ret['trn_score'] = {
        'first': np.mean(scoring_fn(trnY, first_pred)),
        'last': np.mean(scoring_fn(trnY, last_pred))
    }

    first_pred, mean_pred, last_pred = nn_pred_save(model, tstX)
    print('[test] first', np.mean(scoring_fn(tstY, first_pred)))
    print('[test] last', np.mean(scoring_fn(tstY, last_pred)))
    ret['tst_score'] = {
        'first': np.mean(scoring_fn(tstY, first_pred)),
        'last': np.mean(scoring_fn(tstY, last_pred))
    }

    ret['training_history'] = model.history
    ret['nb_params'] = model.nb_params

    print(ret)
    print("score:", ret['tst_score']['last'])
    return ret

if __name__ == "__main__":
    main("./mulan/splitted_data/", "emotions", 0, 'sparse_f1', scaling='minmax',
            random_state=0)
