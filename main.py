import json
import os
import inspect
from functools import partial

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import pairwise_distances, accuracy_score
import tensorflow as tf
#import tensorflow.compat.v1 as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import keras.backend
#import tensorflow.keras.backend

#from tensorflow import keras

from nnattack.variables import auto_var


def set_random_seed(auto_var):
    random_seed = auto_var.get_var("random_seed")

    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
    keras.layers.core.K.set_learning_phase(0)
    #tensorflow.keras.backend.set_session(sess)
    sess.run(tf.global_variables_initializer())
    auto_var.set_intermidiate_variable("sess", sess)
    random_state = np.random.RandomState(auto_var.get_var("random_seed"))
    auto_var.set_intermidiate_variable("random_state", random_state)

    return random_state

def baseline_pert(model, trnX, tstX, tsty, perts, ord, constraint=None):
    pred_trn = model.predict(trnX)
    ret = np.copy(perts)
    for i in np.where(model.predict(tstX + perts) == tsty)[0]:
        tX = trnX[pred_trn != tsty[i]]
        if len(tX) == 0:
            continue
        norms = np.linalg.norm(tX - tstX[i], ord=ord, axis=1)
        if constraint is not None and norms.min() > constraint:
            continue
        ret[i] = tX[norms.argmin()] - tstX[i]
    return ret, (model.predict(tstX + perts) == tsty).sum()


def pass_random_state(fn, random_state):
    if 'random_state' in inspect.getfullargspec(fn).args:
        return partial(fn, random_state=random_state)
    return fn

def estimate_model_roubstness(model, X, y, perturbs, eps_list, ord,
        with_baseline=False, trnX=None):
    assert len(eps_list) == len(perturbs), (eps_list, perturbs.shape)
    ret = []
    for i, eps in enumerate(eps_list):
        assert np.all(np.linalg.norm(perturbs[i], axis=1, ord=ord) <= (eps + 1e-6)), (np.linalg.norm(perturbs[i], axis=1, ord=ord), eps)
        if with_baseline:
            assert trnX is not None
            pert, _ = baseline_pert(model, trnX, X, y, perturbs[i], ord, eps)
            temp_tstX = X + pert
        else:
            temp_tstX = X + perturbs[i]

        pred = model.predict(temp_tstX)

        ret.append({
            'eps': eps_list[i],
            'tst_acc': (pred == y).mean().astype(float),
        })
    return ret

def verify_adv(model, x_adv, perts, y_adv):
    predict = model.predict(x_adv+perts)
    missed_count = 0
    for i in range(len(x_adv)):
        if(predict[i] == y_adv[i]):
            missed_count+=1

    return missed_count

def knockoff(model, x_adv, y_adv, perts, noise_scale, x_test, y_test):
    predictions = model.predict(x_adv+perts*noise_scale)
    knockoff_model = KNeighborsClassifier(n_neighbors=1)
    knockoff_model.fit(x_adv+perts*noise_scale, y_adv)
    
    predictions = knockoff_model.predict(x_test)
    knockoff_acc = accuracy_score(predictions,y_test)

    return knockoff_acc

def eps_accuracy(auto_var):
    random_state = set_random_seed(auto_var)
    ord = auto_var.get_var("ord")

    dataset_name = auto_var.get_variable_name("dataset")
    if ('fullmnist' in dataset_name \
        or 'fullfashion' in dataset_name \
        or 'cifar' in dataset_name \
        or 'fashion_mnist35f' in dataset_name \
        or 'fashion_mnist06f' in dataset_name \
        or 'mnist17f' in dataset_name \
        or 'cifar' in dataset_name
        ):
        X, y, x_test, y_test, eps_list = auto_var.get_var("dataset")
        idxs = np.arange(len(x_test))
        random_state.shuffle(idxs)
        tstX, tsty = x_test[idxs[:200]], y_test[idxs[:200]]
        idxs = np.arange(len(X))
        random_state.shuffle(idxs)
        X, y = X[idxs], y[idxs]

        trnX, tstX = X.reshape((len(X), -1)), tstX.reshape((len(tstX), -1))
        trny = y
    else:
        X, y, eps_list = auto_var.get_var("dataset")
        idxs = np.arange(len(X))
        random_state.shuffle(idxs)
        trnX, tstX, trny, tsty = X[idxs[:-100]], X[idxs[-200:]], y[idxs[:-100]], y[idxs[-200:]]

    scaler = MinMaxScaler()
    trnX = scaler.fit_transform(trnX)
    tstX = scaler.transform(tstX)

    lbl_enc = OneHotEncoder(categories=[np.sort(np.unique(y))], sparse=False)
    lbl_enc.fit(trny.reshape(-1, 1))

    auto_var.set_intermidiate_variable("lbl_enc", lbl_enc)

    ret = {}
    results = []

    x_adv = trnX;
    y_adv = trny;

    trnX = trnX[:-100];
    trny = trny[:-100]

    auto_var.set_intermidiate_variable("trnX", trnX)
    auto_var.set_intermidiate_variable("trny", trny)

    model_name = auto_var.get_variable_name("model")
    attack_name = auto_var.get_variable_name("attack")

    model = auto_var.get_var("model")
    auto_var.set_intermidiate_variable("model", model)
    model.fit(trnX, trny)
    ret['trnX_len'] = len(trnX)

    pred = model.predict(tstX)
    baseline_acc = accuracy_score(pred, tsty)
#    print(f"Baseline accuracy: {(pred == tsty).mean()}")
    ori_tstX, ori_tsty = tstX, tsty # len = 200
    pred_adv = model.predict(x_adv)
    idxs = np.where(pred_adv == y_adv)[0]
    random_state.shuffle(idxs)
    x_adv, y_adv = x_adv[idxs[:len(trnX)]], y_adv[idxs[:len(trnX)]]
    if len(x_adv) != len(trnX):
        print("didn't got enough adversarial examples, abort.")
        ret['avg_pert'] = {'avg': 0, 'missed_count': 100,}
        ret['tst_score'] = (model.predict(ori_tstX) == ori_tsty).mean()
        if ('adv' in model_name) or ('advPruning' in model_name) or ('robustv2' in model_name):
            ret['aug_len'] = len(model.augX)
        return ret
        #raise ValueError("didn't got 100 testing examples")

#    if len(tsty) != 100 or \
#       (np.unique(auto_var.get_intermidiate_variable('trny'))[0] != None and \
#       len(np.unique(auto_var.get_intermidiate_variable('trny'))) == 1):
#        tst_perturbs = np.array([np.zeros_like(tstX) for _ in range(len(eps_list))])
#        ret['single_label'] = True
#        attack_model = None
#    else:
#         attack_model = auto_var.get_var("attack")
#         adv_perturbs = attack_model.perturb(x_adv, y=y_adv, eps=eps_list)

    attack_model = auto_var.get_var("attack")
    adv_perturbs = attack_model.perturb(x_adv, y=y_adv, eps=eps_list)
    
    ret['tst_score'] = (model.predict(ori_tstX) == ori_tsty).mean()

    #########
#    perts = attack_model.perts

    if attack_model is not None and hasattr(attack_model, 'perts'):
        perts = attack_model.perts
    else:
        perts = np.zeros_like(x_adv)
        for pert in adv_perturbs:
            pred = model.predict(x_adv + pert)
            for i in range(len(pred)):
                if (pred[i] != y_adv[i]) and np.linalg.norm(perts[i])==0:
                    perts[i] = pert[i]

    perts = perts.astype(float)

#    missed_count = verify_adv(model, x_adv, perts, y_adv)    

    print('Training size : ', len(trnX))
    print('Baseline accuracy : %s' % '{0:.3%}'.format(baseline_acc)) 
#    print('Failed to find : %s' % '{0:.3%}'.format(missed_count)) 
    
#    noise_scale = auto_var.get_var("noise_scale")
    knockoff_acc = knockoff(model, x_adv, y_adv, perts, 0.9, ori_tstX, ori_tsty) 
   
    print('Knockoff model accuracy : %s' % '{0:.3%}'.format(knockoff_acc)) 

#    print('Perturbation :  %s' % '{0:.3%}'.format(np.linalg.norm(perts, axis=1, ord=ord).mean().astype(float)))

    perts, missed_count = baseline_pert(model, trnX, x_adv, y_adv, perts, ord)
    print('Missed count : ', missed_count)
    #if len(np.unique(model.predict(trnX))) > 1:
    #    assert (model.predict(tstX + perts) == tsty).sum() == 0, model.predict(tstX + perts) == tsty
    #else:
    #    # ignore single label case
    #    ret['single_label'] = True
    ret['avg_pert'] = {
        'avg': np.linalg.norm(perts, axis=1, ord=ord).mean().astype(float),
        'missed_count': int(missed_count),
    }
    print('Distortion avg: ', np.linalg.norm(perts, axis=1, ord=ord).mean().astype(float))
    #########

#    results = estimate_model_roubstness(
#        model, tstX, tsty, tst_perturbs, eps_list, ord, with_baseline=False)
#    ret['results'] = results
#    baseline_results = estimate_model_roubstness(
#        model, tstX, tsty, tst_perturbs, eps_list, ord, with_baseline=True, trnX=trnX)
#    ret['baseline_results'] = baseline_results

#    print(json.dumps(auto_var.var_value))
#    print(json.dumps(ret))
    return ret

def main():
    auto_var.parse_argparse()
    auto_var.run_single_experiment(eps_accuracy)

if __name__ == '__main__':
    main()
