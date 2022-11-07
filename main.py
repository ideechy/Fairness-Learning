#!/usr/bin/env python3

import argparse
import json
import re
import numpy as np
from multiprocessing import Pool
from functools import partial

import datagen
from fairdata import FairData

def true_cf(dat_gen, n_train, n_test, paras, preprocess='m', methods=None):
    if methods is None:
        methods = ['ML', 'FTU', 'FL', 'AA', 'FLAP-1', 'FLAP-2']
    dat_gen_cf = eval(f'{dat_gen.__module__}.{dat_gen.__name__}_counterfactual')
    np.random.seed(0)
    result = np.zeros((paras.shape[0], len(methods)))
    for i, para in enumerate(paras):
        s_train, a_train, y_train = dat_gen(n_train, *para)
        data = FairData(
            s_train=s_train, a_train=a_train, y_train=y_train, 
            preprocess_method=preprocess, mode='predict'
        )
        a_test = dat_gen_cf(n_test, *para)
        result[i] = data.cf_true(a=a_test, methods=methods)
    return result

def fairness(dat_gen, n_train, n_test, paras, preprocess='m', metrics=None, methods=None):
    if metrics is None:
        metrics = ['cf', 'mae', 'roc', 'ap']
    if methods is None:
        methods = ['ML', 'FTU', 'FL', 'AA', 'FLAP-1', 'FLAP-2']
    np.random.seed(None)
    result = np.zeros((paras.shape[0], len(metrics), len(methods)))
    n = n_train + n_test
    for i, para in enumerate(paras):
        s, a, y = dat_gen(n, *para)
        data = FairData(
            s_train=s[:n_train], a_train=a[:n_train], y_train=y[:n_train], 
            preprocess_method=preprocess, mode='predict'
        )
        result[i] = np.asarray(data.evaluate(
            a_test=a[n_train:], s_test=s[n_train:], y_test=y[n_train:], 
            metrics=metrics, methods=methods
        ))
    return result


def parallel_fairness(dat_gen, n_train, n_test, paras, m, num_procs=4, preprocess='m', metrics=None, methods=None):
    if metrics is None:
        metrics = ['cf', 'mae', 'roc', 'ap']
    if methods is None:
        methods = ['ML', 'FTU', 'FL', 'AA', 'FLAP-1', 'FLAP-2']
    pool = Pool(num_procs)
    _fairness_ = partial(fairness, dat_gen, n_train, n_test, paras, preprocess, metrics, methods)
    experiments = [pool.apply_async(_fairness_) for _ in range(m)]
    res = np.asarray([e.get() for e in experiments])
    return np.asarray((res.mean(axis=0), res.std(axis=0), *np.percentile(res, [2.5, 97.5], axis=0)))


def cit(dat_gen, n, paras, preprocess='m', b=99):
    np.random.seed(None)
    p_vals = np.zeros(paras.shape[0])
    for i, para in enumerate(paras):
        s, a, y = dat_gen(n, *para)
        dat = FairData(s, a, y, preprocess_method=preprocess, mode='test')
        p_vals[i] = dat.cit(b=b, type='cdc')
    return p_vals


def parallel_cit(dat_gen, n, paras, m=1000, num_procs=4, preprocess='m', b=99):
    pool = Pool(num_procs)
    _cit_ = partial(cit, dat_gen, n, paras, preprocess, b)
    experiments = [pool.apply_async(_cit_) for _ in range(m)]
    p_vals = [e.get() for e in experiments]
    return np.asarray(p_vals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config_path', type=str, default=None,
                        help='path to the config file')
    parser.add_argument('--mode', dest='mode', type=str, default='eval',
                        help='experiment mode, eval or test')
    parser.add_argument('-m', '-M', dest='M', type=int, default=100,
                        help='number of replicated experiments')
    parser.add_argument('-n', '-N', '-N_train', dest='N', type=int, default=5000,
                        help='training sample size')
    parser.add_argument('-t', '-T', '-N_test', dest='T', type=int, default=None,
                        help='test sample size')
    parser.add_argument('-b', '-B', dest='B', type=int, default=99,
                        help='number of bootstrap for fairness test')
    parser.add_argument('-p', '-P', dest='P', type=int, default=8,
                        help='number of processors for multiprocessing')
    parser.add_argument('--preprocess_method', dest='preprocess_method', type=str, default='m',
                        help='marginal or orthogonal preprocessing of data')
    args = parser.parse_args()

    if args.config_path is None:
        # global options
        mode = args.mode
        M, N, T, B, P = args.M, args.N, args.T, args.B, args.P
        preprocess_method = args.preprocess_method
        data_generator_fun = datagen.dat_gen_loan_univariate
        parameters = np.mgrid[-1:0, 2:3, 1:2:1, 0.5:1:0.5, 1:3:0.4].reshape(5, -1).transpose()
        eval_metrics = ['eo', 'cf', 'acc', 'mae']
        identifier = 'default_config'
    else:
        with open(args.config_path) as f:
            config = json.load(f)
        mode = config['mode']
        M, N, P = config['M'], config['N'], config['P']
        data_generator_fun = eval(config['data_generator_fun'])
        parameters = eval(config['parameters'])
        preprocess_method = config['preprocess_method']
        identifier = 'config_' + args.config_path.split('/')[-1][:-5]
        if mode == 'test':
            B = config['B']
        elif mode == 'eval':
            T = config['T'] if 'T' in config else None
            eval_metrics = config['eval_metrics']
        else:
            pass

    preprocess_method_dict = {'m': 'marginal', 'o': 'orthogonal'}
    file_prefix = 'result/{:s}_n{:d}_{:s}_preprocess_{:s}_'.format(
        data_generator_fun.__name__, N, preprocess_method_dict[preprocess_method], identifier)
    if mode == 'test':
        p_values = parallel_cit(data_generator_fun, N, parameters, M, P, preprocess_method, B)
        power = np.column_stack((parameters, np.mean(p_values <= 0.05, axis=0)))
        print(power)
        np.save(file_prefix + 'p_value', p_values)
        np.save(file_prefix + 'power_table', power)
    elif mode == 'eval':
        if T is None:
            N_train = int(N * .8)
            N_test = N - N_train
        else:
            N_train, N_test = N, T
        eval_results = parallel_fairness(data_generator_fun, N_train, N_test, parameters,
                                         M, P, preprocess_method, eval_metrics)
        for i, metric in enumerate(eval_metrics):
            eval_result = [np.hstack((parameters, res)) for res in eval_results[:, :, i, :]]
            np.save(file_prefix + metric, np.asarray(eval_result))
        # truth = true_cf(data_generator_fun, N_train, N_train, parameters, preprocess_method)
        # np.save(file_prefix + 'cft', np.expand_dims(np.hstack((parameters, truth)), 0))
    else:
        pass
