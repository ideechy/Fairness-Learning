import argparse
import json
import numpy as np
import datagen
from fairdata import FairData
from multiprocessing import Pool
from functools import partial


def fairness(dat_gen, n_train, n_test, paras, preprocess='m', metrics=None):
    if metrics is None:
        metrics = ['eo', 'aa', 'acc']
    np.random.seed(None)
    result = np.zeros((paras.shape[0], len(metrics), 6))
    n = n_train + n_test
    for i, para in enumerate(paras):
        s, a, y = dat_gen(n, *para)
        data = FairData(s[:n_train], a[:n_train], y[:n_train], preprocess_method=preprocess, mode='predict')
        result[i] = np.asarray(data.evaluate(s[n_train:], a[n_train:], y[n_train:], metrics=metrics))
    return result


def parallel_fairness(dat_gen, n_train, n_test, paras, m, num_procs=4, preprocess='m', metrics=None):
    if metrics is None:
        metrics = ['eo', 'aa', 'acc']
    pool = Pool(num_procs)
    _fairness_ = partial(fairness, dat_gen, n_train, n_test, paras, preprocess, metrics)
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
    parser.add_argument(
        '-c', '--config', 
        dest='config_path', 
        type=str,
        default=None,
        help='path to the config file'
    )
    args = parser.parse_args()
    config_path = args.config_path or 'config/admission_n100_test_AllChange.json'

    with open(config_path) as f:
        config = json.load(f)
    mode = config['mode']
    M, N, P = config['M'], config['N'], config['P']
    data_generator_fun = eval(config['data_generator_fun'])
    parameters = eval(config['parameters'])
    preprocess_method = config['preprocess_method']
    identifier = 'config_' + config_path.split('_')[-1][:-5]
    if mode == 'test':
        B = config['B']
    elif mode == 'eval':
        adv_group = config['adv_group']
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
        N_train = int(N * .8)
        N_test = N - N_train
        eval_results = parallel_fairness(data_generator_fun, N_train, N_test, parameters,
                                         M, P, preprocess_method, eval_metrics)
        for i, metric in enumerate(eval_metrics):
            eval_result = [np.hstack((parameters, res)) for res in eval_results[:, :, i, :]]
            np.save(file_prefix + metric, np.asarray(eval_result))
    else:
        pass
