import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import datagen

MARKER = ['o', 'v', 's', 'p', 'h', '^', 'D', '*', 'H']
COLOR = plt.rcParams['axes.prop_cycle'].by_key()['color']


def prediction_metrics(dat_gen, preprocess, size, metrics, para_col, para_row, config_id,
                       methods=None, para_format=None, xlabel=None, ylabels=None):
    if methods is None:
        methods = ['$f_{ML}$', 'FTU', '$f_{EO}$', '$f_{AA}$', '$f_{1}$', '$f_{2}$']
    metric_table_name = ['result/{:s}_n{:s}_{:s}_preprocess_{:s}_'.format(
        dat_gen.__name__, str(size), preprocess, config_id), '.npy']
    metric_tables = ()
    metrics_exist = []
    for metric in metrics:
        try:
            metric_table = np.load(metric.join(metric_table_name))
            metric_tables += (metric_table.take(para_row, axis=1),)
            metrics_exist.append(metric)
        except FileNotFoundError as e:
            print(e)
    if len(metrics_exist) == 0:
        raise Exception('No metric table file is found.')
    metrics = metrics_exist
    figure_name = 'figure/{:s}_{:s}_preprocess_{:s}_{:s}.pdf'.format(
        dat_gen.__name__, preprocess, config_id, '_'.join(metrics))
    para = metric_tables[0][0, :, :-len(methods)]
    para_name = dat_gen.__code__.co_varnames[1:para.shape[1] + 1]
    para = para.take(para_col, axis=1)
    para_name = [para_name[i] for i in para_col]
    if len(para_col) == 1:
        x = para.squeeze()
    else:
        if para_format is None:
            para_format = ', '.join(['{:.1f}'] * len(para_col))
        x = [para_format.format(*p) for p in para]
    if xlabel is None:
        xlabel = "Difference due to " + ', '.join(para_name)
    y_dict = {'eo': 'EO-fairness', 'aa': 'AA-fairness', 'acc': 'Test accuracy'}

    fig, axes = plt.subplots(1, len(metrics), figsize=(3 * len(metrics), 3))
    for j, ax in enumerate(axes):
        for i in range(len(methods)):
            ax.errorbar(x, metric_tables[j][0, :, i - len(methods)],
                        metric_tables[j][1, :, i - len(methods)], label=methods[i])
        ax.set_xlabel(xlabel)
        if len(para_col) > 1:
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
        ylabel = y_dict[metrics[j]] if ylabels is None else ylabels[j]
        ax.set_ylabel(ylabel)
    plt.tight_layout()
    axes[-1].legend(bbox_to_anchor=(1.02, 0.5), loc="center left", ncol=1)
    plt.savefig(figure_name, bbox_inches='tight')


def power_comparison(dat_gen, preprocess, sizes, para_loc, config_id,
                     para_format=None, xlabel=None, ylabel=None):
    power_table_name = ['result/' + dat_gen.__name__ + '_n',
                        '_' + preprocess + '_preprocess_' + config_id + '_power_table.npy']
    figure_name = 'figure/' + dat_gen.__name__ + '_' + preprocess + '_preprocess_' + config_id + '_power.pdf'
    power_tables = ()
    sizes_exist = []
    for size in sizes:
        try:
            power_tables += (np.load(str(size).join(power_table_name)),)
            sizes_exist.append(size)
        except FileNotFoundError as e:
            print(e)
    if len(sizes_exist) == 0:
        raise Exception('No power table file is found.')
    sizes = sizes_exist
    para = power_tables[0][:, :-1]
    para_name = dat_gen.__code__.co_varnames[1:para.shape[1] + 1]
    para_full, para = para, para.take(para_loc, axis=1)
    para_name = [para_name[i] for i in para_loc]
    power = np.column_stack([table[:, -1] for table in power_tables]).transpose()
    marker = MARKER[:len(sizes)]
    if para_format is None:
        try:
            unfairness_metric = eval('datagen.' + dat_gen.__name__ + '_unfairness')
            x = [unfairness_metric(*p) for p in para_full]
        except NameError:
            para_format = '(' + ', '.join(['{:.1f}'] * len(para_loc)) + ')'
            x = [para_format.format(*p) for p in para]
    else:
        x = [para_format.format(*p) for p in para]

    fig, ax = plt.subplots()
    for i, size in enumerate(sizes):
        ax.scatter(x, power[i], marker=marker[i], label=size)
    ax.axhline(0.05, color='black', linewidth=0.5)
    plt.yticks(list(plt.yticks()[0][1:-1]) + [0.05])
    if xlabel is None:
        xlabel = "Bias and historical disadvantage (" + ', '.join(para_name) + ")"
    ax.set_xlabel(xlabel)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    if ylabel is None:
        ylabel = "P(reject $H_0$|" + ', '.join(para_name) + ")"
    ax.set_ylabel(ylabel)
    plt.legend(loc='best', title='sample size')
    plt.tight_layout()
    plt.savefig(figure_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config_path', type=str, default=None,
                        help='path to the config file')
    parser.add_argument('--mode', dest='mode', type=str, default='eval',
                        help='experiment mode, eval or test')
    parser.add_argument('-m', '-M', dest='M', type=int, default=100,
                        help='number of replicated experiments')
    parser.add_argument('-n', '-N', dest='N', type=int, default=5000,
                        help='sample size')
    parser.add_argument('--preprocess_method', dest='preprocess_method', type=str, default='m',
                        help='marginal or orthogonal preprocessing of data')
    args = parser.parse_args()

    if args.config_path is None:
        # global options
        mode = args.mode
        M, N = args.M, args.N
        preprocess_method = args.preprocess_method
        data_generator_fun = datagen.dat_gen_loan_univariate
        eval_metrics = ['eo', 'aa', 'acc']
        sample_sizes = [50, 100, 200]
        parameter_loc = [2, 3, 4]
        parameter_col = [4]
        parameter_row = list(range(5))
        parameter_format, x_label, y_label = None, None, None
        identifier = 'config_default'
    else:
        with open(args.config_path) as f:
            config = json.load(f)
        mode = config['mode']
        M, N = config['M'], config['N']
        data_generator_fun = eval(config['data_generator_fun'])
        preprocess_method = config['preprocess_method']
        parameter_format = config['parameter_format']
        x_label = config['x_label']
        y_label = config['y_label']
        identifier = 'config_' + args.config_path.split('_')[-1][:-5]
        if mode == 'test':
            sample_sizes = config['sample_sizes']
            parameter_loc = config['parameter_loc']
        elif mode == 'eval':
            eval_metrics = config['eval_metrics']
            parameter_col = config['parameter_col']
            parameter_row = config['parameter_row']
        else:
            pass

    preprocess_method_dict = {'m': 'marginal', 'o': 'orthogonal'}

    if mode == 'test':
        power_comparison(data_generator_fun, preprocess_method_dict[preprocess_method],
                         sample_sizes, parameter_loc, identifier,
                         para_format=parameter_format, xlabel=x_label, ylabel=y_label)
    if mode == 'eval':
        prediction_metrics(data_generator_fun, preprocess_method_dict[preprocess_method],
                           N, eval_metrics, parameter_col, parameter_row, identifier,
                           para_format=parameter_format, xlabel=x_label, ylabels=y_label)
