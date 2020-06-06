import os
import numpy as np
import random
import matplotlib.pyplot as plt

from collections import OrderedDict

import json
import pickle
import itertools


cmaps = OrderedDict()
cmaps['Sequential'] = [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

cmaps['Sequential-my'] = [
            'YlOrBr', 'Blues', 'Reds', 'Greens', 'Oranges', 'Purples', 'PuRd', 'GnBu', 'BuGn']

def create_plot_labels():
    pass

def select_relevant_exps():
    pass

def plot_simple_run(data, labels, fig_size=(10,5)):

    n_points = 100

    if data == None:
        data_points = {}
        data_points['bce-loss/train'] = np.random.rand(n_points) * 2
        data_points['bce-loss/test'] = np.random.rand(n_points) * 2

        #data_points['bce-log-loss/train'] = np.random.rand(n_points) * 2
        #data_points['bce-log-loss/test'] = np.random.rand(n_points) * 2

    fig, ax = plt.subplots(figsize=fig_size)
    plt.style.use('seaborn')

    #color = iter(plt.cm.rainbow(np.linspace(0, 1, len(data_points))))
    #c = next(color)
    c = 'blue'
    for key, vals in data_points.items():
        if key.__contains__('test'):
            ls = '--'
        else:
            ls = '-'
        ax.plot(vals, label=key, linestyle=ls)

    ax.set_xlabel('Epoch')  # Add an x-label to the axes.
    ax.set_ylabel('y label')  # Add a y-label to the axes.
    ax.set_title("Simple Plot")  # Add a title to the axes.
    ax.legend()  # Add a legend.

    plt.show()

def plot_experiments(exp_data, exp_labels : [str], metric_first_plot : [str], xlabel, metric_exclude : [str], sub_plots=(1, 2), fig_size=(10, 8)):

    '''
    input: list of dicts
    each list entry (experiment) contains losses (train + test) and metrics (4)
    corresponding metrics have same color but test is dashed

    data format: list(dict) [exp1, exp2, ... ]

    exp1:
    {   metric1 : (train, test)},
        metric2 : (train, test)},
        ... }

    exp2:
    {   metric1 : (train, test)},
        metric2 : (train, test)},
        ... }

    - From list of Sequential Colormaps Select n_experiments cmaps
    - each cmap n_metrics colors

    :return:
    '''
    if False:
        exp_data = []
        exp_labels = []
        metrics = ["loss", "acc", "ap"]
        metric_first_plot = ['loss']

        n = 10
        for i in range(3):
            exp_data.append({key: (np.random.rand(n), np.random.rand(n)) for key in metrics})
            exp_labels.append("exp " + str(i))

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=fig_size)

    n_exps = len(exp_data)
    #seq_cmaps = random.sample(cmaps['Sequential'], n_exps)
    seq_cmaps = cmaps['Sequential-my'][:n_exps]
    for i, exp in enumerate(exp_data):
        #select color map
        cmap = plt.get_cmap(seq_cmaps[i])
        n_metrics = len(exp.keys()) - len(metric_exclude)
        #create colors for each metric
        colors = iter(random.sample(list(cmap(np.linspace(0.5, 0.9, n_metrics))), n_metrics))
        for key in exp.keys():
            if key not in metric_exclude:
                c = next(colors)
                train_vals, test_vals = exp[key]
                lbl = key
                if key in metric_first_plot:
                    ax = ax1
                else:
                    ax = ax2
                lbl = ' '.join([exp_labels[i], lbl])
                ax.plot(train_vals, label=lbl, color=c, linestyle='-')
                ax.plot(test_vals, label=lbl + '/test', color=c, linestyle='--')

    ax1.set_ylabel('loss value')
    ax1.set_xlabel(xlabel)
    ax1.set_title('Loss')
    ax1.legend()

    ax2.set_title('Performance Metrics')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def average_metrics_over_runs(result_metrics, add_var=False):
    # aggregate metrics over multiple runs
    #loss, acc, auc, ap
    agg_metrics = zip(*[run.values() for run in result_metrics]) #
    agg_metrics = dict(zip(list(result_metrics[0].keys()), agg_metrics)) # create dict from zip(keys, vals)

    # compute mean for each position
    for key, m_vals in agg_metrics.items():
        train, test = zip(*m_vals)
        # zipped list where each entry concats values at the same position
        # [ (train1[0], train2[0], train3[0]), (train1[1], train2[1], train3[1]), ... ]

        # zip and average / apply function  +  assign back to avg_results
        agg_metrics[key] = (zip_and_apply(train, np.mean),
                            zip_and_apply(test, np.mean))

        if add_var:
            agg_metrics[key + '-var'] = (zip_and_apply(train, np.var),
                                        zip_and_apply(test, np.var))

    return agg_metrics

def zip_and_apply(iterable, fnc):
    #values = list(zip(*values))
    #values = list(map(np.mean, values))
    return list(map(fnc, list(zip(*iterable))))



if __name__ == "__main__":
    #plot_simple_run(None, None)
    experiments = {}
    experiments['03-21-20'] = {
        'res_path': '../results/03-21-20/',
        'files': {'mean': ['exp15-14:18', 'exp17-14:21', 'exp19-14:25', 'exp21-14:29'],
             'batches': ['exp16-14:19', 'exp18-14:23', 'exp20-14:27', 'exp22-14:31']}
    }

    experiments['03-23-20'] = {
        'res_path': '../results/03-23-20/',
        'files': {'epoch': ['exp4epoch-wu-42-17:26', 'exp5epoch-wu-13-17:28', 'exp6epoch-wu-102-17:30']},
        'comment': "seed 102 is an outlier with very high test loss"
    }

    experiments['03-25-20'] = {
        'res_path': '../results/03-25-20/',
        'files': {'epoch': ['dev-epoch-wu-42-10:51', 'dev-wd1e4-epoch-wu-42-10:53'
                            , 'large-epoch-wu-42-22:03', 'large-epoch-wu-13-02:02', 'large-epoch-wu-102-06:01']},
        'comment': ""
    }

    res_path = '../results/03-25-20/'

    methods = ['epoch', 'batches']
    method = methods[0]
    name = '/metrics_' + method + '.pkl'
    files = experiments['03-25-20']['files']

    result_met_avg = []
    result_labels = ['wu-large-avg3', 'wu-large-13', 'wu-large-102']

    ##
    result_metrics = []
    for file in ['large-epoch-wu-42-22:03', 'large-epoch-wu-13-02:02', 'large-epoch-wu-102-06:01']:
        with open(res_path + file + name, 'rb') as fin:
            result_metrics.append(pickle.load(fin))

    result_met_avg.append(average_metrics_over_runs(result_metrics))

    ##
    #'wu-weight-decay'
    # res_path = '../results/03-24-20/'
    # files = ['epoch-wu-102-17:28', 'epoch-wu-42-17:24', 'epoch-wu-13-17:26']
    # result_metrics = []
    # for file in files:
    #     with open(res_path + file + name, 'rb') as fin:
    #         result_metrics.append(pickle.load(fin))
    #
    # result_met_avg.append(average_metrics_over_runs(result_metrics))

    plot_experiments(exp_data=result_met_avg, exp_labels=result_labels,
                     metric_first_plot=['loss'], xlabel='epoch', metric_exclude=['acc'])