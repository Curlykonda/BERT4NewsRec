import os
from abc import ABCMeta, abstractmethod

import torch
import json
from collections import defaultdict

def save_state_dict(state_dict, path, filename):
    torch.save(state_dict, os.path.join(path, filename))


class LoggerService(object):
    def __init__(self, train_loggers=None, val_loggers=None, grad_histo_logger=None):
        self.train_loggers = train_loggers if train_loggers else []
        self.val_loggers = val_loggers if val_loggers else []
        self.best_model_logger = [log for log in val_loggers if isinstance(log, BestModelLogger)][0]

        self.train_keys = [log.key for log in self.train_loggers]
        self.val_keys = [log.key for log in self.val_loggers if isinstance(log, MetricGraphScalar)]

        self.metrics = {'train': defaultdict(list),
                        'val': defaultdict(list),
                        'test': defaultdict(list)}

        self.user_metrics = {'val': defaultdict(dict),
                             'test': defaultdict(dict)}

        ##
        # currently not used [14.07.20]
        self.grad_histo_logger = grad_histo_logger
        self.grad_reports = defaultdict(list)
        ###

    def complete(self, log_data):
        for logger in self.train_loggers:
            logger.complete(**log_data)
        for logger in self.val_loggers:
            logger.complete(**log_data)

        # add best val to metrics
        best_epoch = self.best_model_logger.best_epoch
        self.metrics['val_best'] = {}
        for key, vals in self.metrics['val'].items():
            self.metrics['val_best'][key] = vals[best_epoch]

    def log_train(self, log_data):
        for logger in self.train_loggers:
            logger.log(**log_data)

        for k, v in log_data.items():
            if k in self.train_keys:
                self.metrics['train'][k].append(v)

    def log_val(self, log_data):
        for logger in self.val_loggers:
            logger.log(**log_data)

        for k, v in log_data.items():
            if k in self.val_keys:
                self.metrics['val'][k].append(v)

    def log_test(self, log_data):
        for k, v in log_data.items():
            self.metrics['test'][k] = v

    def log_user_val_metrics(self, log_data: dict, u_idx2id: dict, code='val', key='None'):
        """
        Stores individual user metrics in dictionary, also maps user index to ID

        Input:
            log_data: contains metrics for one batch of users, indexed by u_idx
            u_idx2id: mapping from working u_idx to real user ID
            code: indicate dataset, i.e. val or test
            key: additional key to create nested dict, e.g. order embedding


        Output:

            { 'user_id':
                    {'key': {'auc': 0.8, 'mrr': 0.6, 'ndcg5': 0.7}
            }

        """
        for u_idx, vals in log_data.items():
            # map user idx to ID
            u_id = u_idx2id[u_idx] if u_idx2id is not None else u_idx
            self.user_metrics[code][u_id][key] = vals


    def log_grad_flow_report(self, report: dict, iter: int):
        self.grad_reports[iter].append(report)

    def save_metric_dicts(self, export_path=None):

        with open(os.path.join(export_path, 'logs', 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=4)

        with open(os.path.join(export_path, 'logs', 'user_metrics.json'), 'w') as f:
            json.dump(self.user_metrics, f, indent=4)

        if len(self.grad_reports.keys()) > 0:
            with open(os.path.join(export_path, 'logs', 'grad_reports.json'), 'w') as fout:
                json.dump(self.grad_reports, fout, indent=4)

    def get_metrics_at_epoch(self, metrics, epoch):
        return {key: vals[epoch] for key, vals in metrics.items()}

    def print_final(self, rel_epochs: list, add_info: dict, mul_lines=False, metric_ks=[5, 10]):

        self.rel_metrics = ['loss'] + ['AUC'] + ['MRR'] + \
                          ['NDCG@%d' % k for k in metric_ks[:2]] + \
                          ['Recall@%d' % k for k in metric_ks[:2]]

        for key, val in add_info.items():
            print("{}: {}".format(key, val))

        print("\n### Val metrics ###")
        for ep in rel_epochs:
            m = {'loss': self.metrics['train']['loss'][ep]}
            m.update(self.get_metrics_at_epoch(self.metrics['val'], ep))
            if -1 == ep:
                ep = len(self.metrics['train']['loss'])
            print("Epoch {}: ".format(ep+1) + self.format_metric_descr(m, mul_lines=mul_lines))

        print("\n### Best Val metrics ###")
        print("Best epoch {}: ".format(self.best_model_logger.best_epoch) +
                self.format_metric_descr(self.metrics['val_best'], mul_lines))

        print("### Test metrics ###")
        print(self.format_metric_descr(self.metrics['test'], mul_lines))

    def format_metric_descr(self, metrics, mul_lines=False):
        res_str = ""
        for key, val in sorted(metrics.items()):
            if key in self.rel_metrics:
                res_str += "{}: {:.3f}".format(key, val)
                if mul_lines:
                    res_str += "\n "
                else:
                    res_str += ", "

        res_str = res_str.replace('NDCG', 'N').replace('Recall', 'R')

        return res_str


class AbstractBaseLogger(metaclass=ABCMeta):
    @abstractmethod
    def log(self, *args, **kwargs):
        raise NotImplementedError

    def complete(self, *args, **kwargs):
        pass


class RecentModelLogger(AbstractBaseLogger):
    def __init__(self, checkpoint_path, filename='checkpoint-recent.pth'):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        self.recent_epoch = None
        self.filename = filename

    def log(self, *args, **kwargs):
        epoch = kwargs['epoch']

        if self.recent_epoch != epoch:
            self.recent_epoch = epoch
            state_dict = kwargs['state_dict']
            state_dict['epoch'] = kwargs['epoch']
            save_state_dict(state_dict, self.checkpoint_path, self.filename)

    def complete(self, *args, **kwargs):
        save_state_dict(kwargs['state_dict'], self.checkpoint_path, self.filename + '.final')


class BestModelLogger(AbstractBaseLogger):
    def __init__(self, checkpoint_path, metric_key='mean_iou', filename='best_auc_model.pth'):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        self.best_metric = 0.
        self.best_epoch = 0
        self.metric_key = metric_key
        self.filename = filename

    def log(self, *args, **kwargs):
        current_metric = kwargs[self.metric_key]
        if self.best_metric < current_metric:
            print("Update Best {} Model at {}".format(self.metric_key, kwargs['epoch']))
            self.best_metric = current_metric
            self.best_epoch = kwargs['epoch']
            save_state_dict(kwargs['state_dict'], self.checkpoint_path, self.filename)


class MetricGraphScalar(AbstractBaseLogger):
    def __init__(self, writer, key='train_loss', graph_name='Train Loss', group_name='metric'):
        self.key = key
        self.graph_label = graph_name
        self.group_name = group_name
        self.writer = writer

    def log(self, *args, **kwargs):
        if self.key in kwargs:
            self.writer.add_scalar(self.group_name + '/' + self.graph_label, kwargs[self.key], kwargs['accum_iter']) # epoch
        else:
            self.writer.add_scalar(self.group_name + '/' + self.graph_label, 0, kwargs['accum_iter'])

    def complete(self, *args, **kwargs):
        self.writer.close()

class MetricGraphScalars(MetricGraphScalar):

    def log(self, *args, **kwargs):
        if self.key in kwargs:
            # uses key-value pairs
            self.writer.add_scalars(self.group_name + '/' + self.graph_label, kwargs[self.key], kwargs['accum_iter'])
        else:
            self.writer.add_scalars(self.group_name + '/' + self.graph_label, 0, kwargs['accum_iter'])

class HistogramLogger(AbstractBaseLogger):
    def __init__(self, writer, key='grad_histo', graph_name='GradFlow', group_name='metric'):
        self.key = key
        self.graph_label = graph_name
        self.group_name = group_name
        self.writer = writer
    #
    # with SummaryWriter(log_dir=log_dir, comment="GradTest", flush_secs=30) as writer:
    #     # ... your learning loop
    #     _limits = np.array([float(i) for i in range(len(gradmean))])
    #     _num = len(gradmean)
    #
    # def log(self, *args, **kwargs):
    #     self.writer.add_histogram_raw(tag=netname + "/abs_mean", min=0.0, max=0.3, num=_num,
    #                              sum=gradmean.sum(), sum_squares=np.power(gradmean, 2).sum(), bucket_limits=_limits,
    #                              bucket_counts=gradmean, global_step=global_step)
    #     # where gradmean is np.abs(p.grad.clone().detach().cpu().numpy()).mean()
    #     # _limits is the x axis, the layers
    #     # and
    #     _mean = {}
    #     for i, name in enumerate(layers):
    #         _mean[name] = gradmean[i]
    #     writer.add_scalars(netname + "/abs_mean", _mean, global_step=global_step
