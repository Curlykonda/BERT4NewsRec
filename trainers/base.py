import numpy as np
from sklearn.preprocessing import OneHotEncoder

from loggers import *
from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from utils import AverageMeterSet, get_hyper_params
from source.utils import get_grad_flow_report

from transformers import get_linear_schedule_with_warmup

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import json
from abc import *
from pathlib import Path
import time

class AbstractTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.is_parallel = args.num_gpu > 1
        if self.is_parallel:
            self.model = nn.DataParallel(self.model)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = self._create_optimizer()
        self.init_lr = args.lr
        if args.lr_schedule and args.decay_step is not None:
            # Decays the learning rate of each parameter group by gamma every step_size epochs
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        self.num_epochs = args.num_epochs
        self.batch_size = args.train_batch_size
        self.max_iters = args.num_epochs * args.log_period_as_iter
        self.num_training_steps = self.max_iters // self.batch_size

        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric

        self.export_root = export_root
        self.writer, self.train_loggers, self.val_loggers = self._create_loggers()
        self.add_extra_loggers()
        self.logger_service = LoggerService(self.train_loggers, self.val_loggers)
        self.log_period_as_iter = args.log_period_as_iter

    @abstractmethod
    def add_extra_loggers(self):
        pass

    @abstractmethod
    def log_extra_train_info(self, log_data):
        pass

    @abstractmethod
    def log_extra_val_info(self, log_data):
        pass

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def calculate_loss(self, batch):
        pass

    @abstractmethod
    def calculate_metrics(self, batch):
        pass

    def train(self):
        accum_iter = 0
        #self.validate(0, accum_iter)
        print("\n > Start training")
        t0 = time.time()
        for epoch in range(self.num_epochs):
            t1 = time.time()
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            t2 = time.time()
            print("> Train epoch {} in {:.3f} min".format(epoch+1, (t2-t1)/60))
            self.validate(epoch, accum_iter)
            t3 = time.time()
            print("> Val epoch in {:.3f} min".format((t3 - t2) / 60))

            if self._reached_max_iterations(accum_iter):
                break

        print("Performed {} iterations in {} epochs (/{})".format(accum_iter, epoch, self.num_epochs))

        self.writer.add_hparams(hparam_dict=get_hyper_params(self.args), metric_dict={"accum_iter": accum_iter})
        self.logger_service.complete({'state_dict': (self._create_state_dict()),})

        #self.writer.close()
        print("\n >> Run completed in {:.1f} h \n".format((time.time() - t0) / 3600))

    def train_one_epoch(self, epoch, accum_iter):
        self.model.train()

        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.train_loader)

        for batch_idx, batch in enumerate(tqdm_dataloader):

            batch = self.batch_to_device(batch)

            # forward pass
            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)

            # backward pass
            loss.backward()
            self.optimizer.step()

            # update metrics
            average_meter_set.update('loss', loss.item())
            average_meter_set.update('lr', self.optimizer.defaults['lr'])

            tqdm_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch + 1, average_meter_set['loss'].avg))
            accum_iter += self.batch_size

            if self.args.local and batch_idx == 20:
                break

        tqdm_dataloader.set_description('Logging to Tensorboard')
        log_data = {
            'state_dict': (self._create_state_dict()),
            'epoch': epoch + 1,
            'accum_iter': accum_iter,
        }
        log_data.update(average_meter_set.averages())
        self.log_extra_train_info(log_data)
        self.logger_service.log_train(log_data)

        # adapt learning rate
        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()
            if epoch % self.lr_scheduler.step_size == 0:
                print(self.optimizer.defaults['lr'])


        return accum_iter

    def validate(self, epoch, accum_iter):
        self.model.eval()

        average_meter_set = self.eval_one_epoch(self.val_loader, epoch)

        log_data = {
            'state_dict': (self._create_state_dict()),
            'epoch': epoch+1,
            'accum_iter': accum_iter,
        }
        log_data.update(average_meter_set.averages())
        self.log_extra_val_info(log_data)
        self.logger_service.log_val(log_data)

    def test(self):
        print('Test best model with test set!')

        best_model = torch.load(os.path.join(self.export_root, 'models', 'best_acc_model.pth')).get('model_state_dict')
        self.model.load_state_dict(best_model)
        self.model.eval()

        average_meter_set = self.eval_one_epoch(self.test_loader)

        average_metrics = average_meter_set.averages()

        self.logger_service.log_test(average_metrics)
        self.logger_service.save_metric_dicts(self.export_root)

        with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
            json.dump(average_metrics, f, indent=4)

        print(average_metrics)
        print("\n")
        print(self.export_root)
        print("############################################\n")

    def eval_one_epoch(self, eval_loader, epoch=None):

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(eval_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = self.batch_to_device(batch)

                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)

                if self.args.local and batch_idx > 20:
                    break

                # if batch_idx % 10 == 0 and batch_idx > 0:
                #     descr = get_metric_descr(average_meter_set, self.metric_ks)
                #     tqdm_dataloader.set_description(descr)

        descr = get_metric_descr(average_meter_set, self.metric_ks)
        tqdm_dataloader.set_description(descr)
        # if epoch is not None:
        #     print("\n Epoch {} avg.: {}".format(epoch+1, descr))
        # else:
        #     print("\n")

        return average_meter_set


    def batch_to_device(self, batch):
        if isinstance(batch, dict):
            device_dict = {}
            for key, val in batch.items():
                if isinstance(val, list):
                    device_dict[key] = [elem.to(self.device) for elem in val]
                elif isinstance(val, dict):
                    device_dict[key] = {k: v.to(self.device) for k, v in val.items()}
                else:
                    device_dict[key] = val.to(self.device)

            batch = device_dict
            # batch = {key: x.to(self.device) for key, x in batch.items() if not isinstance(x, list) else key: [elem.to(self.device) for elem in x]}
        else:
            batch = [x.to(self.device) for x in batch]

        return batch

    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError

    def _create_loggers(self):
        root = Path(self.export_root)
        writer = SummaryWriter(root.joinpath('logs'))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphScalar(writer, key='epoch', graph_name='Epoch', group_name='Train'),
            MetricGraphScalar(writer, key='loss', graph_name='Loss', group_name='Train'),
            MetricGraphScalar(writer, key='lr', graph_name='Learning Rate', group_name='Train'),
        ]

        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphScalar(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphScalar(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))

        # val_loggers.append(MetricGraphPrinter(writer, key='AUC', graph_name='AUC', group_name='Validation'))
        # val_loggers.append(MetricGraphPrinter(writer, key='MRR', graph_name='MRR', group_name='Validation'))

        val_loggers.append(RecentModelLogger(model_checkpoint))
        val_loggers.append(BestModelLogger(model_checkpoint, metric_key=self.best_metric))
        return writer, train_loggers, val_loggers

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }

    def _needs_to_log(self, global_step):
        return global_step % self.log_period_as_iter < self.args.train_batch_size # and global_step != 0

    def _reached_max_iterations(self, global_step):
        return self.max_iters <= global_step

    def print_final(self):

        # get relevant hyper params

        # get loss & metrics at certain epochs

        # get test metrics

        # print string

        pass

class ExtendedTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        self.log_grads = args.log_grads
        self.grad_clip_val = args.grad_clip_val

        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        if self.args.lr_schedule:
            self.warmup_ratio = args.warmup_ratio if args.warmup_ratio is not None else 0
            # Create schedule with learning rate that decreases linearly
            # from the initial lr set in the optimizer to 0,
            # after a warmup period during which it increases linearly
            # from 0 to the initial lr set in the optimizer.
            self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                                num_warmup_steps=self.warmup_ratio * self.num_training_steps,
                                                                num_training_steps=self.num_training_steps)


    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def train(self):
        global_step = 0
        self.validate(0, global_step)
        print("\n > Start training")
        t0 = time.time()
        for epoch in range(self.num_epochs):
            t1 = time.time()
            global_step = self.train_one_epoch(epoch, global_step)
            t2 = time.time()
            print("> Train epoch {} in {:.3f} min \n".format(epoch+1, (t2-t1)/60))

            # t3 = time.time()
            # print("> Val epoch in {:.3f} min".format((t3 - t2) / 60))

            if self._reached_max_iterations(global_step):
                break

        print("Performed {} iterations in {} epochs (/{})".format(global_step, epoch+1, self.num_epochs))

        self.writer.add_hparams(hparam_dict=get_hyper_params(self.args), metric_dict={"accum_iter": global_step})
        self.logger_service.complete({'state_dict': (self._create_state_dict()),})

        #self.writer.close()
        self.total_train_time = "{:.1f} h".format((time.time() - t0) / 3600)
        print("\n >> Run completed in {} \n".format(self.total_train_time))


    def train_one_epoch(self, epoch, global_step):
        self.model.train()

        average_meter_set = AverageMeterSet()

        for batch_idx, batch in enumerate(self.train_loader):

            batch = self.batch_to_device(batch)

            # forward pass
            self.optimizer.zero_grad()
            loss, metrics_train = self.calculate_loss(batch)

            # backward pass
            loss.backward()

            if self.grad_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_val)

            self.optimizer.step()

            # update metrics
            average_meter_set.update('loss', loss.item())
            average_meter_set.update('lr', self._get_cur_lr())
            for k, v in metrics_train.items():
                average_meter_set.update(k, v)

            # adapt learning rate
            if self.args.lr_schedule:
                self.lr_scheduler.step()

            global_step += self.batch_size

            if self._needs_to_log(global_step):
                print('Epoch {}, loss {:.3f},  lr {:.5f}'.format(epoch + 1, average_meter_set['loss'].avg, average_meter_set['lr'].val))
                print('Logging to Tensorboard')
                #tqdm_dataloader.set_description('Logging to Tensorboard')
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch + 1,
                    'accum_iter': global_step,
                }
                log_data.update(average_meter_set.averages())
                self.log_extra_train_info(log_data)

                if self.log_grads:
                    # get grad_flow for abs_mean & sign_max
                    avg_grads, max_grads = get_grad_flow_report(self.model.named_parameters())
                    log_data['grad_abs_mean'] = avg_grads
                    log_data['grad_sign_max'] = max_grads

                    # add grad_flow to respective logger
                    #self.logger_service.log_grad_flow(, accum_iter)

                self.logger_service.log_train(log_data)

                self.validate(epoch, global_step)

                if self._reached_max_iterations(global_step):
                    return global_step
                else:
                    self.model.train()

            #self.model.zero_grad()

            # break condition for local debugging
            if self.args.local and batch_idx > 20:
                break

        return global_step

    def eval_one_epoch(self, eval_loader, epoch=None):

        average_meter_set = AverageMeterSet()

        with torch.no_grad():

            for batch_idx, batch in enumerate(eval_loader):
                batch = self.batch_to_device(batch)

                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)

                if self.args.local and batch_idx > 20:
                    break

                # if batch_idx % 10 == 0 and batch_idx > 0:
                #     descr = get_metric_descr(average_meter_set, self.metric_ks)
                #     tqdm_dataloader.set_description(descr)

        descr = get_metric_descr(average_meter_set, self.metric_ks)
        print(descr)

        return average_meter_set

    def test(self):
        print('Test best model with test set!')

        best_model = torch.load(os.path.join(self.export_root, 'models', 'best_acc_model.pth')).get('model_state_dict')
        self.model.load_state_dict(best_model)
        self.model.eval()

        average_meter_set = self.eval_one_epoch(self.test_loader)

        average_metrics = average_meter_set.averages()

        self.logger_service.log_test(average_metrics)
        self.logger_service.save_metric_dicts(self.export_root)

        with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
            json.dump(average_metrics, f, indent=4)

        add_info = {'exp_dir': self.export_root,
                    'n_params': self.args.n_params,
                    'total_train_time': self.total_train_time}
        self.logger_service.print_final(rel_epochs=[0, -1], add_info=add_info)
        print("\n############################################\n")

    def _get_cur_lr(self):
        if self.args.lr_schedule:
            return self.lr_scheduler.get_last_lr()[0]
        else:
            return self.init_lr

    def _create_loggers(self):
        root = Path(self.export_root)
        writer = SummaryWriter(root.joinpath('logs'))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphScalar(writer, key='epoch', graph_name='Epoch', group_name='Train'),
            MetricGraphScalar(writer, key='loss', graph_name='Loss', group_name='Train'),
            MetricGraphScalar(writer, key='lr', graph_name='Learning Rate', group_name='Train'),
            MetricGraphScalar(writer, key='AUC', graph_name='AUC', group_name='Train'),
            MetricGraphScalar(writer, key='MRR', graph_name='MRR', group_name='Train'),
        ]

        if self.log_grads:
            train_loggers.append(MetricGraphScalars(writer, key='grad_abs_mean', graph_name='GradAbsMean', group_name='GradFlow'))
            train_loggers.append(MetricGraphScalars(writer, key='grad_sign_max', graph_name='GradSignMax', group_name='GradFlow'))

        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphScalar(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphScalar(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))

            train_loggers.append(
                MetricGraphScalar(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Train'))
            train_loggers.append(
                MetricGraphScalar(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))


        val_loggers.append(MetricGraphScalar(writer, key='AUC', graph_name='AUC', group_name='Validation'))
        val_loggers.append(MetricGraphScalar(writer, key='MRR', graph_name='MRR', group_name='Validation'))

        val_loggers.append(RecentModelLogger(model_checkpoint))
        val_loggers.append(BestModelLogger(model_checkpoint, metric_key=self.best_metric))
        return writer, train_loggers, val_loggers

    def one_hot_encode_lbls(self, cat_lbls, n_classes):
        # # one-hot encode lbls
        enc = OneHotEncoder(sparse=False)
        enc.fit(np.array(range(n_classes)).reshape(-1, 1))
        oh_lbls = torch.LongTensor(enc.transform(cat_lbls.cpu().reshape(len(cat_lbls), 1)))

        return oh_lbls

def get_metric_descr(metric_set, metric_ks=[5, 10]):
    description_metrics = ['AUC'] + ['MRR'] + \
                          ['NDCG@%d' % k for k in metric_ks[:3]] + \
                          ['Recall@%d' % k for k in metric_ks[:3]]
    description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
    description = description.replace('NDCG', 'N').replace('Recall', 'R')
    description = description.format(*(metric_set[k].avg for k in description_metrics))

    return description