from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import OneHotEncoder

from .base import ExtendedTrainer
from .utils_metrics import calc_recalls_and_ndcgs_for_ks, calc_auc_and_mrr

class NpaTrainer(ExtendedTrainer):
    def __init__(self, args, model, dataloader, export_root):

        super().__init__(args, model, dataloader, export_root)
        self.ce = nn.CrossEntropyLoss(reduction='mean')

    @classmethod
    def code(cls):
        return 'npa'

    def calculate_loss(self, batch):

        lbls = batch['lbls']
        # forward pass
        logits = self.model(**batch['input']) # (L x N_c)

        # categorical labels indicate which candidate is the correct one C = [0, N_c]
        # for positions where label is unequal -1
        #lbls = cat_labels[cat_labels != -1]

        # calculate a separate loss for each class label per observation and sum the result.
        loss = self.ce(logits, lbls.argmax(dim=1))

        ### calc metrics ###
        # # one-hot encode lbls
        # enc = OneHotEncoder(sparse=False)
        # enc.fit(np.array(range(logits.shape[1])).reshape(-1, 1))
        # oh_lbls = torch.LongTensor(enc.transform(lbls.cpu().reshape(len(lbls), 1)))

        # softmax probabilities
        scores = nn.functional.softmax(logits, dim=1)
        scores = scores.cpu().detach()

        metrics = calc_recalls_and_ndcgs_for_ks(scores, lbls.cpu(), self.metric_ks)
        metrics.update(calc_auc_and_mrr(scores, lbls.cpu()))

        return loss, metrics

    def calculate_metrics(self, batch, avg_metrics=True):
        """
        Performs forward pass through NPA model
         and computes metrics based on model output (predictions)

        Input:
            batch -> {input: {hist: Tensor, u_id: Tensor, ..} , lbls: Tensor}

        Output:

        if "avg_metrics":
            return metrics -> {metric_key: [avg_val]}
        else:
            return user_metrics -> {u_idx: {metric_key: val}}

        """

        #hist, cands, u_idx = batch['input']
        lbls = batch['lbls']

        logits = self.model(**batch['input']) # (L x N_c)

        scores = nn.functional.softmax(logits, dim=1)

        # select scores for the article indices of candidates
        #scores = scores.gather(1, cands)  # (B x n_candidates)
        # labels: (B x N_c)
        metrics = calc_recalls_and_ndcgs_for_ks(scores, lbls, self.metric_ks, avg_metrics)
        metrics.update(calc_auc_and_mrr(scores, lbls, avg_metrics))

        if avg_metrics:
            return metrics  # metric_key: [avg_val]

        else:
            # update individual user metrics
            # { u_idx: {'auc': 0.8, 'mrr': 0.4}}
            user_metrics = defaultdict(dict)

            # get user indices
            u_indices = batch['input']['u_idx'][:, 0].cpu().numpy()

            for i, u_idx in enumerate(u_indices):
                for key, vals in metrics.items():
                    user_metrics[u_idx][key] = vals[i]

                user_metrics[u_idx]['scores'] = scores[i, :].cpu().numpy()

            return user_metrics  # {u_idx: {metric_key: val}}

class NpaModTrainer(ExtendedTrainer):
    """
    Modified NPA trainer to align training with BERT models
    """
    def __init__(self, args, model, dataloader, export_root):
        super().__init__(args, model, dataloader, export_root)
        self.ce = nn.CrossEntropyLoss(reduction='mean')

    @classmethod
    def code(cls):
        return 'npa_mod'

    def calculate_loss(self, batch):
        lbls = batch['lbls']

        # cands: (N_T x N_C)
        # for each target, create instance

        # forward pass
        logits = self.model(cand_mask=lbls, **batch['input'])  # (L x N_c)

        # categorical labels indicate which candidate is the correct one C = [0, N_c]
        # for positions where label is unequal -1
        rel_lbls = lbls[lbls != -1]

        # calculate a separate loss for each class label per observation and sum the result.
        loss = self.ce(logits, rel_lbls)

        ### calc metrics ###
        oh_lbls = self.one_hot_encode_lbls(rel_lbls, n_classes=logits.shape[1])

        # softmax probabilities
        scores = nn.functional.softmax(logits, dim=1)
        scores = scores.cpu().detach()

        metrics = calc_recalls_and_ndcgs_for_ks(scores, oh_lbls, self.metric_ks)
        metrics.update(calc_auc_and_mrr(scores, oh_lbls))

        return loss, metrics

    def calculate_metrics(self, batch, avg_metrics=True):
        # hist, cands, u_idx = batch['input']
        lbls = batch['lbls']

        logits = self.model(cand_mask=None, **batch['input'])  # (L x N_c)

        scores = nn.functional.softmax(logits, dim=1)

        # select scores for the article indices of candidates
        # scores = scores.gather(1, cands)  # (B x n_candidates)
        # labels: (B x N_c)
        metrics = calc_recalls_and_ndcgs_for_ks(scores, lbls, self.metric_ks, avg_metrics)
        metrics.update(calc_auc_and_mrr(scores, lbls, avg_metrics))

        if avg_metrics:
            return metrics # metric_key: [avg_val]

        else:
            # update individual user metrics
            # { u_idx: {'auc': 0.8, 'mrr': 0.4}}
            user_metrics = defaultdict(dict)

            # get user indices
            u_indices = batch['input']['u_idx'][:, 0].cpu().numpy()

            for i, u_idx in enumerate(u_indices):
                for key, vals in metrics.items():
                    user_metrics[u_idx][key] = vals[i]

                user_metrics[u_idx]['scores'] = scores[i, :].cpu().numpy()

            return user_metrics # {u_idx: {metric_key: val}}