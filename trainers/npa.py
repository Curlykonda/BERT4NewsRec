from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import OneHotEncoder

from .base import ExtendedTrainer

from .utils_metrics import calc_recalls_and_ndcgs_for_ks, calc_auc_and_mrr

class NpaTrainer(ExtendedTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):

        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.CrossEntropyLoss(reduction='mean')

    @classmethod
    def code(cls):
        return 'npa'

    def calculate_loss(self, batch):

        cat_labels = batch['lbls']
        self.model.set_train_mode(True)
        # forward pass
        logits = self.model(**batch['input']) # L x N_c

        # categorical labels indicate which candidate is the correct one C = [0, N_c]
        # for positions where label is unequal -1
        lbls = cat_labels[cat_labels != -1]

        # calculate a separate loss for each class label per observation and sum the result.
        loss = self.ce(logits, lbls)

        ### calc metrics ###
        # one-hot encode lbls
        enc = OneHotEncoder(sparse=False)
        enc.fit(np.array(range(logits.shape[1])).reshape(-1, 1))
        oh_lbls = torch.LongTensor(enc.transform(lbls.cpu().reshape(len(lbls), 1)))
        scores = nn.functional.softmax(logits, dim=1)

        scores = scores.cpu().detach()

        metrics = calc_recalls_and_ndcgs_for_ks(scores, oh_lbls, self.metric_ks)
        metrics.update(calc_auc_and_mrr(scores, oh_lbls))

        return loss, metrics

    def calculate_metrics(self, batch):

        input = batch['input'].items()
        lbls = batch['lbls']
        self.model.set_train_mode(False)
        logits = self.model(None, **batch['input']) # (L x N_c)

        scores = nn.functional.softmax(logits, dim=1)

        # select scores for the article indices of candidates
        #scores = scores.gather(1, cands)  # (B x n_candidates)
        # labels: (B x N_c)
        metrics = calc_recalls_and_ndcgs_for_ks(scores, lbls, self.metric_ks)
        metrics.update(calc_auc_and_mrr(scores, lbls))

        return metrics