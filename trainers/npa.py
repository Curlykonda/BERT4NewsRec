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

    def calculate_metrics(self, batch):

        #hist, cands, u_idx = batch['input']
        lbls = batch['lbls']

        logits = self.model(**batch['input']) # (L x N_c)

        scores = nn.functional.softmax(logits, dim=1)

        # select scores for the article indices of candidates
        #scores = scores.gather(1, cands)  # (B x n_candidates)
        # labels: (B x N_c)
        metrics = calc_recalls_and_ndcgs_for_ks(scores, lbls, self.metric_ks)
        metrics.update(calc_auc_and_mrr(scores, lbls))

        return metrics