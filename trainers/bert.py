from .base import AbstractTrainer
from .utils_metrics import calc_recalls_and_ndcgs_for_ks, calc_auc_and_mrr

import torch
import torch.nn as nn


class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        seqs, labels = batch
        logits = self.model(seqs)  # B x T x V

        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch):
        seqs, candidates, labels = batch
        scores = self.model(seqs)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C

        metrics = calc_recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics

class BERT4NewsCategoricalTrainer(BERTTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):

        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.CrossEntropyLoss(reduction='mean')

    @classmethod
    def code(cls):
        return 'bert_news_ce'

    def calculate_loss(self, batch):

        if self.args.incl_time_stamp:
            seqs, mask, cands, labels, time_stamps = batch
            logits = self.model([seqs, time_stamps], mask, cands)
        else:
            seqs, mask, cands, labels = batch
            time_stamps = None
            logits = self.model(seqs, mask, cands) # (B*T) x N_c

        # labels have to indicate which candidate is the correct one
        # convert target item index to categorical label C = [0, N_c]
        lbls = labels[labels > 0]
        c = cands[labels > 0]
        categorical_lbls = (c == lbls.unsqueeze(1).repeat(1, c.size(1))).nonzero()[:, -1]

        rel_logits = logits[labels.view(-1) > 0]
        # calculate a separate loss for each class label per observation and sum the result.
        loss = self.ce(rel_logits, categorical_lbls)
        # note: NPA approach only computes NLL for positive class -> select only logits for positive class?
        return loss

    def calculate_metrics(self, batch):

        input = batch['input']
        lbls = batch['lbls']
        logits = self.model(**batch['input']) # (B x N_c)
        # else:
        #     seqs, mask, cands, labels = batch
        #     time_stamps = None
        #     logits = self.model(seqs, mask, cands) # (B x N_c)

        # TODO: check scores
        scores = nn.functional.softmax(logits, dim=1)

        # select scores for the article indices of candidates
        #scores = scores.gather(1, cands)  # (B x n_candidates)
        # labels: (B x N_c)
        metrics = calc_recalls_and_ndcgs_for_ks(scores, lbls, self.metric_ks)
        metrics.update(calc_auc_and_mrr(scores, lbls))

        return metrics


class Bert4NewsDistanceTrainer(BERTTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root, dist_func='cos'):

        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)

        self.dist_func_string = dist_func
        self.dist_func = None
        self.y_eval = False # indicates, if loss function requires additional target vector y
        self.loss_func = self.get_loss_func()


    @classmethod
    def code(cls):
        return 'bert_news_dist'

    def get_loss_func(self):
        if 'cos' == self.dist_func_string:
            # cos(x1, x2) = x1 * x2 / |x1|*|x2|  \in [-1, 1]
            # loss(x,y) = 1 - cos(x1,x2), if y==1
            # loss is zero when x1 = x2 -> cos = 1
            self.y_eval = True
            self.dist_func = nn.CosineSimilarity(dim=0, eps=1e-6)
            return nn.CosineEmbeddingLoss()
            #crit(x1.unsqueeze(1), x2.unsqueeze(1), y)
        elif 'mse' == self.dist_func_string:
            return nn.MSELoss()
        elif 'hinge' == self.dist_func_string:
            return nn.HingeEmbeddingLoss()
        elif 'mrank' == self.dist_func_string:
            nn.MarginRankingLoss()
        else:
            raise ValueError("{} is not a valid distance function!".format(self.dist_func))


    def calculate_loss(self, batch):
        seqs, mask, cands, labels = batch # (B x T)

        # cands = batch['cands'] # (B x L_m x N_cands)
        # labels = batch['lbls'] # (B x T)

        # select relevant candidates to reduce number of forward passes
        # (B x T x N_cands) -> (L_m x N_cands)
        rel_cands = cands[labels > 0]
        n_cands = cands.shape[2]

        # forward pass
        pred_embs, cand_embs = self.model(seqs, mask, rel_cands)
        # (B x T x D_a), (L_m x D_a)

        # gather relevant embedding for the masking positions
        # L_m := # of masked positions
        pred_embs = pred_embs[labels > 0]  # L_m x D_a
        pred_embs = pred_embs.unsqueeze(1).repeat(1, n_cands, 1) # repeat predicted embedding for n_cands

        # flatten
        cand_embs = cand_embs.view(-1, cand_embs.shape[-1]) # (L_m*N) x D_a
        pred_embs = pred_embs.view(-1, cand_embs.shape[-1]) # (L_m*N) x D_a
        labels = labels.view(-1)  # B*T

        # transpose
        # pred_embs = pred_embs.transpose(0, 1)
        # cand_embs = cand_embs.transpose(0, 1)

        assert pred_embs.size(0) == cand_embs.size(0)

        rel_labels = labels[labels > 0] # L_m
        #cand_labels := (B x T x N_c)
        # -> (L_m x N_c) -> (L_m * N_c)

        if self.y_eval:
            # construct target vector for distance loss
            # y==1 -> vectors should be similar
            # y==-1 -> vectors should be DIS-similar
            # y = (L_m x n_candidates)
            y = (-torch.ones_like(rel_cands))
            ## create mask that indicates, which candidate is the target
            mask = (rel_cands == rel_labels.unsqueeze(1).repeat(1, n_cands))
            y = y.masked_fill(mask, 1).view(-1)

            # compute distance loss
            # Maximise similarity betw. target & pred, while minimising pred and neg. samples
            # Note: CosineEmbeddingLoss 'prefers' vectors of shape (D_a x B) so perhaps transpose(0,1)
            # (L_m * N_c) x D_a
            loss = self.loss_func(pred_embs, cand_embs, y)
        else:
            loss = self.loss_func(pred_embs, cand_embs)

        return loss

    def calculate_metrics(self, batch):
        seqs, mask, cands, labels = batch
        # seqs, mask = batch['input']  # (B x T)
        # cands = batch['cands']  # (B x N_cands)
        # labels = batch['lbls']  # (B x T)
        n_cands = cands.shape[1]

        # forward pass
        pred_embs, cand_embs = self.model(seqs, mask, cands)

        # select masking position
        pred_embs = pred_embs[:, -1, :] # # (B x L_hist x D_a) -> (B x D_a)
        pred_embs = pred_embs.unsqueeze(1).repeat(1, n_cands, 1) # repeat predicted embedding for n_cands

        # flatten
        cand_embs = cand_embs.view(-1, cand_embs.shape[-1]) # (B*N) x D_a
        pred_embs = pred_embs.view(-1, cand_embs.shape[-1]) # (B*N) x D_a

        # transpose
        pred_embs = pred_embs.transpose(0, 1)
        cand_embs = cand_embs.transpose(0, 1)

        # compute distance scores
        with torch.no_grad():
            #y = torch.ones(cand_embs.size(0), cand_embs.size(1))
            # note that the loss internally applies 'mean' reduction so we need simple distance fucntion
            dist_scores = self.dist_func(pred_embs, cand_embs)

        # Note: inside this function, scores are inverted. check if aligns with distance function
        metrics = calc_recalls_and_ndcgs_for_ks(dist_scores.view(-1, n_cands), labels.view(-1, n_cands), self.metric_ks)
        metrics.update(calc_auc_and_mrr(dist_scores.view(-1, n_cands), labels.view(-1, n_cands)))
        return metrics