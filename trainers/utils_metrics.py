import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def recall(scores, labels, k):
    scores = scores
    labels = labels
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / torch.min(torch.Tensor([k]).to(hit.device), labels.sum(1).float())).mean().cpu().item()


def ndcg(scores, labels, k=10):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()

def mrr_score(scores, labels, avg=True):
    """
    Calculate Mean Reciprocal Rank for a set of score-target pairs

    """
    scores = scores.float().cpu()
    labels = labels.float().cpu() # (B x N_c)

    rank = scores.argsort(dim=1, descending=True)
    hits = labels.gather(1, rank)
    recpr_rank = hits / (torch.arange(hits.size(1)) + 1)
    mrr = torch.sum(recpr_rank, dim=1) / torch.sum(hits, dim=1)

    if avg:
        return torch.mean(mrr).item()
    else:
        return list(mrr.cpu().numpy())

    # np variant:
    # order = np.argsort(y_score)[::-1] #Returns the indices that would sort an array
    # y_true = np.take(y_true, order)
    # rr_score = y_true / (np.arange(len(y_true)) + 1)
    # return np.sum(rr_score) / np.sum(y_true)

def calc_auc_and_mrr(scores, labels, avg=True):
    scores = scores.cpu()
    labels = labels.cpu()

    mrr = mrr_score(scores, labels, avg)
    auc = calc_roc_auc_tensor(scores, labels, avg)

    return {"AUC": auc, "MRR": mrr}

def calc_roc_auc_tensor(scores, labels, avg=True):

    auc = list(map(lambda y, x: roc_auc_score(y, x), labels, scores))
    if avg:
        return np.mean(auc)
    else:
        return auc

def calc_recalls_and_ndcgs_for_ks(scores, labels, ks, avg=True):
    """
    scores:
    labels:
    ks (list): list containing the ranks for which to compute metric, e.g. NDCG@k

    out:
        metrics (dict): {metric@k: val}
    """
    metrics = {}

    # scores = scores
    # labels = labels
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = scores.argsort(dim=1, descending=True) # largest score comes first; larger means better
    cut = rank
    for k in sorted(ks, reverse=True):
        if k <= scores.shape[1]:

            cut = cut[:, :k]
            hits = labels_float.gather(1, cut)

            recall = (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device),
                                         labels.sum(1).float()))

            position = torch.arange(2, 2+k)
            weights = 1 / torch.log2(position.float())
            dcg = (hits * weights.to(hits.device)).sum(1)
            idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
            ndcg = (dcg / idcg)

            if avg:
                recall = recall.mean().cpu().item()
                ndcg = ndcg.mean().cpu().item()

            else:
                recall = list(recall.cpu().numpy())
                ndcg = list(ndcg.cpu().numpy())

            metrics['Recall@%d' % k] = recall
            metrics['NDCG@%d' % k] = ndcg

    return metrics