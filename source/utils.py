import datetime
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from matplotlib.lines import Line2D
from torch.backends import cudnn
import arrow

import matplotlib.pyplot as plt

from source.modules.activation_funcs import GELU




def get_grad_flow(named_parameters):
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())

    return {n: [ave_grads[i], max_grads[i]] for i, n in enumerate(layers)}

def get_grad_flow_report(named_parameters):
    avg_grads = {}
    max_grads = {}

    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            n = prettify_layer_names(n)
            avg_grads[n] = p.grad.abs().mean().cpu().item()
            max_grads[n] = p.grad.abs().max().cpu().item()

    return avg_grads, max_grads

def prettify_layer_names(l_name: str):
    excl_layer_parts = ["weight", "embedding"]
    n = ".".join([m for m in l_name.split('.') if m not in excl_layer_parts])
    return n

def plot_grad_flow(grad_report=None, named_parameters=None, iter=None):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

    avg_grads = []
    max_grads = []
    layers = []

    if grad_report is not None:
        for n, (avg_g, max_g) in grad_report.items():
            layers.append(n)
            avg_grads.append(avg_g)
            max_grads.append(max_g)
    elif named_parameters is not None:
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                avg_grads.append(p.grad.abs().mean().cpu().item())
                max_grads.append(p.grad.abs().max().cpu().item())

    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.3, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), avg_grads, alpha=0.6, lw=1, color="b")
    plt.hlines(0, 0, len(avg_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(avg_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(avg_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    title = "Gradient flow"
    if iter is not None:
        title += " @iter {}".format(iter)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    #plt.gcf().subplots_adjust(bottom=0.15)
    plt.show()

    grad_report = {n: [avg_grads[i], max_grads[i]] for i, n in enumerate(layers)}
    #print_grad_report(grad_report)

    return grad_report

def print_grad_report(grad_report):
    for n, (avg_g, max_g) in grad_report.items():
        print("{}: avg {}, max {}".format(n, avg_g, max_g))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_random_seeds(rnd_seed):
    random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)
    torch.cuda.manual_seed_all(rnd_seed)
    np.random.seed(rnd_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def map_time_stamp_to_vector(ts, rel_parts):
    """
    input:
        ts (int): UNIX time stamp
        rel_parts list(str): parts of datetime relevant for time vector
    output:
        ts_vector (list): vector representation of the datetime
    """

    if rel_parts is None:
        rel_parts = ['WD', 'HH', 'mm']

    ts = arrow.get(ts)

    dt_parts = {'YY': ts.year, 'MM': ts.month, 'DD': ts.day,
              'WD': ts.weekday(), 'HH': ts.hour, 'mm': ts.minute, 'ss': ts.second}

    ts_vector = []

    for key in rel_parts:
        if key not in dt_parts:
            raise ValueError("{} is not valid part of datetime".format(key))
        ts_vector.append(dt_parts[key])

    return ts_vector

def init_weights(m):
    if isinstance(m, nn.Linear):
        if m.reset_parameters():
            return
    elif isinstance(m, nn.Embedding) and not m.requires_grad:
        pass
    torch.nn.init.xavier_uniform_(m)



    # if isinstance(m, nn.Linear):
    #     if m.reset_parameters():
    #         pass
    #     else:
    #         torch.nn.init.xavier_uniform(m)
    #         #m.bias.data.fill_(0.01)
    # if isinstance(m, nn.Conv1d):
    #     torch.nn.init.xavier_uniform_(m.weight)
    #     if m.bias is not None:
    #         torch.nn.init.zeros_(m.bias)
    # if isinstance(m, nn.Conv2d):
    #     torch.nn.init.xavier_uniform_(m.weight)
    #     if m.bias is not None:
    #         torch.nn.init.zeros_(m.bias)
    # elif isinstance(m, nn.Embedding):
    #     pass
    #     # embeddings are initialised in a different way
    #     # if m.requires_grad:
    #     #     nn.init.xavier_uniform(m)

def check_all_equal(iterator):
    # check if all elements have the same value
    return len(set(iterator)) <= 1

def pad_sequence(seq, max_len, pad_value=0, pad='post', trunc='last'):
    if len(seq) < max_len:
        if pad == 'post':
            return seq + [pad_value] * (max_len - len(seq))
        elif pad == 'pre':
            return [pad_value] * (max_len - len(seq)) + seq
        else:
            raise NotImplementedError()

    elif len(seq) > max_len:
        if trunc == 'last':
            return seq[:max_len]
        elif trunc == 'first':
            return seq[len(seq) - max_len:]
        else:
            raise NotImplementedError()

    return seq

def get_art_id_from_dpg_history(articles_read, with_time=False):

    # format:
    # entry[0] = news_paper
    # entry[1] = article_id
    # entry[2] = time_stamp
    if with_time:
        return [(art_id, time_stamp) for _, art_id, time_stamp in articles_read]
    else:
        return [art_id for _, art_id, _ in articles_read]

def build_vocab_from_word_counts(vocab_raw, max_vocab_size, min_counts_for_vocab):
    vocab = {}
    # vocab = dict(heapq.nlargest(max_vocab_size, vocab.items(), key=lambda i: i[1]))
    for word, counter in vocab_raw.most_common(max_vocab_size):
        if counter >= min_counts_for_vocab:
            vocab[word] = len(vocab)
        else:
            break
    return vocab

def reverse_mapping_dict(item2idx):
    return {idx: item for item, idx in item2idx.items()}


def print_setting(config, valid_keys=None):
    if valid_keys == None:
        valid_keys = vars(config).keys()
    for key, value in vars(config).items():
        if key in valid_keys:
            print("{0}: {1}".format(key, value))

# def create_checkpoint(check_dir, filename, dataset, model, optimizer, results, step):
#
#     checkpoint_path = check_dir / (f'{filename}_step_{step}.pt')
#
#     print(f"Saving checkpoint to {checkpoint_path}")
#
#     torch.save(
#         {
#             'step': step,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'results': results,
#             'dataset': dataset
#         },
#         checkpoint_path
#     )
#
#     print("Saved.")

# def load_checkpoint(checkpoint_path, model, optimizer):
#     #load checkpoint saved at checkpoint_path
#
#     checkpoint = torch.load(checkpoint_path)
#     dataset = checkpoint['dataset']
#     step = checkpoint['step'] + 1
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     results = checkpoint['results']
#
#     return dataset, results, step

def create_exp_name(config, seperator='_'):

    now = datetime.datetime.now()
    date = now.strftime("%m-%d-%y")

    res_path = Path(config.results_path)
    res_path = res_path / date
    try:
        n_exp = len(os.listdir(res_path)) + 1
    except:
        n_exp = 1

    sep = seperator #
    exp_name = "exp" + str(n_exp)

    if config.exp_name is not None:
        exp_name += str(config.exp_name)
        exp_name += sep

    exp_name += config.eval_method \
                + sep + "lr{:.0E}".format(config.lr) \
                + sep + "seed" + str(config.random_seed) \
                + sep + "T" + str(now.strftime("%H:%M"))

    print(exp_name)
    res_path = res_path / exp_name
    res_path.mkdir(parents=True, exist_ok=True)

    return exp_name, res_path

def save_metrics_as_pickle(metrics, res_path : Path, file_name : str):
    with open(res_path / (file_name + '.pkl'), 'wb') as fout:
        pickle.dump(metrics, fout, protocol=pickle.HIGHEST_PROTOCOL)
    print("Metrics saved to {}\n".format((res_path / (file_name + '.pkl'))))

def save_config_as_json(config, res_path : Path):
    if not isinstance(config, dict):
        config = {key: val for key, val in vars(config).items()}
    with open(res_path / 'config.json', 'w') as fout:
        json.dump(config, fout, sort_keys=True, indent=4)

def save_exp_name_label(config, res_path : Path, exp_name : str):
    # Currently un-used [03.06] -> consider removing

    #exp_name.json
    # 'lbl_short' : wu-sin-42
    # 'exp_name_long': exp_name

    # Goal: read in json and extract exp label with key 'lbl_short'

    exp_dict = {}

    # relevant: eval_method, rnd_seed,
    rel_config = {}
    rel_config['eval_method'] = config.eval_method
    rel_config['seed'] = config.random_seed

    exp_dict['lbl_short'] = "-".join(list(map(str, rel_config.values())))
    exp_dict['exp_name_long'] = exp_name

    with open(res_path / 'exp_name.json', 'w') as fout:
        json.dump(exp_dict, fout, indent=4)

def create_exp_lbl_short(rel_items : dict):
    lbl = "-".join(list(map(str, rel_items.values())))
    return lbl