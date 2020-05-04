import datetime
import json
import os
import pickle
import random
from pathlib import Path
import sys

import numpy as np
import torch
from torch.backends import cudnn

sys.path.append("..")


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

def set_random_seeds(rnd_seed):
    random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)
    torch.cuda.manual_seed_all(rnd_seed)
    np.random.seed(rnd_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

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