import argparse
import json
import arrow
import gzip
import os
import pickle
from pathlib import Path
import smart_open
import random

import datetime

from collections import OrderedDict, defaultdict, Counter
from tqdm import tqdm

import sys
sys.path.append("..")

DATA_SIZES = {'dev': [10e3, 2e3], 'medium': [45e3, 10e3], 'large': [250e3, 50e3]}

USER_ITEM_RATIO = 10

def data_stream_generator(data_dir):
    files = [file for file in Path(data_dir).iterdir() if
             file.is_file() and file.name[0] not in '_.']
    for file in files:
        with smart_open.open(file) as rf:
            yield from (json.loads(line) for line in rf.read().splitlines() if line)

def get_text_snippet(text, len_snippet, tokenizer=None):
    if tokenizer is None:
        naive_tokens = str(text).split(" ")[:len_snippet]
        if len(naive_tokens) > len_snippet:
            tokens = naive_tokens[:len_snippet]
        else:
            tokens = naive_tokens
    else:
        #tokens = tokenizer.tokenize(text)
        raise NotImplementedError()

    return " ".join(tokens)

def get_n_words(text, tokenizer=None):
    if tokenizer is None:
        tokens = str(text).split(" ")
    else:
        # tokens = tokenizer.tokenize(text)
        raise NotImplementedError()

    return len(tokens)


# def subsample_items(n_items, snippet_len, data_dir, keys_to_exclude = ["short_id", "url"], test_time_thresh=None):
#     item_dict = OrderedDict()
#     item_dict['all'] = {}
#
#     # if test_time_thresh is not None:
#     #     item_dict['train'] = {}
#     #     item_dict['test'] = {}
#
#     for i, item in enumerate(data_stream_generator(data_dir + "items")):
#         #item.keys() = dict_keys(['text', 'pub_date', 'author', 'url', 'short_id'])
#         if snippet_len is not None:
#             item['snippet'] = get_text_snippet(item['text'], snippet_len)
#
#         item['n_words'] = get_n_words(item['text'])
#         val_dict = {key: val for (key, val) in item.items() if key not in keys_to_exclude}
#         item_dict['all'][item['short_id']] = val_dict
#
#         if item['pub_date']: #< test_time_thresh:
#              pass # currently no pub_date available [30.03.]
#
#         if i == n_items-1:
#             break
#
#     return item_dict

def update_logging_dates(articles_read, logging_dates):
    first, last = logging_dates

    for art_id, time_stamp in articles_read:
        # entry[0] = article_id
        # entry[1] = time_stamp

        if first is None or last is None:
            first = last = time_stamp

        if time_stamp < first:
            first = time_stamp
        if time_stamp > last:
            last = time_stamp

    return (first, last)


def subsample_users(n_users, article_data, data_dir, min_hist_len, max_hist_len=None, remove_unk_arts=True, test_time_thresh=None):

    if isinstance(article_data, dict):
        # use a given item sub sample
        valid_item_ids = set(article_data['all'].keys())
        n_max_items = len(article_data['all'])
    elif isinstance(article_data, set):
        valid_item_ids = article_data
        article_data = {}

    c_articles_raw = Counter() # without threshold
    c_articles_thresh = Counter() # count only those from 'valid' histories

    if test_time_thresh is not None:
        article_data['train'] = set()
        article_data['test'] = set()

    user_dict = OrderedDict()
    unk_arts = defaultdict(int)
    removed_users = 0

    logging_dates = (None, None) # first & last reading date

    if max_hist_len is None:
        max_hist_len = int(1e10)

    for i, user in tqdm(enumerate(data_stream_generator(data_dir + "users"))):
        # quick preliminary eval
        if len(user['articles_read']) >= min_hist_len and len(user['articles_read']) <= max_hist_len:
            # remove unknown articles from reading history
            if remove_unk_arts:
                history = []
                # check if article ID is present in our set of valid IDs
                for entry in user['articles_read']:
                    if len(entry) == 3: ## december 19 data has 3 fields for each interaction
                        _, art_id, time_stamp = entry

                    elif len(entry) == 5: ## november 19 data has 5
                        art_id, ts = entry[-2:]
                        # convert time_stamp to UNIX
                        time_stamp = time_stamp2unix(ts)


                    if art_id not in valid_item_ids:
                        unk_arts[art_id] += 1

                        # if replace_with_unk:
                        #     entry[1] = unk_art_id
                        #     history.append(entry)
                    else:
                        if test_time_thresh is not None:
                            if 'articles_train' not in user.keys():
                                user['articles_train'] = []
                                user['articles_test'] = []

                            if time_stamp < test_time_thresh:
                                user['articles_train'].append([art_id, time_stamp])
                            else:
                                user['articles_test'].append([art_id, time_stamp])

                        history.append([art_id, time_stamp])
                        c_articles_raw.update([art_id])

                user['articles_read'] = history

            # evaluate length reading history
            if len(user['articles_read']) >= min_hist_len and len(user['articles_read']) <= max_hist_len:
                # add valid user that fulfills condition
                # exclude very high frequency users (potentially bots) and very low ones (too little interaction for proper modelling)
                # item.keys() = dict_keys(['user_id', 'articles_read', 'opened_pushes', 'articles_pushed'])
                user['articles_read'] = sorted(user['articles_read'], key=lambda entry: entry[1])  # sort by time_stamp

                if test_time_thresh is not None:
                    if len(user['articles_train']) >= min_hist_len:
                        #
                        art_train = [art_id for art_id, _ in user['articles_train']]
                        article_data['train'].update(art_train)
                        c_articles_thresh.update(art_train)

                        art_test = [art_id for art_id, _ in user['articles_test']]
                        article_data['test'].update(art_test)
                        c_articles_thresh.update(art_test)
                    else:
                        continue

                user['n_arts_read'] = len(user['articles_read'])
                user_dict[user['user_id']] = {key: val for (key, val) in user.items() if key != 'user_id'}
                #
                logging_dates = update_logging_dates(user['articles_read'], logging_dates)
            else:
                removed_users += 1
                if removed_users % 1e4 == 0:
                    print("{} \t {}".format(len(user_dict), removed_users))

        # break condition
        if len(user_dict) == n_users or i == n_users * 200: #200
            break
    #print("Users sampled {}".format(len(user_dict)))
    print("\n Start logging date: {} \t End: {}".format(*logging_dates))

    return user_dict, article_data, logging_dates

def count_article_interactions(data_dir, n_users=None):

    c_art_ids = Counter()

    if n_users is not None:
        limit = n_users*10
    else:
        limit = -1

    print("Determine most common articles based on user interaction ...")
    for i, user in tqdm(enumerate(data_stream_generator(data_dir + "users"))):
        if len(user['articles_read'][0]) == 3:
            _, art_ids, _ = zip(*user['articles_read'])
        elif len(user['articles_read'][0]) == 5:
            _, _, _, art_ids, _ = zip(*user['articles_read'])
        else:
            raise NotImplementedError("Unknown format of 'articles_read'")
        c_art_ids.update(art_ids)

        if i == limit:
            break
        elif i % 1e4 == 0 and i > 0:
            print("{} users evaluated".format(i))

    return c_art_ids

def subsample_items_from_id(data_dir: str, valid_ids: set, news_len: int, n_news: int, add_items=False, keys_to_exclude = ["short_id", "url"], test_time_thresh=None):
    article_dict = OrderedDict()
    article_dict['all'] = {}

    if add_items:
        additional_items = n_news - (len(valid_ids) if valid_ids is not None else 0)
    else:
        additional_items = 0

    for i, item in enumerate(data_stream_generator(data_dir + "items")):
        # item.keys() = dict_keys(['text', 'pub_date', 'author', 'url', 'short_id'])
        if item['short_id'] in valid_ids or additional_items:

            if news_len is not None:
                item['snippet'] = get_text_snippet(item['text'], news_len)

            item['n_words'] = get_n_words(item['text'])
            val_dict = {key: val for (key, val) in item.items() if key not in keys_to_exclude}
            article_dict['all'][item['short_id']] = val_dict

            if item['pub_date'] is not None:
                item['pub_date'] = (item['first_pub_date'] if item['first_pub_date'] is not None else item['pub_date'])
            #item['pub_date'] = (item['first_pub_date'] if item['first_pub_date'] is not None else item['pub_date'])
            # currently no pub_date available [30.03.]

            if item['short_id'] not in valid_ids:
                additional_items -= 1

        else:
            if n_news == len(article_dict['all']):
                break

        if i % 1e5 == 0 and i>0:
            print("{} items scanned ..".format(i))

    return article_dict

def save_data_to_dir(save_path, sample_name, news_data, user_data, logging_dates):

    # save data
    if save_path is not None:
        save_path = Path(save_path)

        if sample_name is None:
            sample_name = ('i{}k_u{}k_s{}'.format(int(n_news/1e3), int(n_users/1e3), len(news_data['all'])))
        save_path = save_path / sample_name
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / 'news_data.pkl', 'wb') as fout:
            pickle.dump(news_data, fout)

        with open(save_path / 'user_data.pkl', 'wb') as fout:
            pickle.dump(user_data, fout)

        with open(save_path / 'logging_dates.json', 'w') as fout:
            json.dump(logging_dates, fout)


def get_data_common_interactions(data_dir, n_news, n_users, news_len=30, min_hist_len=5, max_hist_len=None, sample_name=None, save_path=None, test_time_thresh=None, overwrite_existing=False):
    try:
        os.listdir(data_dir)
    except:
        raise FileNotFoundError("Given data directory is wrong!")

    save_path = Path(save_path)
    if (save_path / sample_name).exists():
        print("Directory already exists: {}".format(save_path / sample_name))
        if not overwrite_existing:
            print("Data will NOT be overwritten => Exit")
            return (None, None, None)
        # reply = str(input('Overwrite? (y/n): ')).lower().strip()
        # if reply[0] != 'y':
        #     return (None, None, None)

    n_news = int(n_news)
    n_users = int(n_users)

    #
    # subsample items
    if "most_common" == config.item_sample_method:

        if (save_path / "counter_article_ids.pkl").exists():
            with open((save_path / "counter_article_ids.pkl"), 'rb') as fin:
                c_art_ids = pickle.load(fin)
        else:
            c_art_ids = count_article_interactions(data_dir)
            with open(save_path / "counter_article_ids.pkl", 'wb') as fout:
                pickle.dump(c_art_ids, fout)

        valid_article_ids = set([e[0] for e in c_art_ids.most_common(n_news)])

        # get all valid items
        add_items = (True if len(valid_article_ids) < n_news else False) # flag to indicate whether to add random articles. only applies if most common is insufficient
        news_data = subsample_items_from_id(data_dir, valid_article_ids, n_news=n_news, news_len=news_len, add_items=add_items, test_time_thresh=test_time_thresh)

    elif "random" == config.item_sample_method:
        news_data = subsample_items_from_id(data_dir, set(), n_news=n_news, news_len=news_len, add_items=True, test_time_thresh=test_time_thresh)
    else:
        raise NotImplementedError()

    # subsample users
    print("Sample users ...")
    user_data, news_data, logging_dates = subsample_users(n_users, news_data, data_dir, min_hist_len,
                                               max_hist_len=max_hist_len,
                                               test_time_thresh=test_time_thresh)

    logging_dates = {'start': logging_dates[0], 'end': logging_dates[1]}

    if test_time_thresh is not None:
        logging_dates['threshold'] = test_time_thresh

    return news_data, user_data, logging_dates

def time_stamp2unix(time_stamp, format=None):
    if format is None:
        try:
            t_unix = arrow.get(time_stamp).timestamp
            #t_unix = int(datetime.datetime.strptime(time_stamp, '%d-%m-%Y-%H-%M-%S').strftime("%s"))

        except:
            print("Error in converting a timestamp! Please specify format")
            t_unix = -1
    else:
        t_unix = arrow.get(time_stamp, format).timestamp

    return t_unix

def get_all_item_ids(data_dir):
    item_ids = set()

    for i, item in enumerate(data_stream_generator(data_dir + "items")):
        # item.keys() = dict_keys(['text', 'pub_date', 'author', 'url', 'short_id'])
        item_ids.add(item['short_id'])
        if i % 1e5 == 0 and i > 0:
            print("{} items scanned ..".format(i))

    return item_ids


def get_data_wu_sampling(data_dir, n_users, news_len, min_hist_len, max_hist_len, save_path, sample_name,
                         test_time_thresh, overwrite_existing):
    try:
        os.listdir(data_dir)
    except:
        raise FileNotFoundError("Given data directory is wrong!")

    save_path = Path(save_path)
    if (save_path / sample_name).exists():
        print("Directory already exists: {}".format(save_path / sample_name))
        if not overwrite_existing:
            print("Data will NOT be overwritten => Exit")
            return (None, None, None)
        else:
            print("Data will be overwritten")

    #read in all valid item ids
    f_name_all_item_ids = "all_item_ids.pkl"
    try:
        with open(data_dir + "/" + f_name_all_item_ids, 'rb') as fin:
            all_item_ids = pickle.load(fin)
    except:
        print("Did not find {} in {}. Continue to create this first".format(f_name_all_item_ids, data_dir))

        all_item_ids = get_all_item_ids(data_dir)

        with open(data_dir + "/" + f_name_all_item_ids, 'wb') as fout:
            pickle.dump(all_item_ids, fout)

    #sample n_users and their item interactions
    print("Sample users ...")
    user_data, news_data, logging_dates = subsample_users(n_users, all_item_ids, data_dir,
                                               min_hist_len=min_hist_len,
                                               max_hist_len=max_hist_len,
                                               test_time_thresh=test_time_thresh)

    #specify logging dates
    logging_dates = {'start': logging_dates[0], 'end': logging_dates[1]}

    if test_time_thresh is not None:
        logging_dates['threshold'] = test_time_thresh


    #sample item data for those items
    valid_article_ids = set(news_data['train']).union(set(news_data['test']))

    article_data = subsample_items_from_id(data_dir, valid_article_ids,
                                        n_news=-1, news_len=news_len,
                                        test_time_thresh=test_time_thresh)
    news_data['all'] = article_data['all']

    assert len(news_data['all']) == len(valid_article_ids), "Article mismatch"
    assert len(user_data) == n_users, "Not enough users sampled"

    return news_data, user_data, logging_dates

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='../datasets/DPG_nov19/', help='data path')
    parser.add_argument('--save_path', type=str, default='../datasets/DPG_nov19/', help='path to save data')
    #parser.add_argument('--sample_name', type=str, default='', help='name for directory')
    parser.add_argument('--overwrite_existing', type=bool, default=True)

    parser.add_argument('--item_sample_method', type=str, default='most_common', choices=['random', 'most_common', 'wu'], help='')
    parser.add_argument('--size', type=str, default='medium', choices=["dev", "medium", "custom"], help='size of dataset')
    parser.add_argument('--n_articles', type=int, default=2000, help='number of articles')
    parser.add_argument('--n_users', type=int, default=2000, help='number of users')
    parser.add_argument('--ratio_user_items', type=int, default=USER_ITEM_RATIO, help='ratio of user to items, e.g. 1 : 10')

    #parser.add_argument('--vocab_size', type=int, default=30000, help='vocab')
    parser.add_argument('--time_threshold', type=str, default="23-11-2019 23:59:59", help='date for splitting train/test')
    parser.add_argument('--time_format', type=str, default=None, choices=['YYYY-MM-DD HH:mm:ss', 'DD-MM-YYYY HH:mm:ss', 'YYYY-MM-DD-HH-mm-ss'],
                        help="Specify the format for the time threshold if it defiates from ISO 8601, e.g. >> 2013-09-30T15:34:00.000-07:00 <<")

    parser.add_argument('--news_len', type=int, default=30, help='number of words from news body')
    parser.add_argument('--min_hist_len', type=int, default=6, help='minimum number of articles in reading history')
    parser.add_argument('--max_hist_len', type=int, default=300, help='max number of articles in reading history')


    config = parser.parse_args()


    if config.size in DATA_SIZES.keys():
        n_news, n_users = DATA_SIZES[config.size]
    elif "custom" == config.size:
        n_users = config.n_users
        n_news = config.n_articles
    else:
        raise NotImplementedError()

    delim = "_"
    sample_name = config.size + delim + "time_split" + delim + config.item_sample_method

    if config.time_format is not None:
        threshold_time = time_stamp2unix(config.time_threshold, config.time_format)
    else:
        threshold_time = arrow.get(config.time_threshold, 'DD-MM-YYYY HH:mm:ss').timestamp

    #threshold_date = int(datetime.datetime.strptime(config.time_threshold, '%d-%m-%Y-%H-%M-%S').strftime("%s")) #1577228399

    if "wu" == config.item_sample_method:
        news_data, user_data, logging_dates = get_data_wu_sampling(config.data_dir, n_users,
                                                                   news_len=config.news_len,
                                                                   min_hist_len=config.min_hist_len,
                                                                   max_hist_len=config.max_hist_len,
                                                                   save_path=config.save_path, sample_name=sample_name,
                                                                   test_time_thresh=threshold_time,
                                                                   overwrite_existing=config.overwrite_existing)
    else:
        news_data, user_data, logging_dates = get_data_common_interactions(config.data_dir, n_news, n_users,
                                                                           news_len=config.news_len,
                                                                           min_hist_len=config.min_hist_len,
                                                                           max_hist_len=config.max_hist_len,
                                                                           save_path=config.save_path, sample_name=sample_name,
                                                                           test_time_thresh=threshold_time,
                                                                           overwrite_existing=config.overwrite_existing)

    save_data_to_dir(config.save_path, sample_name, news_data, user_data, logging_dates)

    print("Done")
    print("User: {} \t Articles: {}".format(len(user_data), len(news_data['all'])))
    print("Logging dates: ")
    for key, val in logging_dates.items():
        print("{} {}".format(key, arrow.get(val).format()))