import argparse
import os
import pickle
import string

from collections import defaultdict, Counter, OrderedDict
from pathlib import Path
from source.preprocessing.bert_feature_extractor import BertFeatureExtractor
from source.preprocessing.utils_news_prep import get_emb_file_name

import sys
sys.path.append("..")


def compute_bert_embeddings(data_path, path_to_pt_model, feature_method, max_article_len: int,
                            batch_size: int, lower_case=False, store=True):
    """

    feature_method: tuple or list of tuples of format (method, N), e.g. ("sum_last_n", 4)
    max_article_len: int
    batch_size: int

    """

    if isinstance(data_path, str):
        with open(data_path, 'rb') as f:
            news_data = pickle.load(f) # encoding='utf-8'
    elif isinstance(data_path, dict):
        news_data = data_path
    else:
        raise NotImplementedError()

    all_articles = news_data['all']

    bert_feat_extractor = BertFeatureExtractor(path_to_pt_model, lower_case=lower_case)
    print("encode news articles ...")

    if feature_method is None:
        print('No BERTje method specified, using default "last_cls"')
        methods = [('last_cls', None), ('sum_last_n', 4)]
        feature_method = methods[1]
    else:
        if not isinstance(feature_method, tuple):
            raise ValueError("Feature method should be tuple (method, N). If 'N' does not apply, pass None")
        methods = [feature_method]
        # raise ValueError('specify method to extract BERTje features!')

    if feature_method not in methods:
        methods.append(feature_method)

    # subset = {k: all_articles[k] for k in list(all_articles.keys())[:100]}

    bert_embeddings = bert_feat_extractor.encode_text_to_features_batches(
        all_articles, methods,
        batch_size, max_article_len)
    if store:
        bert_export_path = Path("/".join(['.', 'pc_article_embeddings', "BERTje"]))
        if not bert_export_path.is_dir():
            os.makedirs(bert_export_path)

        # save pre computed bert embeddings
        for i, (m, n) in enumerate(methods):
            n = 0 if n is None else n
            method_name = get_emb_file_name(m, n, max_article_len, lower_case)
            print(method_name)

            with bert_export_path.joinpath(method_name + ".pkl").open('wb') as fout:
                pickle.dump(bert_embeddings[i], fout)

        print("Saved embeddings to {}".format(bert_export_path))
    else:
        return bert_embeddings

if __name__ == '__main__':
    #parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='../../Data/DPG_nov19/medium_time_split_n_rnd_users/news_data.pkl', help='data path')
    parser.add_argument('--model_path', type=str, default='../../BertModelsPT/bert-base-dutch-cased', help='model path')
    parser.add_argument('--max_article_len', type=int, default=30, help='Max number of words per article')
    parser.add_argument('--lower_case', type=int, default=0, help="Lowercase the article content")
    parser.add_argument('--batch_size', type=int, default=64)
    #parser.add_argument('--pkl_protocol', type=int, default=4, help="Pickle protocol with which data has been stored")

    args = parser.parse_args()

    #BERTje
    compute_bert_embeddings(args.data_dir, args.model_path, None, args.max_article_len,
                                batch_size=args.batch_size, lower_case=bool(args.lower_case))