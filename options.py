from templates import set_template
from datasets import DATASETS
from dataloaders import DATALOADERS
from source.models import MODELS
from source.modules import NEWS_ENCODER
from trainers import TRAINERS

import argparse


parser = argparse.ArgumentParser(description='RecPlay')

################
# Top Level
################
parser.add_argument('--mode', type=str, default='train', choices=['train'])
parser.add_argument('--template', type=str, default='local_test', choices=['train_bert_pcp', 'train_bert', 'train_bert_ml', 'local_test', 'train_npa', 'train_mod_npa'])
parser.add_argument('--local', type=bool, default=False, help="Run model locally reduces the batch size and other params")

################
# Test
################
parser.add_argument('--test_model_path', type=str, default=None)

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default='DPG_nov19', choices=DATASETS.keys())
parser.add_argument('--dataset_path', type=str, default="./Data/DPG_nov19/40k_time_split_n_rnd_users")

## MovieLens
parser.add_argument('--min_rating', type=int, default=None, help='Only keep ratings greater than equal to this value')
parser.add_argument('--min_uc', type=int, default=None, help='Only keep users with more than min_uc ratings')
parser.add_argument('--min_sc', type=int, default=None, help='Only keep items with more than min_sc ratings')
parser.add_argument('--split', type=str, default='leave_one_out', help='How to split the datasets')
parser.add_argument('--dataset_split_seed', type=int, default=None)
parser.add_argument('--eval_set_size', type=int, default=None,
                    help='Size of val and test set. 500 for ML-1m and 10000 for ML-20m recommended')

## DPG
parser.add_argument('--min_hist_len', type=int, default=8, help='Only keep users with reading histories longer than this value')
parser.add_argument('--min_counts_for_vocab', type=int, default=2, help='Include word in vocabulary with this minimal occurrences')
parser.add_argument('--max_vocab_size', type=int, default=30000, help='Max number of words in the vocabulary')
parser.add_argument('--max_article_len', type=int, default=30, help='Max number of words per article')
parser.add_argument('--max_hist_len', type=int, default=100, help='max number of articles in reading history')
parser.add_argument('--min_test_len', type=int, default=2, help='minimum number of articles in test interval')

parser.add_argument('--use_article_content', type=bool, default=False, help="Indicate whether to create contextualised article embeddings or randomly initialised ones")
parser.add_argument('--precompute_art_emb', type=bool, default=False, help="Precompute article embeddings in preprocessing step and use fixed embeddings for training")

parser.add_argument('--incl_time_stamp', type=bool, default=False, help="Time stamps for article reads or not")
parser.add_argument('--incl_u_id', type=bool, default=False, help="User ID passed to model")
parser.add_argument('--time_threshold', type=str, default="23-11-2019 23:59:59", help='date for splitting train/test. format: "DD-MM-YYYY HH:mm:ss"')

parser.add_argument('--train_method', type=str, default='cloze', choices=['cloze', 'npa', 'pos_cut_off'])
parser.add_argument('--n_articles', type=int, help="Number of articles in the dataset")
parser.add_argument('--n_users', type=int, help="Number of users in the dataset")

parser.add_argument('--validation_portion', type=float, default=0.1, help="Portion of test data used for validation")
parser.add_argument('--lower_case', type=bool, default=False, help="Lowercase the article content")
parser.add_argument('--language', type=str, default=None, choices=['dutch'])


################
# Dataloader
################
parser.add_argument('--dataloader_code', type=str, default='bert', choices=DATALOADERS.keys())
parser.add_argument('--dataloader_random_seed', type=float, default=0.0)
parser.add_argument('--train_batch_size', type=int, default=100)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--eval_method', type=str, choices=['last_as_target', 'random_as_target'])


################
# NegativeSampler
################
parser.add_argument('--train_negative_sampler_code', type=str, default='random', choices=['popular', 'random', 'random_common'],
                    help='Method to sample negative items for training. Not used in bert')
parser.add_argument('--train_negative_sample_size', type=int, default=None)
parser.add_argument('--train_negative_sampling_seed', type=int, default=None)
parser.add_argument('--test_negative_sampler_code', type=str, default='random', choices=['popular', 'random', 'random_common'],
                    help='Method to sample negative items for evaluation')
parser.add_argument('--test_negative_sample_size', type=int, default=None)
parser.add_argument('--test_negative_sampling_seed', type=int, default=None)
#parser.add_argument('--most_common', type=int, default=None, help='M most common articles for neg sampling')

################
# Trainer
################
parser.add_argument('--trainer_code', type=str, default='bert', choices=TRAINERS.keys())
# device #
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--device_idx', type=str, default='0')
parser.add_argument('--cuda_launch_blocking', type=bool, default=False)

# optimizer #
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
# lr scheduler #
parser.add_argument('--lr_schedule', type=int, default=0, help="Enable lr scheduler")
parser.add_argument('--decay_step', type=int, default=None, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
# epochs #
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training') #100
parser.add_argument('--num_samples', type=float, default=1e6, help='Number of samples for training')
# logger #
parser.add_argument('--log_period_as_iter', type=int, default=10000)
parser.add_argument('--log_grads', type=bool, default=True, help="Log gradients during training")
# evaluation #
parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 20, 50], help='ks for Metric@k')
parser.add_argument('--best_metric', type=str, default='AUC', help='Metric for determining the best model')

# Finding optimal beta for VAE #
# parser.add_argument('--find_best_beta', type=bool, default=False,
#                     help='If set True, the trainer will anneal beta all the way up to 1.0 and find the best beta')
# parser.add_argument('--total_anneal_steps', type=int, default=2000, help='The step number when beta reaches 1.0')
# parser.add_argument('--anneal_cap', type=float, default=0.2, help='Upper limit of increasing beta. Set this as the best beta found')

################
# Pretrained Embeddings
################
parser.add_argument('--pt_word_emb_path', type=str, default=None, help='Path to pretrained word embeddings')
parser.add_argument('--dim_word_emb', type=int, default=300, help='Dimension of word embedding vectors')

parser.add_argument('--pt_art_emb_path', type=str, default=None, help='Path to pretrained article embeddings')
parser.add_argument('--dim_art_emb', type=int, default=300, help='Dimension of word embedding vectors')

parser.add_argument('--rel_pc_art_emb_path', type=str, default=None, help='Path to relevant precomputed article embeddings')
parser.add_argument('--bert_feature_method', type=str, default=None, nargs=2, help='Method for BERT-based article embs')


################
# Model
################
parser.add_argument('--model_code', type=str, default='bert', choices=MODELS.keys())
parser.add_argument('--model_init_seed', type=int, default=None)

## News Encoder #

# pre-trained stuff
parser.add_argument('--pt_news_encoder', type=str, default=None, choices=["BERTje"], help='Pretrained model to use as News Encoder')
parser.add_argument('--path_pt_news_enc', type=str, default=None, help="Path to pre-trained News Encoder")
parser.add_argument('--fix_pt_art_emb', type=bool, default=None, help='fix pre-computed article embeddings')
parser.add_argument('--pd_vocab', type=bool, default=None, help='use pre-defined vocabulary')
parser.add_argument('--vocab_path', type=str, default=None, help='Path to vocab with relevant words')

# end-to-end
parser.add_argument('--news_encoder', type=str, default=None, choices=["wucnn", "transf", 'bertje'], help='Model to use as News Encoder')

# Transformer Encoder #
parser.add_argument('--transf_hidden_units', type=int, default=None, help='Size of hidden vectors (d_model)')
parser.add_argument('--transf_enc_num_layers', type=int, default=1, help='Number of transformer layers')
parser.add_argument('--tranf_enc_num_heads', type=int, default=2, help='Number of heads for multi-attention')
parser.add_argument('--transf_enc_dropout', type=float, default=0.1, help='Dropout probability to use throughout the model')

# Positional Embeddings #
parser.add_argument('--add_emb_size', type=int, default=None, help='Size of additive embedding')
parser.add_argument('--add_embs_func', type=str, default='add', choices=['add', 'concat'], help='Incorporate additional information to article embeddings')
parser.add_argument('--norm_art_pos_embs', type=bool, default=True, help='Normalise article & pos/temp embeddings')

parser.add_argument('--pos_embs', type=str, default=None, choices=['tpe', 'lpe', 'gnoise'], help='Type of positional embedding')

# Temporal Embeddings #
parser.add_argument('--normalise_time_stamps', type=str, default='standard', help="specify scaler for time stamps")
parser.add_argument('--len_time_vec', type=int, default=4, help='Which information to include from UNIX timestamp')
parser.add_argument('--temp_embs', type=str, default=None, choices=['lte', 'nte', 'tte'], help='Type of temporal embedding')
parser.add_argument('--temp_embs_hidden_units', type=int, default=[256, 768], nargs='*', help='Hidden units for neural temporal embedding')
parser.add_argument('--temp_embs_act_func', type=str, default=None, choices=['relu', 'gelu', 'tanh'], help='Activation function for neural temporal embedding')

# Prediction Layer #
parser.add_argument('--pred_layer', type=str, default=None, choices=['l2', 'cos'], help='Type of prediction layer')
parser.add_argument('--nie_layer', type=str, default=None, choices=['lin_gelu', 'lin'], help='Type of next-item embedding layer')

###########################################

# NPA #
parser.add_argument('--dim_u_id_emb', type=int, default=None, help='Dimension of embedded user ID for pers. attn.')
parser.add_argument('--dim_pref_query', type=int, default=None, help='Dimension for User Preference Query for pers. attn.')
parser.add_argument('--npa_dropout', type=float, default=None, help='Dropout probability to use for NPA model')
parser.add_argument('--npa_variant', type=str, default='vanilla', choices=['vanilla', 'bertje', 'custom'])

# BERT #
parser.add_argument('--bert_max_len', type=int, default=100, help='Length of sequence for bert')
parser.add_argument('--bert_num_items', type=int, default=None, help='Number of total items')
parser.add_argument('--bert_hidden_units', type=int, default=None, help='Size of hidden vectors (d_model)')
parser.add_argument('--bert_num_blocks', type=int, default=2, help='Number of transformer layers')
parser.add_argument('--bert_num_heads', type=int, default=4, help='Number of heads for multi-attention')
parser.add_argument('--bert_dropout', type=float, default=0.1, help='Dropout probability to use throughout the model')
parser.add_argument('--bert_mask_prob', type=float, default=0.15, help='Probability for masking items in the training sequence')
parser.add_argument('--bert_mask_token', type=int, default=None, help='Token id for Mask')


################
# Experiment
################
parser.add_argument('--experiment_dir', type=str, default='experiments')
parser.add_argument('--experiment_description', type=str, default='test', nargs='*')


################
args = parser.parse_args()
set_template(args)
