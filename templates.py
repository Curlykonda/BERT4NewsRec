from utils import set_up_gpu

def set_template(args):
    if args.template is None:
        return

    elif args.template.startswith('train_bert_ml'):
        args.mode = 'train'

        #args.dataset_code = 'ml-' + input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
        args.dataset_code = 'ml-1m'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'leave_one_out'

        local = True

        args.dataloader_code = 'bert'
        batch = 128 if not local else 10
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.train_negative_sampling_seed = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 100
        args.test_negative_sampling_seed = 98765

        args.trainer_code = 'bert'
        args.device = 'cuda' if not local else 'cpu'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 25
        args.gamma = 1.0

        args.num_epochs = 10 if args.dataset_code == 'ml-1m' else 200
        args.metric_ks = [1, 5, 10, 20, 50, 100]
        args.best_metric = 'NDCG@10'

        args.model_code = 'bert4rec'
        args.model_init_seed = 0

        args.bert_dropout = 0.1
        args.bert_hidden_units = 256
        args.bert_mask_prob = 0.15
        args.bert_max_len = 100
        args.bert_num_blocks = 2
        args.bert_num_heads = 4

    elif args.template.startswith('train_bert_pcp'):
        set_args_bert_pcp(args)

    # elif args.template.startswith('train_bert_nie'):
    #     # Bert4News with the Next-Item Embedding objective
    #     args.mode = 'train'
    #     args.local = True
    #
    #     # dataset
    #     #args.dataset_code = 'ml-' + input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
    #     args.dataset_code = 'DPG_nov19' if args.dataset_code is None else args.dataset_code
    #
    #     # preprosessing
    #     args.n_users = 10000
    #     args.use_article_content = True
    #     args.incl_time_stamp = False
    #     args.pt_news_encoder = 'rnd'
    #     args.fix_pt_art_emb_fix = True
    #     args.pd_vocab = True
    #
    #     args.max_article_len = 128
    #     args.dim_art_emb = 256
    #
    #     # split strategy
    #     args.split = 'time_threshold'
    #
    #     # dataloader
    #     args.dataloader_code = 'bert_news'
    #     batch = 128 if not args.local else 10
    #     args.train_batch_size = batch
    #     args.val_batch_size = batch
    #     args.test_batch_size = batch
    #     args.eval_method = 'last_as_target'
    #
    #     # negative sampling
    #     args.train_negative_sampler_code = 'random'
    #     args.train_negative_sample_size = 5
    #     args.train_negative_sampling_seed = 42 if args.model_init_seed is None else args.model_init_seed
    #     args.test_negative_sampler_code = 'random'
    #     args.test_negative_sample_size = 99
    #     args.test_negative_sampling_seed = 42 if args.model_init_seed is None else args.model_init_seed  #98765
    #
    #     # training
    #     args.trainer_code = 'bert_news_dist'
    #     args.device = 'cuda' if not args.local else 'cpu'
    #     # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     args.train_method = 'masked_item'
    #
    #     args.num_gpu = 1
    #     args.device_idx = '0'
    #     args.optimizer = 'Adam'
    #     args.lr = 0.001
    #     args.enable_lr_schedule = True
    #     args.decay_step = 25
    #     args.gamma = 1.0
    #

    #     args.metric_ks = [5, 10, 50]
    #     args.best_metric = 'NDCG@10'
    #
    #     # model
    #     args.model_code = 'bert4nie'
    #     args.model_init_seed = 42 if args.model_init_seed is None else args.model_init_seed
    #
    #     args.bert_dropout = 0.1
    #     args.bert_hidden_units = 256
    #     args.bert_mask_prob = 0.15

    #     args.bert_num_blocks = 2
    #     args.bert_num_heads = 4

    elif args.template.startswith("local_test"):
        args.local = True
        args.device = 'cuda'
        args.num_epochs = 5
        args.train_batch_size = 10
        args.train_negative_sample_size=4
        args.log_period_as_iter=10000
        args.n_users=10000
        args.dataset_path="./Data/DPG_nov19/10k_time_split_n_rnd_users"
        args.experiment_description = ["10k"]

        args.lr_schedule=1
        #args.warmup_ratio=0.1

        args.max_hist_len = 100
        args.max_article_len = 30

        args.pt_news_encoder = 'BERTje'
        args.news_encoder = 'bertje'
        args.path_pt_news_enc = "./BertModelsPT/bert-base-dutch-cased"
        args.language = "dutch"

        args.bert_num_blocks=1
        # args.bert_num_heads=4

        args.nie_layer = 'lin_gelu'

        args.lr = 1e-4

        # args.news_encoder = "transf"

        # args.news_encoder = "wucnn"
        # args.dim_art_emb = 400

        # args.add_emb_size=256
        # args.add_embs_func='concat'
        # args.add_embs_func = 'add'

        set_args_bert_pcp(args)

        args.train_negative_sampler_code = 'random' # random_common
        args.test_negative_sampler_code = args.train_negative_sampler_code

        args.experiment_description.append(args.news_encoder)

        ### pos embs ###
        args.norm_art_pos_embs = True

        # args.pos_embs = None
        # args.pos_embs = 'lpe'
        # args.incl_time_stamp = False
        #
        # args.temp_embs = 'nte'
        # if 'concat' == args.add_embs_func:
        #     args.temp_embs_hidden_units = [128, args.add_emb_size]
        # else:
        #     args.temp_embs_hidden_units = [128, args.dim_art_emb]
        # args.temp_embs_act_func = "relu"
        # args.incl_time_stamp = True

        if args.pos_embs is not None:
            args.experiment_description.append(args.pos_embs)
            args.experiment_description.append(args.add_embs_func)
        elif args.temp_embs is not None:
            args.experiment_description.append(args.temp_embs)
            args.experiment_description.append(args.add_embs_func)
        else:
            args.experiment_description.append("none")

        # args.lower_case=False

        args.cuda_launch_blocking=True

    elif args.template.startswith('train_npa'):
        # local debugging
        # args.local = True

        if args.local:
            args.device = 'cuda'

            args.max_hist_len = 50
            args.npa_variant = 'custom'
            args.news_encoder = 'wucnn'

            args.num_epochs = 5
            args.train_negative_sample_size = 4
            args.log_period_as_iter = 200
            args.n_users = 10000
            args.dataset_path = "./Data/DPG_nov19/10k_time_split_n_rnd_users"

        # NPA model trained with pseudo categorical prediction
        args.mode = 'train'
        #args.local = True
        args.cuda_launch_blocking = True

        # dataset
        args.dataset_code = 'DPG_nov19' if args.dataset_code is None else args.dataset_code

        # preprosessing
        args.use_article_content = True
        args.incl_time_stamp = False

        args.max_hist_len = 50 if args.max_hist_len is None else args.max_hist_len
        args.max_article_len = 30 if args.max_article_len is None else args.max_article_len
        args.dim_art_emb = 400 if args.dim_art_emb is None else args.dim_art_emb

        # split strategy
        args.split = 'time_threshold'

        # dataloader
        args.dataloader_code = 'npa'

        args.val_batch_size = args.train_batch_size
        args.test_batch_size = args.train_batch_size
        args.dataloader_random_seed = args.model_init_seed

        # negative sampling
        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 5 if args.train_negative_sample_size is None else args.train_negative_sample_size
        args.train_negative_sampling_seed = 42 if args.model_init_seed is None else args.model_init_seed

        args.test_negative_sampler_code = args.train_negative_sampler_code = 'random'
        args.test_negative_sample_size = args.train_negative_sample_size if args.test_negative_sample_size is None else args.test_negative_sample_size
        args.test_negative_sampling_seed = 42 if args.model_init_seed is None else args.model_init_seed  #98765

        # training
        args.trainer_code = 'npa'
        args.device = 'cuda' #if not args.local else 'cpu'
        args.train_method = 'npa'

        #args.num_gpu = 1
        #args.device_idx = '0'
        args.optimizer = 'Adam'

        args.enable_lr_schedule = False

        args.num_epochs = 50 if args.num_epochs is None else args.num_epochs
        args.metric_ks = [5, 10]
        args.best_metric = 'AUC'

        # model
        if 'vanilla' == args.npa_variant:
            args.model_code = 'vanilla_npa'
        elif 'custom' == args.npa_variant:
            args.model_code = 'npa'
        else:
            raise NotImplementedError()

        args.model_init_seed = 42 if args.model_init_seed is None else args.model_init_seed

        args.dim_u_id_emb = 50 if args.dim_u_id_emb is None else args.dim_u_id_emb
        args.dim_pref_query = 200 if args.dim_pref_query is None else args.dim_pref_query
        args.npa_dropout = 0.2 if args.npa_dropout is None else args.npa_dropout

    elif args.template.startswith('train_mod_npa'):
        # args.local = True
        # args.device = 'cuda'
        #
        # args.max_hist_len = 100
        #
        # args.num_epochs = 5
        # args.train_negative_sample_size = 4
        # args.log_period_as_iter = 200
        # args.n_users = 10000
        # args.dataset_path = "./Data/DPG_nov19/10k_time_split_n_rnd_users"
        #
        # args.news_encoder="wucnn"
        # args.npa_variant = 'custom'

        if 'bertje' == args.npa_variant:
            args.fix_pt_art_emb = True
            args.pd_vocab = True
            args.dim_art_emb = 768

            args.pt_news_encoder = 'BERTje'
            args.path_pt_news_enc = "./BertModelsPT/bert-base-dutch-cased"
            args.language = "dutch"

        #######

        # Modified NPA model trained with pseudo categorical prediction
        args.mode = 'train'

        args.cuda_launch_blocking = True

        # dataset
        args.dataset_code = 'DPG_nov19' if args.dataset_code is None else args.dataset_code

        # preprosessing
        args.use_article_content = True
        args.incl_time_stamp = False

        args.max_hist_len = 50 if args.max_hist_len is None else args.max_hist_len
        args.max_article_len = 30 if args.max_article_len is None else args.max_article_len
        args.dim_art_emb = 400 if args.dim_art_emb is None else args.dim_art_emb

        # split strategy
        args.split = 'time_threshold'

        # dataloader
        args.dataloader_code = 'npa_mod'

        args.val_batch_size = args.train_batch_size
        args.test_batch_size = args.train_batch_size
        args.dataloader_random_seed = args.model_init_seed

        # negative sampling
        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 5 if args.train_negative_sample_size is None else args.train_negative_sample_size
        args.train_negative_sampling_seed = 42 if args.model_init_seed is None else args.model_init_seed

        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 9 if args.test_negative_sample_size is None else args.test_negative_sample_size
        args.test_negative_sampling_seed = 42 if args.model_init_seed is None else args.model_init_seed  #98765

        # training
        args.trainer_code = 'npa_mod'
        args.device = 'cuda' #if not args.local else 'cpu'
        args.train_method = 'cloze'

        #args.num_gpu = 1
        #args.device_idx = '0'
        args.optimizer = 'Adam'
        #args.lr = 0.001
        args.enable_lr_schedule = False
        # args.decay_step = 25
        # args.gamma = 1.0

        #50
        args.num_epochs = 10 if args.num_epochs is None else args.num_epochs
        args.metric_ks = [5, 10]
        args.best_metric = 'AUC'

        # model
        args.model_code = 'npa_mod'
        args.model_init_seed = 42 if args.model_init_seed is None else args.model_init_seed
        args.cuda_launch_blocking = True

        args.dim_u_id_emb = 50 if args.dim_u_id_emb is None else args.dim_u_id_emb
        args.dim_pref_query = 200 if args.dim_pref_query is None else args.dim_pref_query
        args.npa_dropout = 0.2 if args.npa_dropout is None else args.npa_dropout

    set_up_gpu(args)



def set_args_bert_pcp(args):
    # pseudo categorical prediction
    args.mode = 'train'

    # dataset
    args.dataset_code = 'DPG_nov19' if args.dataset_code is None else args.dataset_code

    # preprosessing
    #args.n_users = 10000
    args.use_article_content = True

    if 'BERTje' == args.pt_news_encoder:
        args.fix_pt_art_emb = True
        args.pd_vocab = True
        args.dim_art_emb = 768

    elif "wucnn" == args.news_encoder:
        set_args_npa_cnn(args)

    elif "transf" == args.news_encoder:
        set_args_transf(args)

    # pos embeddings
    if 'add' == args.add_embs_func:
        args.add_emb_size = args.dim_art_emb

    # split strategy
    args.split = 'time_threshold'

    # dataloader
    args.dataloader_code = 'bert_news'
    batch = args.train_batch_size if not args.local else 10
    args.train_batch_size = batch
    args.val_batch_size = batch
    args.test_batch_size = batch
    args.dataloader_random_seed = args.model_init_seed

    # negative sampling
    args.train_negative_sampler_code = 'random' if args.train_negative_sampler_code is None else args.train_negative_sampler_code
    #5
    args.train_negative_sample_size = 24 if args.train_negative_sample_size is None else args.train_negative_sample_size
    args.train_negative_sampling_seed = 42 if args.model_init_seed is None else args.model_init_seed

    args.test_negative_sampler_code = args.train_negative_sampler_code
    # 9
    args.test_negative_sample_size = args.train_negative_sample_size
    args.test_negative_sampling_seed = 42 if args.model_init_seed is None else args.model_init_seed  # 98765

    # training
    args.trainer_code = 'bert_news_ce'
    #args.device = 'cuda' #if not args.local else 'cpu'

    #args.num_gpu = 1 if args.num_gpu is None else args.num_gpu
    #args.device_idx = '0'
    args.optimizer = 'Adam'
    args.lr = 0.001 if args.lr is None else args.lr

    if args.lr_schedule:
        # args.decay_step = 25 if args.decay_step is None else args.decay_step
        # args.gamma = 0.1 if args.gamma is None else args.gamma
        args.warmup_ratio = 0.1 if args.warmup_ratio is None else args.warmup_ratio

    # evaluation
    args.metric_ks = [5, 10]
    args.best_metric = 'AUC'

    # model
    args.model_code = 'bert4news'
    args.model_init_seed = 42 if args.model_init_seed is None else args.model_init_seed
    # bert
    args.bert_hidden_units = args.dim_art_emb

    # args.bert_dropout = 0.1
    # args.bert_mask_prob = 0.15
    # args.bert_num_blocks = 2 if args.bert_num_blocks is None else args.bert_num_blocks
    # args.bert_num_heads = 4 if args.bert_num_heads is None else args.bert_num_heads

    args.pred_layer = 'l2'  # prediction layer
    #args.nie_layer = None


def set_args_npa_cnn(args):
    """ Set up NpaCNN as News Encoder """
    args.incl_u_id = True
    args.pt_news_encoder = None  # 'BERTje'
    args.fix_pt_art_emb = False
    args.pd_vocab = True

    args.dim_art_emb = 400 if args.dim_art_emb is None else args.dim_art_emb
    args.bert_hidden_units = args.dim_art_emb

    args.path_pt_news_enc = None


def set_args_transf(args):
    """ Set up Transformer as News Encoder """
    args.incl_u_id = False
    args.pt_news_encoder = None  # 'BERTje'
    args.fix_pt_art_emb = False
    args.pd_vocab = True

    args.dim_art_emb = 300 if args.dim_word_emb is None else args.dim_word_emb
    args.bert_hidden_units = args.dim_art_emb
    args.path_pt_news_enc = None
