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
        # num_epochs = 100
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
        # pseudo categorical prediction
        args.mode = 'train'
        #args.local = True

        # dataset
        #args.dataset_code = 'ml-' + input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
        args.dataset_code = 'DPG_nov19' if args.dataset_code is None else args.dataset_code
        args.min_hist_len = 8

        # preprosessing
        args.n_users = 10000
        args.use_article_content = True
        args.incl_time_stamp = False
        args.pt_news_encoder = 'BERTje'
        #args.path_pt_news_enc = "./BertModelsPT/bert-base-dutch-cased"

        args.fix_pt_art_emb = True
        args.pd_vocab = True
        #args.max_article_len = 30
        args.dim_art_emb = 768
        #args.lower_case = False

        # split strategy
        args.split = 'time_threshold'

        # dataloader
        args.dataloader_code = 'bert_news'
        batch = 128 if not args.local else 10
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        # negative sampling
        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 5
        args.train_negative_sampling_seed = 42 if args.model_init_seed is None else args.model_init_seed
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 99
        args.test_negative_sampling_seed = 42 if args.model_init_seed is None else args.model_init_seed  #98765

        # training
        args.trainer_code = 'bert_news_ce'
        args.device = 'cuda' if not args.local else 'cpu'

        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 25
        args.gamma = 1.0
        args.num_epochs = 100 if args.dataset_code == 'DPG_nov19' else 100
        # evaluation
        args.metric_ks = [5, 10, 50]
        args.best_metric = 'NDCG@10'

        # model
        args.model_code = 'bert4nie'
        args.model_init_seed = 42 if args.model_init_seed is None else args.model_init_seed
        # bert
        args.bert_dropout = 0.1
        args.bert_hidden_units = args.dim_art_emb
        args.bert_mask_prob = 0.15
        args.bert_max_len = 100
        args.bert_num_blocks = 2
        args.bert_num_heads = 4

        #args.pos_embs = 'tpe' # embeddings
        args.pred_layer = 'l2' # prediction layer
        args.nie_layer = None

    elif args.template.startswith('train_bert_nie'):
        # Bert4News with the Next-Item Embedding objective
        args.mode = 'train'
        args.local = True

        # dataset
        #args.dataset_code = 'ml-' + input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
        args.dataset_code = 'DPG_nov19' if args.dataset_code is None else args.dataset_code
        args.min_hist_len = 6

        # preprosessing
        args.n_users = 10000
        args.use_article_content = True
        args.incl_time_stamp = False
        args.pt_news_encoder = 'rnd'
        args.fix_pt_art_emb_fix = True
        args.pd_vocab = True

        args.max_article_len = 128
        args.dim_art_emb = 256

        # split strategy
        args.split = 'time_threshold'

        # dataloader
        args.dataloader_code = 'bert_news'
        batch = 128 if not args.local else 10
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch
        args.eval_method = 'last_as_target'

        # negative sampling
        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 5
        args.train_negative_sampling_seed = 42 if args.model_init_seed is None else args.model_init_seed
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 99
        args.test_negative_sampling_seed = 42 if args.model_init_seed is None else args.model_init_seed  #98765

        # training
        args.trainer_code = 'bert_news_dist'
        args.device = 'cuda' if not args.local else 'cpu'
        # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args.train_method = 'masked_item'

        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 25
        args.gamma = 1.0

        args.num_epochs = 100 if args.dataset_code == 'DPG_nov19' else 100
        args.metric_ks = [5, 10, 50]
        args.best_metric = 'NDCG@10'

        # model
        args.model_code = 'bert4nie'
        args.model_init_seed = 42 if args.model_init_seed is None else args.model_init_seed

        args.bert_dropout = 0.1
        args.bert_hidden_units = 256
        args.bert_mask_prob = 0.15
        args.bert_max_len = 100
        args.bert_num_blocks = 2
        args.bert_num_heads = 4

        #assert args.bert_max_len == args.max_article_len

    elif args.template.startswith('train_dae'):
        args.mode = 'train'

        args.dataset_code = 'ml-' + input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'holdout'
        args.dataset_split_seed = 98765
        args.eval_set_size = 500 if args.dataset_code == 'ml-1m' else 10000

        args.dataloader_code = 'ae'
        batch = 128 if args.dataset_code == 'ml-1m' else 512
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        args.trainer_code = 'dae'
        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 1e-3
        args.enable_lr_schedule = False
        args.weight_decay = 0.00
        args.num_epochs = 100 if args.dataset_code == 'ml-1m' else 200
        args.metric_ks = [1, 5, 10, 20, 50, 100]
        args.best_metric = 'NDCG@10'

        args.model_code = 'dae'
        args.model_init_seed = 0
        args.dae_num_hidden = 2
        args.dae_hidden_dim = 600
        args.dae_latent_dim = 200
        args.dae_dropout = 0.5

    elif args.template.startswith('train_vae_search_beta'):
        args.mode = 'train'

        args.dataset_code = 'ml-' + input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'holdout'
        args.dataset_split_seed = 98765
        args.eval_set_size = 500 if args.dataset_code == 'ml-1m' else 10000

        args.dataloader_code = 'ae'
        batch = 128 if args.dataset_code == 'ml-1m' else 512
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        args.trainer_code = 'vae'
        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 1e-3
        args.enable_lr_schedule = False
        args.weight_decay = 0.01
        args.num_epochs = 100 if args.dataset_code == 'ml-1m' else 200
        args.metric_ks = [1, 5, 10, 20, 50, 100]
        args.best_metric = 'NDCG@10'
        args.total_anneal_steps = 3000 if args.dataset_code == 'ml-1m' else 20000
        args.find_best_beta = True

        args.model_code = 'vae'
        args.model_init_seed = 0
        args.vae_num_hidden = 2
        args.vae_hidden_dim = 600
        args.vae_latent_dim = 200
        args.vae_dropout = 0.5
    
    elif args.template.startswith('train_vae_give_beta'):
        args.mode = 'train'

        args.dataset_code = 'ml-' + input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'holdout'
        args.dataset_split_seed = 98765
        args.eval_set_size = 500 if args.dataset_code == 'ml-1m' else 10000

        args.dataloader_code = 'ae'
        batch = 128 if args.dataset_code == 'ml-1m' else 512
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        args.trainer_code = 'vae'
        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 1e-3
        args.enable_lr_schedule = False
        args.weight_decay = 0.01
        args.num_epochs = 100 if args.dataset_code == 'ml-1m' else 200
        args.metric_ks = [1, 5, 10, 20, 50, 100]
        args.best_metric = 'NDCG@100'
        args.find_best_beta = False
        args.anneal_cap = 0.342
        args.total_anneal_steps = 3000 if args.dataset_code == 'ml-1m' else 20000

        args.model_code = 'vae'
        args.model_init_seed = 0
        args.vae_num_hidden = 2
        args.vae_hidden_dim = 600
        args.vae_latent_dim = 200
        args.vae_dropout = 0.5

