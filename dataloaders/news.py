


class NpaDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)

        args.num_items = len(self.smap)
        self.max_hist_len = args.max_hist_len

        self.split_method = args.split
        self.multiple_eval_items = args.split == "time_threshold"
        self.valid_items = self.get_valid_items()

        self.w_time_stamps = args.incl_time_stamp

        data = dataset.load_dataset()
        self.vocab = data['vocab']
        if self.vocab is not None:
            args.vocab_path = dataset._get_preprocessed_dataset_path()

        self.art_index2word_ids = data['art2words'] # art ID -> [word IDs]
        self.max_article_len = args.max_article_len

        if args.fix_pt_art_emb:
            args.rel_pc_art_emb_path = dataset._get_precomputed_art_emb_path()

        self.mask_token = args.max_vocab_size + 1
        args.bert_mask_token = self.mask_token

        if self.args.fix_pt_art_emb:
            self.art_id2word_ids = None
        else:
            # create direct mapping art_id -> word_ids
            self.art_id2word_ids = {art_idx: self.art_index2word_ids[art_id] for art_id, art_idx in self.smap.items()}
        del self.art_index2word_ids


        ####################
        # Negative Sampling

        self.train_neg_sampler = self.get_negative_sampler("train",
                                                           args.train_negative_sampler_code,
                                                           args.train_negative_sample_size,
                                                           args.train_negative_sampling_seed,
                                                           self.valid_items['train'],
                                                           self.get_seq_lengths(self.train))

        self.train_negative_samples = self.train_neg_sampler.get_negative_samples()

        self.test_neg_sampler = self.get_negative_sampler("test",
                                                           args.test_negative_sampler_code,
                                                           args.test_negative_sample_size,
                                                           args.test_negative_sampling_seed,
                                                           self.valid_items['test'],
                                                           None)

        self.test_negative_samples = self.test_neg_sampler.get_negative_samples()

    @classmethod
    def code(cls):
        return 'npa'
