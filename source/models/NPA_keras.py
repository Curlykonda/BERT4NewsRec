import numpy as np
import pickle
import os
import keras
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.optimizers import *
from sklearn.model_selection import train_test_split
import sys
sys.path.append("..")

from source.metrics import mrr_score, ndcg_score
from sklearn.metrics import roc_auc_score

from source.utils_npa import gen_batch_data, prep_dpg_user_file, get_embeddings_from_pretrained, gen_batch_data_test

npratio = 4
results = []

MAX_SENT_LENGTH = 30
MAX_SENTS = 50

BATCH_SIZE = 100

DEBUG = False


class HelperConfig(object):
    pass

def get_default_params():
    config = HelperConfig()
    # training
    config.batch_size = 100
    config.n_epochs = 3

    # prep
    config.neg_sample_ratio = 4
    config.max_len_hist = 50
    config.max_len_title = 30
    config.random_seed = 42

    #data
    if DEBUG:
        sys_add = '../'
    else:
        sys_add = ''
    data_path = sys_add + '../datasets/dpg/'

    config.user_data = data_path + 'i10k_u5k_s30/user_data.pkl'
    config.data_path = data_path + 'i10k_u5k_s30/'
    config.emb_path = sys_add + '../embeddings/cc.nl.300.bin'

    #results
    config.model_save_path = sys_add + '../saved_models/'
    config.results_path = sys_add + '../results/'

    return config

def test_model(config):
    # TODO: check how to use **kwargs properly to initialise models
    model, model_test = build_model(config, n_users=10, vocab_len=10, pretrained_emb=None, emb_dim_user_id=5, emb_dim_pref_query=10,
                emb_dim_words=10, n_filters_cnn=2)
    print(model.summary())

    return model

def build_model(config, n_users, vocab_len, pretrained_emb, emb_dim_user_id=50, emb_dim_pref_query=200, emb_dim_words=300,
                n_filters_cnn=400, dropout_p=0.2, **kwargs):

    ##user embedding - word & article level
    user_id = Input(shape=(1,), dtype='int32')
    user_embedding_layer = Embedding(n_users, emb_dim_user_id, trainable=True)
    user_embedding = user_embedding_layer(user_id)
    user_embedding_word = Dense(emb_dim_pref_query, activation='relu')(user_embedding)
    user_embedding_word = Flatten()(user_embedding_word)
    user_embedding_news = Dense(emb_dim_pref_query, activation='relu')(user_embedding)
    user_embedding_news = Flatten()(user_embedding_news)

    ##news encoder
    news_input = Input(shape=(config.max_len_title,), dtype='int32')

    if pretrained_emb:
        embedding_layer = Embedding(vocab_len, emb_dim_words, weights=[pretrained_emb], trainable=True) # weights=[pretrained_emb],
    else:
        embedding_layer = Embedding(vocab_len, emb_dim_words, trainable=True) # random initialisation

    embedded_sequences = embedding_layer(news_input)
    embedded_sequences = Dropout(dropout_p)(embedded_sequences)

    cnnouput = Convolution1D(nb_filter=n_filters_cnn, filter_length=3, padding='same', activation='relu', strides=1)(
        embedded_sequences)  # original nb_filter=400
    cnnouput = Dropout(dropout_p)(cnnouput)

    # personalised attention - word level
    attention_a = Dot((2, 1))([cnnouput, Dense(n_filters_cnn, activation='tanh')(user_embedding_word)])
    attention_weight = Activation('softmax')(attention_a)
    news_rep = keras.layers.Dot((1, 1))([cnnouput, attention_weight])
    newsEncoder = Model([news_input, user_id], news_rep)

    # browsing history as concatenation of MAX_SENTS articles
    all_news_input = [keras.Input((config.max_len_title,), dtype='int32') for _ in range(config.max_len_hist)]
    browsed_news_rep = [newsEncoder([news, user_id]) for news in all_news_input]
    browsed_news_rep = concatenate([Lambda(lambda x: K.expand_dims(x, axis=1))(news) for news in browsed_news_rep],
                                   axis=1)

    ## user encoder
    # personalised attention - article level
    attention_news = keras.layers.Dot((2, 1))([browsed_news_rep, Dense(n_filters_cnn, activation='tanh')(user_embedding_news)])
    attention_weight_news = Activation('softmax')(attention_news)
    user_rep = keras.layers.Dot((1, 1))([browsed_news_rep, attention_weight_news])

    # candidate items - as pseudo K + 1 classification task
    candidates = [keras.Input((config.max_len_title,), dtype='int32') for _ in range(1 + config.neg_sample_ratio)]
    candidate_vecs = [newsEncoder([candidate, user_id]) for candidate in candidates]
    # logits
    scores_raw = [keras.layers.dot([user_rep, candidate_vec], axes=-1) for candidate_vec in candidate_vecs]
    # normalised probs
    softm_probs = keras.layers.Activation(keras.activations.softmax)(keras.layers.concatenate(scores_raw))

    model = Model(candidates + all_news_input + [user_id], softm_probs)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])

    candidate_one = keras.Input((config.max_len_title,))
    candidate_one_vec = newsEncoder([candidate_one, user_id])
    score_raw = keras.layers.dot([user_rep, candidate_one_vec], axes=-1)
    score_sigmoid = keras.layers.Activation(keras.activations.sigmoid)(score_raw)
    model_test = keras.Model(inputs=[candidate_one] + all_news_input + [user_id], outputs=score_sigmoid)

    return model, model_test

def train():


    # 1. set hyper params
    config = get_default_params()

    # 2. prep data

    #load vocab
    with open(config.data_path + "news_prepped.pkl", 'rb') as fin:
        (vocab, news_as_word_ids, art_id2idx) = pickle.load(fin)

    u_id2idx, data = prep_dpg_user_file(config.user_data, set(art_id2idx.keys()), art_id2idx,
                                        neg_sample_ratio=config.neg_sample_ratio, max_hist_len=config.max_len_hist)

    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True, random_state=config.random_seed)

    # get pretrained embeddings
    word_embeddings = get_embeddings_from_pretrained(vocab, config.emb_path, emb_dim=300)

    # 3. build model
    #model, model_test = build_model(config, n_users=len(u_id2idx), vocab_len=len(vocab), pretrained_emb=word_embeddings)
    model = test_model(config)

    # 4. training loop
    results = {}

    for ep in range(config.n_epochs):
        traingen = gen_batch_data(train_data, news_as_word_ids, config.batch_size)
        model.fit_generator(traingen, epochs=1, steps_per_epoch=len(train_data) // config.batch_size)

        # test
        testgen = gen_batch_data(test_data, news_as_word_ids, config.batch_size)
        #click_score = model_test.predict_generator(testgen, steps=len(testgen) // config.batch_size, verbose=1)
        #scores = model.predict_generator(testgen, steps=len(testgen) // config.batch_size, verbose=1)

        auc = []
        mrr = []
        ndcg5 = []

        for test_inputs, labels in gen_batch_data_test(test_data, news_as_word_ids, config.batch_size):

            preds = model.predict(test_inputs)
            auc.append(roc_auc_score(labels, preds, 0))
            mrr.append(mrr_score(labels, preds))
            ndcg5.append(ndcg_score(labels, preds, k=5))

        print("AUC in ep {}: {}".format(ep, auc[-1]))
        results.append({'auc': auc, 'mrr': mrr, 'ndcg5': ndcg5})

        '''
        Evaluation: 
        
        Option1: adapt model to process test batches with multiple candidates and output scores
        
        Option2: adapt test generator to mirror format of org. generator (i.e. one candidate)
        '''

        '''
        all_auc = []
        all_mrr = []
        all_ndcg5 = []
        all_ndcg10 = []

        #evaluation
        for m in all_test_index:
            if np.sum(all_test_label[m[0]:m[1]]) != 0 and m[1] < len(click_score):
                all_auc.append(roc_auc_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0]))
                all_mrr.append(mrr_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0]))
                all_ndcg5.append(ndcg_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0], k=5))
                all_ndcg10.append(ndcg_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0], k=10))
        results.append([np.mean(all_auc), np.mean(all_mrr), np.mean(all_ndcg5), np.mean(all_ndcg10)])
        print(np.mean(all_auc), np.mean(all_mrr), np.mean(all_ndcg5), np.mean(all_ndcg10))
        '''

    # save results
    model.save(config.model_save_path + 'npa_keras_1')
    with open(config.results_path + 'npa_keras_1', 'wb') as fout:
        pickle.dump(results, fout)


if __name__ == "__main__":
    train()
    print("gg")