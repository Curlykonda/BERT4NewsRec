#!/bin/bash
#SBATCH --job-name=npa_cnn_te
#SBATCH -n 8
#SBATCH -t 24:00:00
#SBATCH -p gpu_shared
#SBATCH --mem=60000M


module load pre2019
module load Miniconda3/4.3.27
source activate thesis-user-modelling

python --version

#srun -n 2 -t 00:30:00 --pty bash -il

data=("../Data/DPG_nov19/medium_time_split_n_rnd_users/")
w_emb="./pc_word_embeddings/cc.nl.300.bin"
#pt_news_enc="./BertModelsPT/bert-base-dutch-cased"
art_len=30
SEEDS=(113 42)

enc="wucnn"

TEMP_EMBS=("lte" "nte")
t_act_func="relu"

d_art=400

nie="lin"
#LR=(0.01, 0.001, 0.0001)
lr=0.001
decay_step=25
batch=64

exp_descr="pcp_NpaCNN"

echo "$datapath"

echo "$SEED"

for TE in "${TEMP_EMBS[@]}"
do
  #1
  for K in "${neg_ratios[@]}"
  do
  #1
  python -u main.py --template train_bert_pcp --model_init_seed=$SEED \
  --dataset_path=$data --train_negative_sample_size=$K \
  --news_encoder $enc --dim_art_emb $d_art  --pt_word_emb_path=$w_emb --lower_case=1 \
  --temp_embs=$TE --incl_time_stamp=1 --temp_embs_hidden_units 256 $d_art --temp_embs_act_func $t_act_func \
  --max_article_len=$art_len --nie_layer $nie \
  --lr $lr --decay_step $decay_step --cuda_launch_blocking=1 --train_batch_size=$batch \
  --experiment_description $exp_descr $TE al$art_len k$K s$SEED
  done
done

