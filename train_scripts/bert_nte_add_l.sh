#!/bin/bash
#SBATCH --job-name=bertje_nte_l
#SBATCH -n 8
#SBATCH -t 37:00:00
#SBATCH -p gpu_shared
#SBATCH --mem=60000M


module load pre2019
module load Miniconda3/4.3.27
source activate thesis-user-modelling

python --version

#srun -n 2 -t 00:30:00 --pty bash -il

data=("./Data/DPG_nov19/100k_time_split_n_rnd_users/")
#embeddings="../embeddings/cc.nl.300.bin"
pt_news_enc="BERTje"
pt_news_enc_path="./BertModelsPT/bert-base-dutch-cased"

SEED=$SLURM_ARRAY_TASK_ID

art_len=30
neg_ratios=(4 9 24)
lr=0.001

TEMP_EMBS=("nte") # "lte"
t_act_func="relu"

nie="lin_gelu"
d_model=768

n_users=100000
COUNTER=0
#################

exp_descr="100k_add"

for K in "${neg_ratios[@]}"
do
  for TE in "${TEMP_EMBS[@]}"
  do
    echo "$exp_descr $TE al$art_len k$K s$SEED"
      #1
    python -u main.py --template train_bert_pcp --model_init_seed=$SEED --dataset_path=$data \
    --train_negative_sampler_code random_common --train_negative_sample_size=$K \
    --pt_news_enc=$pt_news_enc --path_pt_news_enc=$pt_news_enc_path \
    --temp_embs=$TE --incl_time_stamp=1 --temp_embs_hidden_units 256 $d_model --temp_embs_act_func $t_act_func \
    --max_article_len=$art_len --nie_layer $nie --n_users=$n_users \
    --lr $lr --cuda_launch_blocking=1 \
    --experiment_description $exp_descr $TE al$art_len k$K s$SEED

    ((COUNTER++))
    echo "Exp counter: $COUNTER"
  done
done