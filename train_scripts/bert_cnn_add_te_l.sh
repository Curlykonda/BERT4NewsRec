#!/bin/bash
#SBATCH --job-name=npa_cnn_te
#SBATCH -n 2
#SBATCH -t 30:00:00
#SBATCH -p gpu_shared
#SBATCH --gres=gpu:2
#SBATCH --mem=60G

module load pre2019
module load Miniconda3/4.3.27
source activate thesis-user-modelling

python --version

#srun -n 2 -t 00:30:00 --pty bash -il

data=("./Data/DPG_nov19/100k_time_split_n_rnd_users/")
w_emb="./pc_word_embeddings/cc.nl.300.bin"

SEED=$SLURM_ARRAY_TASK_ID

art_len=30

TEMP_EMBS=("nte" "ntev2") # "lte"
t_act_func="relu"

neg_ratios=(49 74) # 9 24

enc="wucnn"
d_art=400

n_bert_layers=2

nie="lin_gelu"
LR=(1e-4)
n_epochs=50

n_users=100000
exp_descr="100k_NpaCNN_add"
COUNTER=0
#############################3


echo "$SLURM_JOBID"
echo "$datapath"

for K in "${neg_ratios[@]}"
do
  for lr in "${LR[@]}"
  do
    for TE in "${TEMP_EMBS[@]}"
    do
      echo "$exp_descr $TE al$art_len k$K lr$lr s$SEED" # nl$n_bert_layers

      CUDA_VISIBLE_DEVICES=0,1 python -u main.py --template train_bert_pcp --model_init_seed=$SEED --dataset_path=$data \
        --bert_num_blocks=$n_bert_layers --train_negative_sample_size=$K \
        --news_encoder $enc --dim_art_emb $d_art --pt_word_emb_path=$w_emb --lower_case=1 \
        --temp_embs=$TE --incl_time_stamp=1 --add_embs_func=add \
        --temp_embs_hidden_units 256 $d_art --temp_embs_act_func $t_act_func \
        --max_article_len=$art_len --nie_layer=$nie --n_users=$n_users \
        --lr=$lr --num_epochs=$n_epochs --cuda_launch_blocking=1 \
        --experiment_description $exp_descr $TE al$art_len k$K lr$lr s$SEED

      ((COUNTER++))
      echo "Exp counter: $COUNTER"

    done
  done
done

