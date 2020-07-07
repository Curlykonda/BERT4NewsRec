#!/bin/bash
#SBATCH --job-name=npa_cnn_te
#SBATCH -n 8
#SBATCH -t 20:00:00
#SBATCH -p gpu_shared
#SBATCH --mem=60000M


module load pre2019
module load Miniconda3/4.3.27
source activate thesis-user-modelling

python --version

#srun -n 2 -t 00:30:00 --pty bash -il

data=("./Data/DPG_nov19/40k_time_split_n_rnd_users/")
w_emb="./pc_word_embeddings/cc.nl.300.bin"

art_len=30
SEED=$SLURM_ARRAY_TASK_ID

neg_sampler="random"

TEMP_EMBS=("lte" "nte")
t_act_func="relu"
neg_ratios=(4 24 49 99)

enc="wucnn"
d_art=400

nie="lin"
#LR=(0.01, 0.001, 0.0001)
lr=0.001

n_users=40000
exp_descr="40k_NpaCNN_rnd"
COUNTER=0

echo "$datapath"
echo "$SEED"

for TE in "${TEMP_EMBS[@]}"
do
  #1
  for K in "${neg_ratios[@]}"
  do
    #1
    echo "$exp_descr $TE al$art_len k$K s$SEED neg $neg_sampler"

    python -u main.py --template train_bert_pcp --model_init_seed=$SEED --dataset_path=$data \
    --train_negative_sampler_code=$neg_sampler --train_negative_sample_size=$K \
    --news_encoder $enc --dim_art_emb $d_art --pt_word_emb_path=$w_emb --lower_case=1 \
    --temp_embs=$TE --incl_time_stamp=1 --temp_embs_hidden_units 256 $d_art --temp_embs_act_func $t_act_func \
    --max_article_len=$art_len --nie_layer=$nie --n_users=$n_users \
    --lr=$lr --cuda_launch_blocking=1 \
    --experiment_description $exp_descr $TE al$art_len k$K s$SEED

    ((COUNTER++))
    echo "Exp counter: $COUNTER"

  done
done

