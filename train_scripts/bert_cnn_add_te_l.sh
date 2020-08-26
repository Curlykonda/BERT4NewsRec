#!/bin/bash
#SBATCH --job-name=npa_cnn_te
#SBATCH -n 4
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

TEMP_EMBS=("nte") #  "ntev2"
t_act_func="relu"

neg_ratios=(4) #

enc="wucnn"
d_art=768

n_layers=(2 3)
n_heads=4
p_dropout=(0.1 0.2 0.3)

nie="lin_gelu"
lr=1e-3
n_epochs=80

n_users=100000
exp_descr="100k_NpaCNN_add"
COUNTER=0
#############################3


echo "$SLURM_JOBID"
echo "$datapath"

for K in "${neg_ratios[@]}"
do
  for TE in "${TEMP_EMBS[@]}"
  do
    for nl in "${n_layers[@]}"
    do
      for p_d in "${p_dropout[@]}"
      do
        echo "$exp_descr $TE al$art_len hl$hist_len k$K lr$lr L$nl H$n_heads pD$p_d s$SEED"
          #1
        CUDA_VISIBLE_DEVICES=0,1 python -u main.py --template train_bert_pcp --model_init_seed=$SEED --dataset_path=$data \
          --bert_num_blocks=$nl --bert_num_heads=$n_heads --bert_dropout=$p_d \
          --train_negative_sample_size=$K \
          --news_encoder $enc --dim_art_emb $d_art --pt_word_emb_path=$w_emb --lower_case=1 \
          --temp_embs=$TE --incl_time_stamp=1 --add_embs_func=add \
          --temp_embs_hidden_units 256 $d_art --temp_embs_act_func $t_act_func \
          --max_article_len=$art_len --nie_layer=$nie --n_users=$n_users \
          --lr=$lr --num_epochs=$n_epochs --cuda_launch_blocking=1 \
          --experiment_description $exp_descr $TE al$art_len k$K lr$lr L$nl H$n_heads pD$p_d s$SEED

        ((COUNTER++))
        echo "Exp counter: $COUNTER"

      done
    done
  done
done

