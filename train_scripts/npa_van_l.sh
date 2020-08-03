#!/bin/bash
#SBATCH --job-name=npa_van
#SBATCH -n 4
#SBATCH -t 24:00:00
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

art_len=(30)
neg_ratios=(49 74 99) # 4 9

d_art=400

lr=0.001

n_users=100000
exp_descr="100k_npa"
COUNTER=0
##################3

echo "$SEED"
for LEN in "${art_len[@]}"
do
  for K in "${neg_ratios[@]}"
  do
    echo "$exp_descr l$LEN k$K s$SEED"
      #1
    CUDA_VISIBLE_DEVICES=0,1 python -u main.py --template train_npa --model_init_seed=$SEED --dataset_path=$data \
      --dim_art_emb $d_art --pt_word_emb_path=$w_emb --lower_case=1 \
      --max_article_len=$LEN --n_users=$n_users --train_negative_sample_size=$K \
      --lr $lr --cuda_launch_blocking=1 \
      --experiment_description $exp_descr l$LEN k$K s$SEED
    ((COUNTER++))
    echo "$COUNTER"

  done
done

