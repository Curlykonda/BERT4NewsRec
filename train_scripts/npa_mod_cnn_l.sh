#!/bin/bash
#SBATCH --job-name=npa_mod_cnn
#SBATCH -n 4
#SBATCH -t 35:00:00
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
hist_len=100
neg_ratios=(4 9) # 4

d_art=400

lr=1e-3
epochs=100
n_users=100000

exp_descr="100k_npa_mod"
COUNTER=0
#######

for LEN in "${art_len[@]}"
do
  for K in "${neg_ratios[@]}"
  do
    echo "$exp_descr al$LEN hl$hist_len k$K lr$lr s$SEED"
      #1
    CUDA_VISIBLE_DEVICES=0,1 python -u main.py --template train_mod_npa --model_init_seed=$SEED --dataset_path=$data \
      --dim_art_emb $d_art --pt_word_emb_path=$w_emb --lower_case=1 \
      --train_negative_sample_size=$K --max_article_len=$LEN \
      --max_hist_len=$hist_len --news_encoder=wucnn --npa_variant=custom \
      --n_users=$n_users --num_epochs=$epochs \
      --lr $lr --cuda_launch_blocking=1 \
      --experiment_description $exp_descr al$LEN hl$hist_len k$K lr$lr s$SEED

    ((COUNTER++))
    echo "$COUNTER"
  done
done

