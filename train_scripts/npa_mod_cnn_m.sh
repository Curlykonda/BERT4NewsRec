#!/bin/bash
#SBATCH --job-name=m_npa_mod_cnn
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

SEED=$SLURM_ARRAY_TASK_ID

art_len=(30)
neg_ratios=(4 9 24)

d_art=400

lr=0.001
epochs=100
n_users=40000

exp_descr="40k_npa_mod"
COUNTER=0
#######

echo "$exp_descr $datapath"

echo "$SEED"
for LEN in "${art_len[@]}"
do
  for K in "${neg_ratios[@]}"
  do
    echo "$exp_descr l$LEN K$K s$SEED"
      #1
    python -u main.py --template train_mod_npa --model_init_seed=$SEED --dataset_path=$data \
    --dim_art_emb $d_art --pt_word_emb_path=$w_emb --lower_case=1 \
    --train_negative_sample_size=$K --max_article_len=$LEN \
    --n_users=$n_users --num_epochs=$epochs \
    --lr $lr --cuda_launch_blocking=1 \
    --experiment_description $exp_descr l$LEN k$K s$SEED

    ((COUNTER++))
    echo "$COUNTER"
  done
done

