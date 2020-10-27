#!/bin/bash
#SBATCH --job-name=mod_qt
#SBATCH -n 4
#SBATCH -t 1:00:00
#SBATCH -p gpu_short
#SBATCH --gres=gpu:1
#SBATCH --mem=60G

module load pre2019
module load Miniconda3/4.3.27
source activate thesis-user-modelling
export PYTHONIOENCODING=utf8

python --version

#srun -n 2 -t 00:30:00 --pty bash -il

data=("./Data/DPG_nov19/10k_time_split_n_rnd_users/")
w_emb="./pc_word_embeddings/cc.nl.300.bin"
model='experiments/10k_cnn_cat_ntev2_al30_k4_lr1e-3_L2_H4_pD0.2_s2_2020-10-23_0'


#pt_news_enc="BERTje"
#pt_news_enc_path="./BertModelsPT/bert-base-dutch-cased"
#
#SEED=$SLURM_ARRAY_TASK_ID
#
#TE=None
#
#art_len=30
#hist_len=100
#
#neg_ratios=(9) # 4 9
#
#n_layers=(2 3 4)
#n_heads=4
#
#lr=1e-4
#n_epochs=100
#
#nie="lin_gelu"
#d_model=768

n_users=10000
COUNTER=0
#################

echo "modify query times"
echo "model path: $model"
  #1
CUDA_VISIBLE_DEVICES=0 python -u main.py --mode mod_query_time \
  --path_test_model=$model --load_config=1 --use_test_model_dir=1 \

((COUNTER++))
echo "Exp counter: $COUNTER"