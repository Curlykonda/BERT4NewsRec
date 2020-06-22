#!/bin/bash
#SBATCH --job-name=bertje_te
#SBATCH -n 8
#SBATCH -t 20:00:00
#SBATCH -p gpu_shared
#SBATCH --mem=60000M


module load pre2019
module load Miniconda3/4.3.27
source activate thesis-user-modelling

python --version

#srun -n 2 -t 00:30:00 --pty bash -il

data=("./Data/DPG_nov19/medium_time_split_n_rnd_users/")
#embeddings="../embeddings/cc.nl.300.bin"
pt_news_enc="BERTje"
pt_news_enc_path = "./BertModelsPT/bert-base-dutch-cased"

art_len=30
SEED=$SLURM_ARRAY_TASK_ID
TEMP_EMBS=("lte" "nte")
neg_ratios=(4 24 49 99)

nie="lin"

d_model=768
t_act_func="relu"

lr=0.001
decay_step=25

exp_descr="medium"
COUNTER=0

echo "$datapath"

echo "$SEED"
for TE in "${TEMP_EMBS[@]}"
do
  for K in "${neg_ratios[@]}"
  do
   #1
  python -u main.py --template train_bert_pcp --model_init_seed=$SEED \
  --dataset_path=$data --train_negative_sample_size=$K \
  --pt_news_enc=$pt_news_enc --path_pt_news_enc=$pt_news_enc_path \
  --temp_embs=$TE --incl_time_stamp=1 --temp_embs_hidden_units 256 $d_model --temp_embs_act_func $t_act_func \
  --max_article_len=$art_len  --nie_layer $nie \
  --lr $lr --decay_step $decay_step --cuda_launch_blocking=1 \
  --experiment_description $exp_descr $TE al$art_len k$K s$SEED
  ((COUNTER++))
  echo "Exp counter: $COUNTER"
  done
done

