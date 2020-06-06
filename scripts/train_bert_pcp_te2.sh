#!/bin/bash
#SBATCH --job-name=bertje_te
#SBATCH -n 8
#SBATCH -t 12:00:00
#SBATCH -p gpu_shared
#SBATCH --mem=60000M


module load pre2019
module load Miniconda3/4.3.27
source activate thesis-user-modelling

python --version

#srun -n 2 -t 00:30:00 --pty bash -il

#data=("../Data/DPG_nov19/medium_time_split_most_common/")
#embeddings="../embeddings/cc.nl.300.bin"
pt_news_enc="BERTje"
pt_news_enc_path = "./BertModelsPT/bert-base-dutch-cased"

art_len=128
SEEDS=(113 42)
TEMP_EMBS=("lte" "nte")
method="last_cls"
N=0

d_model=768
t_act_func="relu"

lr=0.002
decay_step=25
exp_descr="pcp"


echo "$datapath"
for SEED in "${SEEDS[@]}"
do
  echo "$SEED"
  for TE in "${TEMP_EMBS[@]}"
  do
  python -u main.py --template train_bert_pcp --model_init_seed=$SEED \
  --pt_news_enc=$pt_news_enc --path_pt_news_enc=$pt_news_enc_path \
  --temp_embs=$TE --incl_time_stamp=1 --temp_embs_hidden_units 256 $d_model --temp_embs_act_func $t_act_func \
  --num_epochs=100 --bert_num_blocks=2 --max_hist_len=100 \
  --max_article_len=$art_len --bert_feature_method $method $N \
  --lr $lr --decay_step $decay_step --cuda_launch_blocking=1 --device="cuda" \
  --experiment_description $exp_descr $TE s$SEED

  python -u main.py --template train_bert_pcp --model_init_seed=$SEED \
  --pt_news_enc=$pt_news_enc --path_pt_news_enc=$pt_news_enc_path \
  --temp_embs=$TE --incl_time_stamp=1 --temp_embs_hidden_units 256 $d_model --temp_embs_act_func $t_act_func \
  --num_epochs=50 --bert_num_blocks=1 --max_hist_len=50 \
  --max_article_len=$art_len --bert_feature_method $method $N \
  --lr $lr --decay_step $decay_step --cuda_launch_blocking=1 \
  --experiment_description $exp_descr $TE s$SEED

  done
done

