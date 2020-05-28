#!/bin/bash
#SBATCH --job-name=bert4rec_m_common
#SBATCH -n 8
#SBATCH -t 06:00:00
#SBATCH -p gpu_shared
#SBATCH --mem=60000M


module load pre2019
module load Miniconda3/4.3.27
source activate thesis-user-modelling

python --version

#srun -n 2 -t 00:30:00 --pty bash -il

#data=("../Data/DPG_nov19/medium_time_split_most_common/")
#embeddings="../embeddings/cc.nl.300.bin"
pt_news_enc = "./BertModelsPT/bert-base-dutch-cased"
art_len=30
SEEDS=(42)
TEMP_EMBS=("lte" "nte")
method="last_cls"
N=0

d_model=768
t_act_func="relu"

lr=0.01
decay_step=25

echo "$datapath"
for SEED in "${SEEDS[@]}"
do
  echo "$SEED"
  for TE in "${TEMP_EMBS[@]}"
  do
  python -u main.py --template train_bert_pcp --model_init_seed=$SEED --path_pt_news_enc=$pt_news_enc \
  --temp_embs=$TE --max_article_len=$art_len --bert_feature_method $method $N \
  --temp_embs_hidden_units 256 $d_model --temp_embs_act_func $t_act_func \
  --lr $lr --decay_step $decay_step
  done
done


