#!/bin/bash

sbatch -a 3-4 train_scripts/train_bert_npa_te.sh &

sbatch -a 3-4 train_scripts/train_bert_npa_pe.sh &

sbatch -a 3-4 train_scripts/train_bert_pe_medium.sh &
sbatch -a 3-4 train_scripts/train_bert_pe_large.sh &

sbatch -a 3-4 train_scripts/train_bert_te_medium.sh &
sbatch -a 3-4 train_scripts/train_bert_te_large.sh &

sbatch -a 3-4 train_scripts/train_npa &

echo "all submitted"


#sbatch -a 1-2 train_scripts/train_bert_npa_te.sh &
#
#sbatch -a 1-2 train_scripts/train_bert_npa_pe.sh &
#
#sbatch -a 1-2 train_scripts/train_bert_pe_medium.sh &
#sbatch -a 1-2 train_scripts/train_bert_pe_large.sh &
#
#sbatch -a 1-2 train_scripts/train_bert_te_medium.sh &
#sbatch -a 1-2 train_scripts/train_bert_te_large.sh &
#
#sbatch -a 1-2 train_scripts/train_npa &