#!/bin/bash

sbatch train_scripts/train_bert_npa_te.sh &

sbatch train_scripts/train_bert_npa_pe.sh &