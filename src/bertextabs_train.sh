#!/usr/bin/env bash

# Barolo
#BERT_DATA_PATH=/home/sajad/datasets/longsumm/new-abs-set/bert-files/oracle
#BERT_DATA_PATH=/home/sajad/datasets/longsumm/new-abs-set/splits/bert-files/oracle-rg/
BERT_DATA_PATH=/home/sajad/datasets/longsumm/new-abs-set/splits/bert-files/oracle/
MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-oracle-abs-v2/

# Brunello
#BERT_DATA_PATH=/disk1/sajad/datasets/sci/arxiv/bert-files/5l-oracle-1700/
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/arxiv-oracle-abs/


python train.py  -task abs \
                -mode train \
                -bert_data_path $BERT_DATA_PATH \
                -dec_dropout 0.2  \
                -model_path $MODEL_PATH \
                -sep_optim true \
                -lr_bert 0.002 \
                -lr_dec 0.2 \
                -save_checkpoint_steps 5000 \
                -batch_size 1 \
                -test_batch_size 1000 \
                -train_steps 200000 \
                -report_every 50 \
                -val_interval 2500 \
                -accum_count 2 \
                -use_bert_emb true \
                -use_interval true \
                -warmup_steps_bert 20000 \
                -warmup_steps_dec 10000 \
                -max_pos 1700 \
                -visible_gpus 1 \
                -log_file ../logs/abs_bert_cnndm \
                -load_from_extractive /disk1/sajad/sci-trained-models/presum/lsum-arxiv-first-phase/model_step_35000.pt
#                -train_from /disk1/sajad/pretrained-bert/cnn-ext-abs/model_step_148000.pt43