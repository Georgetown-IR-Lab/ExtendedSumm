#!/usr/bin/env bash

#BERT_DATA_PATH=/home/sajad/datasets/longsumm/bs-bert-data-bert/
MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-abs-first-phase/
TEST_FROM=model_step_15000.pt
BERT_DATA_PATH_BRUNELLO=/disk1/sajad/datasets/sci/lsum/bert-files/bs-bert-data-phase1-1100/
BERT_DIR_PATH=/home/sajad/datasets/longsumm/bertUncased-data-1100/

python train.py -task abs \
                -mode test \
                -batch_size 300 \
                -test_batch_size 3000 \
                -bert_data_path $BERT_DIR_PATH \
                -log_file val_abs_bert_cnndm \
                -model_path $MODEL_PATH \
                -sep_optim true \
                -use_interval true \
                -visible_gpus 1 \
                -max_pos 1100 \
                -alpha 0.9 \
                -test_from $MODEL_PATH/$TEST_FROM\
                -result_path train-abs.txt
#                -min_length 600 \
#                -max_length 600 \
