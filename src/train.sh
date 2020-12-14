#!/usr/bin/env bash

DATA_PATH=/disk1/sajad/datasets/sci/pubmed-dataset//bert-files/512-whole/
MODEL_PATH=/disk1/sajad/sci-trained-models/presum/pubmed-512-whole-sectioned-baseline-classi/
LOG_DIR=../logs/$(echo $MODEL_PATH | cut -d \/ -f 6).log
mkdir -p ../results/$(echo $MODEL_PATH | cut -d \/ -f 6)
RESULT_PATH_TEST=../results/$(echo $MODEL_PATH | cut -d \/ -f 6)/
MAX_POS=512

export CUDA_VISIBLE_DEVICES=0,1

python train.py -task ext \
                -mode train \
                -model_name scibert \
                -bert_data_path $DATA_PATH \
                -ext_dropout 0.1 \
                -model_path $MODEL_PATH \
                -lr 2e-3 \
                -visible_gpus $CUDA_VISIBLE_DEVICES \
                -val_pred_len 7 \
                -report_every 50 \
                -log_file $LOG_DIR \
                -val_interval 3000 \
                -save_checkpoint_steps 50000 \
                -batch_size 5000 \
                -test_batch_size 1000 \
                -max_length 600 \
                -train_steps 200000 \
                -alpha 0.95 \
                -use_interval true \
                -warmup_steps 10000 \
                -max_pos $MAX_POS \
                -result_path_test $RESULT_PATH_TEST \
                -accum_count 2