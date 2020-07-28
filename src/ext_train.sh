#!/usr/bin/env bash

#DATA_PATH=/disk1/sajad/datasets/sci/csabs/bert-files/5l/
#DATA_PATH=/disk1/sajad/datasets/sci/arxiv//bert-files/5l-new/
#DATA_PATH=/disk1/sajad/datasets/sci/csp/bert-files/5l-rg-labels-whole-3/
#DATA_PATH=/disk1/sajad/datasets/sci/lsum/bert-files/6labels/
#DATA_PATH=/home/sajad/datasets/longsumm/bs-bert-data-ext-phase2/
#DATA_PATH=/disk1/sajad/datasets/sci/arxiv/bert-files/5l-bin/

#LONGSUM
#DATA_PATH=/home/sajad/datasets/longsumm/bs-data-1700/
#DATA_PATH=/home/sajad/datasets/longsumm/bert_files/
DATA_PATH=/home/sajad/datasets/longsumm/new-abs-set/bert-files-section/

MAX_POS=1700
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-first-phase/
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-arxiv-new/
MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-arxiv-section/
CHECKPOINT=/disk1/sajad/sci-trained-models/presum/arxiv-first-phase/model_step_30000.pt


rm -r $MODEL_PATH
mkdir -p $MODEL_PATH/stats
#rm -r $MODEL_PATH/stats

python train.py -task ext \
                -mode train \
                -bert_data_path $DATA_PATH \
                -ext_dropout 0.1 \
                -model_path $MODEL_PATH \
                -lr 2e-3 \
                -visible_gpus 0,1 \
                -report_every 50 \
                -val_interval 1500 \
                -save_checkpoint_steps 5000 \
                -batch_size 2000 \
                -test_batch_size 1 \
                -train_steps 150000 \
                -log_file arxiv \
                -use_interval true \
                -warmup_steps 10000 \
                -max_pos $MAX_POS \
                -accum_count 2 \
                -train_from $CHECKPOINT
#                -alpha_mtl 0.60 \
#                -section_prediction