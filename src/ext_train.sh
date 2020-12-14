#!/usr/bin/env bash

################################################################################
##### CHECKPOINTS –to train from #######
################################################################################
#CHECKPOINT=/disk1/sajad/sci-trained-models/presum/arxiv-first-phase/model_step_30000.pt

################################################################################
##### Data #######
################################################################################

DATA_PATH=/disk1/sajad/datasets/sci/longsumm/bert-files/2500-segmented-seqLabelled-30/

################################################################################
##### MODEL #######
################################################################################

MODEL_PATH=/disk1/sajad/sci-trained-models/presum/LSUM-2500-segmented-sectioned-baseline-classi-v1/

################################################################################
##### TRAINING SCRIPT #######
################################################################################
MAX_POS=2500
BSZ=1
export CUDA_VISIBLE_DEVICES=1

LOG_DIR=../logs/$(echo $MODEL_PATH | cut -d \/ -f 6).log
mkdir -p ../results/$(echo $MODEL_PATH | cut -d \/ -f 6)
RESULT_PATH_TEST=../results/$(echo $MODEL_PATH | cut -d \/ -f 6)/


python3 train.py -task ext \
                -mode train \
                -model_name longformer \
                -val_pred_len 30 \
                -bert_data_path $DATA_PATH \
                -ext_dropout 0.1 \
                -model_path $MODEL_PATH \
                -lr 2e-3 \
                -visible_gpus $CUDA_VISIBLE_DEVICES \
                -report_every 50 \
                -log_file $LOG_DIR \
                -val_interval 1000 \
                -save_checkpoint_steps 200000 \
                -batch_size $BSZ \
                -test_batch_size 5000 \
                -max_length 600 \
                -train_steps 200000 \
                -alpha 0.95 \
                -use_interval true \
                -warmup_steps 10000 \
                -max_pos $MAX_POS \
                -result_path_test $RESULT_PATH_TEST \
                -accum_count 2
#                -section_prediction \
#                -alpha_mtl 0.50
#                -rg_predictor

#                -train_from $CHECKPOINT \
# -train_from $CHECKPOINT \

#                -rg_predictor