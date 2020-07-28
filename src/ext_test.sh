#!/usr/bin/env bash

#DATA_PATH=/home/sajad/datasets/longsumm/bs-bert-data-ext/
#DATA_PATH=/home/sajad/datasets/longsumm/bs-bert-data-ext-phase2/

#DATA_PATH=/home/sajad/datasets/longsumm/new-abs-set/bert-files/
DATA_PATH=/home/sajad/datasets/longsumm/new-abs-set/bert-files/


MAX_POS=1700
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-first-phase/
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-second-phase/

MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-arxiv-first-phase/
MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-first-phase-main/
MODEL_PATH=/disk1/sajad/sci-trained-models/presum/arxiv-first-phase/

#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-first-phase-cnn
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-first-phase-cnn
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-first-phase-main/


#CHECKPOINT=$MODEL_PATH/model_step_180000.pt
#CHECKPOINT=$MODEL_PATH/model_step_100000.pt
#CHECKPOINT=$MODEL_PATH/model_step_34500.pt
CHECKPOINT=$MODEL_PATH/model_step_30000.pt
#CHECKPOINT=$MODEL_PATH/BEST_model.pt

RESULT_PATH=../results/$(echo $MODEL_PATH | cut -d \/ -f 6)

#
for i in 1 2 3 4 5
do
    DATA_PATH=/home/sajad/datasets/longsumm/new-abs-set/5-folds-1700-section/fold-$i/bert-files-section
    python train.py -task ext \
                    -mode test \
                    -test_batch_size 1 \
                    -bert_data_path $DATA_PATH \
                    -log_file ../logs/val_ext \
                    -model_path $MODEL_PATH \
                    -sep_optim true \
                    -use_interval true \
                    -visible_gpus 1 \
                    -batch_size 1000 \
                    -max_pos $MAX_POS \
                    -max_length 600 \
                    -alpha 0.95 \
                    -min_length 50 \
                    -result_path $RESULT_PATH \
                    -test_from $CHECKPOINT \
                    -bart_dir_out /home/sajad/datasets/longsumm/new-abs-set/5-folds-1700-section/fold-$i/bart/lsum-finetuned
    #                -alpha_mtl 0.60 \
    #                -section_prediction
done

#python train.py -task ext \
#                -mode test \
#                -test_batch_size 1 \
#                -bert_data_path $DATA_PATH \
#                -log_file ../logs/val_ext \
#                -model_path $MODEL_PATH \
#                -sep_optim true \
#                -use_interval true \
#                -visible_gpus 1 \
#                -batch_size 1000 \
#                -max_pos $MAX_POS \
#                -max_length 600 \
#                -alpha 0.95 \
#                -min_length 50 \
#                -result_path $RESULT_PATH-pp-new \
#                -test_from $CHECKPOINT
##                -alpha_mtl 0.60 \
##                -section_prediction
