#!/usr/bin/env bash


BASE_DIR=/disk1/sajad/datasets/sci/arxiv/


BERT_DIR=/disk1/sajad/datasets/sci/csabs/bert-files/5l-csabs/
#BERT_DIR=/home/sajad/datasets/longsumm/test-set-2020/bert-files/sectioned-512/
#BERT_DIR=/home/sajad/datasets/csp/bert-files/sectioned-512 #CSPubsum
#BERT_DIR=$BASE_DIR/my-format-sample/bert-files/512-section-arxiv/ # arxiv
#BERT_DIR=/disk1/sajad/datasets/sci/pubmed-dataset//bert-files/512-sectioned/ # pubmed-long
#BERT_DIR=/disk1/sajad/datasets/sci/arxiv-long/v1/bert-files/512-sectioned-myIndex/ # pubmed-long


MAX_POS=1536
MODEL_PATH=/disk1/sajad/sci-trained-models/presum/csabs-5l-main/
CHECKPOINT=$MODEL_PATH/model_step_3700.pt
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/sect-prediction-lsum-3level/
#CHECKPOINT=$MODEL_PATH/model_step_15750.pt

RESULT_PATH=../results/$(echo $MODEL_PATH | cut -d \/ -f 6)

for ST in test
do
    python train.py -task ext \
                    -mode test \
                    -test_batch_size 1000 \
                    -bert_data_path $BERT_DIR \
                    -log_file ../logs/val_ext \
                    -model_path $MODEL_PATH \
                    -sep_optim true \
                    -use_interval true \
                    -visible_gpus 0 \
                    -batch_size 1000 \
                    -max_pos $MAX_POS \
                    -max_length 600 \
                    -alpha 0.95 \
                    -min_length 600 \
                    -result_path $RESULT_PATH-pp-new \
                    -test_from $CHECKPOINT \
                    -alpha_mtl 0.60 \
                    -finetune false \
                    -exp_set $ST \
                    -section_prediction
done