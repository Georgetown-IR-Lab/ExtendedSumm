#!/usr/bin/env bash

PT_DIRS=/disk1/sajad/datasets/sci/arxiv-long/v1/bert-files/512-sectioned-section-rg/
PT_DIRS_DEST=/disk1/sajad/datasets/sci/arxiv-long/v1/bert-files/512-sectioned-section-rg-relabled/

SET=test
DATASET=arxiv


python3 modify_sent_labels_bertfiles.py -pt_dirs_src $PT_DIRS \
        -write_to $PT_DIRS_DEST

########

#rm /home/sajad/presum/src/$DATASET.$SET.jsonl

#python3 convert_bert_data_to_sequential.py -read_from $PT_DIRS_DEST \
#        -dataset $DATASET \
#        -set $SET

########
#cd /home/sajad/packages/sequential_sentence_classification
#
#sh scripts/predict.sh $DATASET $SET
#
#cd /home/sajad/presum/src

########
#PREDICTED_LABELS=/home/sajad/packages/sequential_sentence_classification/$DATASET.long.$SET.json
#python3 change_labels_sequential.py -read_from $PT_DIRS_DEST \
#        -write_to $PT_DIRS_DEST \
#        -predicted_labels $PREDICTED_LABELS \
#        -set $SET