#!/usr/bin/env bash

PT_DIRS=/disk1/sajad/datasets/sci/pubmed-dataset//bert-files/512-whole-segmented/
PT_DIRS_DEST=/disk1/sajad/datasets/sci/pubmed-dataset//bert-files/512-whole-relabled-7/

#SET=train
DATASET=longsum


for SET in train
do
    echo "Modifying labels... for $SET"
    python3 modify_sent_labels_bertfiles.py -pt_dirs_src $PT_DIRS \
            -write_to $PT_DIRS_DEST \
            -set $SET \
            -n_sents 7 \
            -greedy True

#
#    echo "Calculating oracle...for $SET"
#    python3 calculate_oracle_from_bertfiles.py -pt_dirs_src $PT_DIRS_DEST \
#            -set $SET
done

#######

#rm /home/sajad/presum/src/$DATASET.$SET.jsonl
##
##for SET in train test val
##do
#python3 convert_bert_data_to_sequential.py -read_from $PT_DIRS_DEST \
#        -dataset $DATASET \
#        -set $SET
##done
#########
#export CUDA_VISIBLE_DEVICES=0
#cd /home/sajad/packages/sequential_sentence_classification
#
#sh scripts/predict.sh $DATASET $SET
#
#cd /home/sajad/presum/src
#
#########
#PREDICTED_LABELS=/home/sajad/packages/sequential_sentence_classification/$DATASET.long.$SET.json
#python3 change_labels_sequential.py -read_from $PT_DIRS_DEST \
#        -write_to $PT_DIRS_DEST \
#        -predicted_labels $PREDICTED_LABELS \
#        -set $SET


## chunk bert files
#python3 spliting_bertfiles.py -pt_dirs_src $PT_DIRS_DEST
