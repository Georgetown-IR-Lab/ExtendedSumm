#!/usr/bin/env bash

# BERT_DIR=/home/sajad/datasets/longsumm/bert_files/
# RAW_DIR_FILES=/home/sajad/datasets/longsumm/files/
# RAW_DIR_JSON=/home/sajad/datasets/longsumm/files/json/


#
#SECT_LABLE_DIR=/disk1/sajad/datasets/download_google_drive/arxiv/json/
#
#SAVE_JSON=/disk1/sajad/datasets/sci/arxiv/json
# mkdir -p /home/sajad/datasets/talksumm/files/json/

#BASE_DIR_JSON=/home/sajad/datasets/longsumm/new-abs-set/

id_files_src=/home/sajad/presum/src
#id_files_src=/home/sajad/packages/summarization/PreSumm/src

#export CLASSPATH=/home/sajad/packages/tools/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar


############### Base JSON FILES dir (train/val/test...)
#BASE_DIR=/home/sajad/datasets/longsumm/new-abs-set/splits/
BASE_DIR=/home/sajad/datasets/csp/ # for CSP
#BASE_DIR=/disk1/sajad/datasets/sci/arxiv/json/splits/
#BASE_DIR=/home/sajad/datasets/longsumm/test-set-2020/
#BASE_DIR=/home/sajad/datasets/longsumm/new-abs-set/5-folds-section/ # Cross-validation


############### Raw json files
#SAVE_JSON=/home/sajad/datasets/longsumm/new-abs-set/5-folds-1700-section/
RAW_PATH=$BASE_DIR/json-files/

############### JSON aggregated paths
#SAVE_JSON=/home/sajad/datasets/longsumm/new-abs-set/5-folds-1700-section/
SAVE_JSON=$BASE_DIR/jsons/
#SAVE_JSON=/disk1/sajad/datasets/sci/arxiv/json/splits/jsons/


############### BERT final paths
#BERT_DIR=/home/sajad/datasets/longsumm/new-abs-set/bert-files-section/
#BERT_DIR=/home/sajad/datasets/longsumm/new-abs-set/splits/bert-files/
#BERT_DIR=/disk1/sajad/datasets/sci/arxiv/bert-files/arxiv-4096/
#BERT_DIR=/home/sajad/datasets/longsumm/test-set-2020/bert-files/sectioned-512/ # official test set
BERT_DIR=$BASE_DIR/bert-files/


############### Cross-validation #################

#for i in 1 2 3 4 5
##for i in 5
#do
#    RAW_PATH=$BASE_DIR/fold-$i/ #Cross-validation
#    SAVE_JSON=$BASE_DIR/fold-$i/aggregated/ #Cross-validation
#    BERT_DIR=$BASE_DIR/fold-$i/bert-files/
#    for SET in val test
#    do
#        python3 preprocess.py -mode format_longsum_to_lines \
#                            -dataset $SET \
#                            -raw_path $RAW_PATH/jsons/raw-with-abs/$SET \
#                            -save_path $SAVE_JSON/sectionID-with-abs/  \
#                            -n_cpus 5 \
#                            -keep_sect_num \
#                            -log_file ../logs/preprocess.log
#    done
#
#    for SET in test val
#    do
#        python3 preprocess.py -mode format_to_bert_longsumm \
#                            -dataset $SET \
#                            -id_files_src $id_files_src \
#                            -raw_path $SAVE_JSON/sectionID-with-abs/ \
#                            -save_path $BERT_DIR/sectionID-with-abs/ \
#                            -n_cpus 5 \
#                            -log_file ../logs/preprocess.log
#    done
#done



############### Normal-official test set #################

#BASE_DIR=/home/sajad/datasets/longsumm/test-set-2020/
#SAVE_JSON=/home/sajad/datasets/longsumm/test-set-2020/json-aggregated/sectioned-512/ #official test set
#BERT_DIR=/home/sajad/datasets/longsumm/test-set-2020/bert-files/sectioned-512/ # official test set
#
#for SET in test
#do
#    python3 preprocess.py -mode format_longsum_to_lines \
#                        -dataset $SET \
#                        -raw_path $BASE_DIR/parsed-my-format/with-abs \
#                        -save_path $SAVE_JSON  \
#                        -keep_sect_num \
#                        -n_cpus 5 \
#                        -log_file ../logs/preprocess.log
#done
#
#for SET in test
#do
#    python3 preprocess.py -mode format_to_bert_longsumm \
#                        -bart \
#                        -dataset $SET \
#                        -id_files_src $id_files_src \
#                        -raw_path $SAVE_JSON/ \
#                        -save_path $BERT_DIR \
#                        -n_cpus 5 \
#                        -log_file ../logs/preprocess.log
#done



############### Normal- experiments Longsumm #################

# Arxiv
#BASE_DIR=/disk1/sajad/datasets/sci/arxiv/
#RAW_PATH=$BASE_DIR/my-format-sample/
#SAVE_JSON=$BASE_DIR/my-format-sample/jsons/
#BERT_DIR=$BASE_DIR/my-format-sample/bert-files/512-section-arxiv/


# CSP
#BASE_DIR=/disk1/sajad/datasets/sci/arxiv/
#RAW_PATH=$BASE_DIR/my-format-sample/
#SAVE_JSON=$BASE_DIR/my-format-sample/jsons/
#BERT_DIR=$BASE_DIR/my-format-sample/bert-files/512-section-arxiv/

# Longsumm
#BASE_DIR=/disk1/sajad/datasets/sci/longsumm/new-abs-set/
#RAW_PATH=$BASE_DIR/splits/
#SAVE_JSON=$BASE_DIR/jsons/whole-raw-abs/
#BERT_DIR=$BASE_DIR/bert-files/512-whole-sectioned-raw-abs/

# PubMed-long
BASE_DIR=/disk1/sajad/datasets/sci/pubmed-dataset/
RAW_PATH=$BASE_DIR/single_files/my-format/
SAVE_JSON=$BASE_DIR/jsons-whole/
BERT_DIR=$BASE_DIR/bert-files/512-whole-segmented/

# arxiv-long
#BASE_DIR=/disk1/sajad/datasets/sci/arxiv-long/v1/
#RAW_PATH=$BASE_DIR/my-format-splits/
#SAVE_JSON=$BASE_DIR/jsons/whole/
#BERT_DIR=$BASE_DIR/bert-files/512-whole-sectioned-sectiontext-sectRg/

# csabs
#BASE_DIR=/disk1/sajad/datasets/sci/csabs/
#RAW_PATH=$BASE_DIR/my-format-splits/
#SAVE_JSON=$BASE_DIR/json/
#BERT_DIR=$BASE_DIR/bert-files/5l-csabs/

# main-arxiv
#BASE_DIR=//disk1/sajad/datasets/sci/arxiv-dataset/
#RAW_PATH=$BASE_DIR/single_files/my-format/
#SAVE_JSON=$BASE_DIR/jsons/whole/
#BERT_DIR=$BASE_DIR/bert-files/512-whole-sectioned/



echo "Starting to write aggregated json files..."
echo "-----------------"


for SET in train
do
    python3 preprocess.py -mode format_longsum_to_lines \
                        -save_path $SAVE_JSON  \
                        -n_cpus 22 \
                        -keep_sect_num \
                        -shard_size 300 \
                        -log_file ../logs/preprocess.log \
                        -raw_path $RAW_PATH/$SET/ \
                        -dataset $SET
done

echo "-----------------"
echo "Now starting to write torch files..."
echo "-----------------"

for SET in train
do
    python3 preprocess.py -mode format_to_bert \
                        -bart \
                        -model_name scibert \
                        -lower \
                        -dataset $SET \
                        -id_files_src $id_files_src \
                        -raw_path $SAVE_JSON/ \
                        -save_path $BERT_DIR/ \
                        -n_cpus 24 \
                        -lower \
                        -log_file ../logs/preprocess.log
done
##
