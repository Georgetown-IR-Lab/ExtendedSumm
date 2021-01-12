#!/usr/bin/env bash


############### Normal- experiments Longsumm #################

# Longsumm
#BASE_DIR=/path/to/datasets/single/files/directory/
BASE_DIR=/disk1/sajad/datasets/sci/arxivL/
RAW_PATH=$BASE_DIR/splits-with-sections/
SAVE_JSON=$BASE_DIR/jsons/whole-test/
BERT_DIR=$BASE_DIR/bert-files/2500-segmented-test/

echo "Starting to write aggregated json files..."
echo "-----------------"


#for SET in val
#do
#    python3 preprocess.py -mode format_longsum_to_lines \
#                        -save_path $SAVE_JSON  \
#                        -n_cpus 24 \
#                        -keep_sect_num \
#                        -shard_size 150 \
#                        -log_file ../logs/preprocess.log \
#                        -raw_path $RAW_PATH/$SET/ \
#                        -dataset $SET
#done

echo "-----------------"
echo "Now starting to write torch files..."
echo "-----------------"

for SET in val
do
    python3 preprocess.py -mode format_to_bert \
                        -bart \
                        -model_name longformer \
                        -dataset $SET \
                        -raw_path $SAVE_JSON/ \
                        -save_path $BERT_DIR/ \
                        -n_cpus 24 \
                        -log_file ../logs/preprocess.log \
                        -lower \

done