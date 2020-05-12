#!/usr/bin/env bash

# BERT_DIR=/home/sajad/datasets/longsumm/bert_files/
# RAW_DIR_FILES=/home/sajad/datasets/longsumm/files/
# RAW_DIR_JSON=/home/sajad/datasets/longsumm/files/json/

BERT_DIR=/disk1/sajad/datasets/sci/arxiv//bert-files/5l-rg/

RAW_PATH=/disk1/sajad/datasets/sci/arxiv/

SECT_LABLE_DIR=/disk1/sajad/datasets/download_google_drive/arxiv/json/

SAVE_JSON=/disk1/sajad/datasets/sci/arxiv/json
# mkdir -p /home/sajad/datasets/talksumm/files/json/

export CLASSPATH=/home/sajad/packages/tools/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar

for SET in train
do

#    python preprocess.py -mode format_arxiv_to_lines \
#                        -dataset $SET \
#                        -save_path $SAVE_JSON \

    python3 preprocess.py -mode format_to_bert_arxiv \
                        -dataset $SET \
                        -raw_path $RAW_PATH \
                        -save_path $BERT_DIR  \
                        -n_cpus 5 \
                        -log_file ../logs/preprocess.log
done