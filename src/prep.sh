#!/usr/bin/env bash

# BERT_DIR=/home/sajad/datasets/longsumm/bert_files/
# RAW_DIR_FILES=/home/sajad/datasets/longsumm/files/
# RAW_DIR_JSON=/home/sajad/datasets/longsumm/files/json/

#BERT_DIR=/disk1/sajad/datasets/sci/csabs/bert-files/5l/
#
#RAW_PATH=/disk1/sajad/datasets/sci/csabs/json/
#
#SECT_LABLE_DIR=/disk1/sajad/datasets/download_google_drive/arxiv/json/
#
#SAVE_JSON=/disk1/sajad/datasets/sci/arxiv/json
# mkdir -p /home/sajad/datasets/talksumm/files/json/

BASE_DIR_JSON=/home/sajad/datasets/longsumm/new-abs-set/
SAVE_JSON=/home/sajad/datasets/longsumm/new-abs-set/bs-jsons-ext/
BERT_DIR=/home/sajad/datasets/longsumm/new-abs-set/bert-files/

id_files_src=/home/sajad/presum/src
#id_files_src=/home/sajad/packages/summarization/PreSumm/src

#export CLASSPATH=/home/sajad/packages/tools/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar



BASE_DIR_JSON=/home/sajad/datasets/longsumm/new-abs-set/5-folds-1700-section/
SAVE_JSON=/home/sajad/datasets/longsumm/new-abs-set/5-folds-1700-section/

#BERT_DIR=/home/sajad/datasets/longsumm/new-abs-set/bert-files-section/

for i in 1 2 3 4 5
do
#    for SET in val test
#    do
#        python3 preprocess.py -mode format_longsum_to_lines \
#                            -dataset $SET \
#                            -raw_path $BASE_DIR_JSON/fold-$i/ \
#                            -save_path $SAVE_JSON/fold-$i/aggregated/  \
#                            -n_cpus 5 \
#                            -log_file ../logs/preprocess.log
#    done
#
    for SET in val test
    do
        python3 preprocess.py -mode format_to_bert_longsumm \
                            -dataset $SET \
                            -id_files_src $id_files_src \
                            -raw_path $SAVE_JSON/fold-$i/aggregated/ \
                            -save_path $BASE_DIR_JSON/fold-$i/bert-files-section/  \
                            -n_cpus 5 \
                            -log_file ../logs/preprocess.log
    done
done




#for SET in train val test
#do
#    python3 preprocess.py -mode format_longsum_to_lines \
#                        -dataset $SET \
#                        -raw_path $BASE_DIR_JSON/ \
#                        -save_path $SAVE_JSON/  \
#                        -n_cpus 5 \
#                        -log_file ../logs/preprocess.log
#done

#for SET in train val test
#do
#    python3 preprocess.py -mode format_to_bert_longsumm \
#                        -bart \
#                        -dataset $SET \
#                        -id_files_src $id_files_src \
#                        -raw_path $SAVE_JSON/ \
#                        -save_path $BERT_DIR/ \
#                        -n_cpus 5 \
#                        -log_file ../logs/preprocess.log
#done