
# BERT_DIR=/home/sajad/datasets/longsumm/bert_files/
# RAW_DIR_FILES=/home/sajad/datasets/longsumm/files/
# RAW_DIR_JSON=/home/sajad/datasets/longsumm/files/json/

BERT_DIR=/home/sajad/datasets/cspubsum/bert-files/
RAW_PATH=/home/sajad/datasets/cspubsum/files/
SECT_LABLE_DIR=/home/sajad/datasets/cspubsum/
SAVE_JSON=/home/sajad/datasets/cspubsum/files/json/
TOKENIZED_PATH=/home/sajad/datasets/cspubsum/files/

export CLASSPATH=/home/sajad/packages/tools/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar

for SET in val
do

    python preprocess.py -mode format_to_lines_mine \
                        -raw_path $RAW_PATH \
                        -dataset $SET \
                        -save_path $SAVE_JSON \
                        -n_cpus 1 \
                        -use_bert_basic_tokenizer false \
                        -map_path MAP_PATH

    python preprocess.py -mode format_to_bert \
                        -dataset $SET \
                        -sect_label_path $SECT_LABLE_DIR \
                        -raw_path $RAW_PATH \
                        -save_path $BERT_DIR  \
                        -n_cpus 1 \
                        -log_file ../logs/preprocess.log
done