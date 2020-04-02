
# BERT_DIR=/home/sajad/datasets/longsumm/bert_files/
# RAW_DIR_FILES=/home/sajad/datasets/longsumm/files/
# RAW_DIR_JSON=/home/sajad/datasets/longsumm/files/json/

BERT_DIR=/home/sajad/datasets/CSPUBSUM/bert-files/
RAW_DIR_FILES=/home/sajad/datasets/CSPUBSUM/files/
RAW_DIR_JSON=/home/sajad/datasets/CSPUBSUM/files/json/

export CLASSPATH=/home/sajad/packages/tools/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar

for SET in train val test
do

    python preprocess.py -mode tokenize \
                        -raw_path $RAW_DIR_FILES$SET \
                        -save_path $RAW_DIR_FILES$SET/tokenized


    python preprocess.py -mode format_to_lines \
                        -raw_path $RAW_DIR_FILES \
                        -save_path $RAW_DIR_JSON \
                        -n_cpus 1 \
                        -dataset $SET\
                        -use_bert_basic_tokenizer false

    python preprocess.py -mode format_to_bert \
                        -dataset $SET \
                        -raw_path $RAW_DIR_JSON \
                        -save_path $BERT_DIR  \
                        -n_cpus 1 \
                        -log_file ../logs/preprocess.log
done