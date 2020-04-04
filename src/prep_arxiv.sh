
# BERT_DIR=/home/sajad/datasets/longsumm/bert_files/
# RAW_DIR_FILES=/home/sajad/datasets/longsumm/files/
# RAW_DIR_JSON=/home/sajad/datasets/longsumm/files/json/

BERT_DIR=/disk1/sajad/datasets/download_google_drive/arxiv/bert-files/
RAW_PATH=/disk1/sajad/datasets/download_google_drive/arxiv/
SAVE_JSON=/disk1/sajad/datasets/download_google_drive/arxiv/json/

export CLASSPATH=/home/sajad/packages/tools/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar

for SET in val test
do
    python preprocess.py -mode format_arxiv_to_lines -raw_path $RAW_PATH \
                        -save_path $SAVE_JSON \
                        -n_cpus 15 \
                        -use_bert_basic_tokenizer false \
                        -map_path MAP_PATH \
                        -dataset $SET

    python preprocess.py -mode format_to_bert_arxiv \
                        -dataset $SET \
                        -raw_path $RAW_PATH \
                        -save_path $BERT_DIR  \
                        -n_cpus 15 \
                        -log_file ../logs/preprocess.log
done