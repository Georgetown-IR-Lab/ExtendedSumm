
export CORENLP_HOME=/home/sajad/packages/tools/stanford-corenlp-full-2018-10-05/

RAW_JSON_PATH=/disk1/sajad/datasets/sci/arxiv-dataset/my-format/


python preprocess.py -mode sent_sect \
                    -raw_path $RAW_JSON_PATH