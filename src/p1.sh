
export CORENLP_HOME=/home/sajad/packages/tools/stanford-corenlp-full-2018-10-05/

RAW_JSON_PATH=/home/sajad/datasets/CSPUBSUM/json-splits/


python preprocess.py -mode sent_sect \
                    -raw_path $RAW_JSON_PATH