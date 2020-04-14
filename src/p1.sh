
export CLASSPATH=/home/sajad/packages/tools/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar

RAW_JSON_PATH=/home/sajad/datasets/cspubsum/


python preprocess.py -mode sent_sect_mine \
                    -raw_path $RAW_JSON_PATH