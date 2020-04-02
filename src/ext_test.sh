
DATA_PATH=/home/sajad/datasets/CSPUBSUM/bert-files
MAX_POS=1024
RESULT_PATH=../logs/ext1024-bertsum-colins.txt
CHECKPOINT=../models/ext1024-seg-multi-colins/model_step_70000.pt
MODEL_PATH=../models/ext1024-seg-multi-colins/

python train.py -task ext \
                -mode test \
                -batch_size 1 \
                -test_batch_size 1 \
                -bert_data_path $DATA_PATH \
                -log_file ../logs/val_ext \
                -model_path  $MODEL_PATH \
                -sep_optim true \
                -use_interval true \
                -visible_gpus 0 \
                -max_pos $MAX_POS \
                -max_length 500 \
                -alpha 0.95 \
                -min_length 50 \
                -result_path $RESULT_PATH \
                -test_from $CHECKPOINT
