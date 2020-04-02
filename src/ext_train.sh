DATA_PATH=/home/sajad/datasets/CSPUBSUM/bert-files
MAX_POS=1024
RESULT_PATH=../logs/ext1024-seg-bertsum-colins.txt
# CHECKPOINT=../models/ext18500/model_step_50000.pt
MODEL_PATH=../models/ext1024-bertsum-colins/

python train.py -task ext \
                -mode train \
                -bert_data_path $DATA_PATH \
                -ext_dropout 0,1 \
                -model_path $MODEL_PATH \
                -lr 2e-3 \
                -visible_gpus 0,1 \
                -report_every 50 \
                -save_checkpoint_steps 10000 \
                -batch_size 1000 \
                -train_steps 70000 \
                -accum_count 2 \
                -log_file ../logs/ext_bert \
                -use_interval true \
                -warmup_steps 10000 \
                -max_pos $MAX_POS