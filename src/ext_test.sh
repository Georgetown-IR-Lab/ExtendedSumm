#!/usr/bin/env bash



server() {
    servername="$1"
    if servername==barolo
    then
        DATA_PATH=/disk1/sajad/datasets/cspubsum/bert-files/
        MAX_POS=1024
        MODEL_PATH=/disk1/sajad/sci-trained-models/presum/cspubsum-bertsum-multi-shared/
    else
        DATA_PATH=/disk1/sajad/datasets/sci/cspubsum/bert-files
        MAX_POS=1024
        MODEL_PATH=/disk1/sajad/sci-trained-models/presum/cspubsum-bertsum-multi-shared4-seperate/
    fi
    echo "$DATA_PATH $MODEL_PATH $MAX_POS"
}


# DATA_PATH=/disk1/sajad/datasets/cspubsum/bert-files/5label
#DATA_PATH=/home/sajad/datasets/csp-ext/bert-files/
DATA_PATH=/disk1/sajad/datasets/sci/csp/bert-files/5l-rg/
MAX_POS=1024
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/csp-bertsum-multi-rg1/
CHECKPOINT=$MODEL_PATH/model_step_144000.pt
#CHECKPOINT=$MODEL_PATH/BEST_model.pt
RESULT_PATH=../logs/linearmt

python train.py -task ext \
                -mode test \
                -batch_size 1 \
                -test_batch_size 1 \
                -bert_data_path $DATA_PATH \
                -log_file ../logs/val_ext \
                -model_path $MODEL_PATH \
                -sep_optim true \
                -use_interval true \
                -visible_gpus 0,1 \
                -max_pos $MAX_POS \
                -max_length 500 \
                -alpha 0.95 \
                -min_length 50 \
                -result_path $RESULT_PATH \
                -test_from $CHECKPOINT \
                -section_prediction