
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

DATA_PATH=/disk1/sajad/datasets/sci/cspubsum/bert-files
MAX_POS=1024
MODEL_PATH=/disk1/sajad/sci-trained-models/presum/cspubsum-linear-mt-mlp-alpha6/
CH=model_step_4000.pt

rm -r /disk1/sajad/sci-trained-models/presum/cspubsum-linear-mt-mlp-alpha6/stats
mkdir -p /disk1/sajad/sci-trained-models/presum/cspubsum-linear-mt-mlp-alpha6/stats

python train.py -task ext \
                -mode train \
                -bert_data_path $DATA_PATH \
                -ext_dropout 0.1 \
                -model_path $MODEL_PATH \
                -lr 2e-3 \
                -visible_gpus 0,1 \
                -report_every 50 \
                -save_checkpoint_steps 8000 \
                -batch_size 1000 \
                -train_steps 150000 \
                -accum_count 2 \
                -log_file ../logs/bertmulti_cspubsum \
                -use_interval true \
                -warmup_steps 10000 \
                -max_pos $MAX_POS \
                -section_prediction