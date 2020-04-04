# DATA_PATH=/disk1/sajad/datasets/download_google_drive/arxiv/bert-files/
DATA_PATH=/disk1/sajad/datasets/sci/arxiv/bert-files/
MAX_POS=1024
MODEL_PATH=/disk1/sajad/sci-trained-models/presum/arxiv-bertsum-multi/

python train.py -task ext \
                -mode train \
                -bert_data_path $DATA_PATH \
                -ext_dropout 0.1 \
                -model_path $MODEL_PATH \
                -lr 2e-3 \
                -visible_gpus 0,1 \
                -report_every 50 \
                -save_checkpoint_steps 10000 \
                -batch_size 1000 \
                -train_steps 70000 \
                -accum_count 2 \
                -log_file ../logs/ext_bert_arxiv \
                -use_interval true \
                -warmup_steps 10000 \
                -max_pos $MAX_POS \
                -section_prediction