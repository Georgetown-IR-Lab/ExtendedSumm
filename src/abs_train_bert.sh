#!/usr/bin/env bash
#BERT_DATA_PATH=/home/sajad/datasets/longsumm/bs-bert-data/
#BERT_DATA_PATH=/home/sajad/datasets/longsumm/bs-bert-data-bert/
MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-oracle-abs-test/
#BERT_DATA_PATH=/home/sajad/datasets/longsumm/bs-bert-data-ext-phase2/
#BERT_DATA_PATH_BRUNELLO=/disk1/sajad/datasets/sci/lsum/bert-files/bs-bert-data-ext-phase2/
BERT_DIR_PATH=/disk1/sajad/datasets/sci/arxiv-long/v1/bert-files/512-sectioned-seqAllen-real/

#BERT_DIR_PATH=/home/sajad/datasets/longsumm/bs-bert-data-phase1-1100/

export CUDA_VISIBLE_DEVICES=0

python train.py  -task abs \
                -mode train \
                -bert_data_path $BERT_DIR_PATH \
                -dec_dropout 0.2  \
                -model_name scibert \
                -test_batch_size 3000 \
                -model_path $MODEL_PATH \
                -sep_optim true \
                -lr_bert 0.002 \
                -lr_dec 0.2 \
                -save_checkpoint_steps 5000 \
                -batch_size 200 \
                -train_steps 100000 \
                -report_every 50 \
                -val_interval 10 \
                -accum_count 5 \
                -use_bert_emb true \
                -use_interval true \
                -warmup_steps_bert 20000 \
                -warmup_steps_dec 10000 \
                -max_pos 512 \
                -visible_gpus 1  \
                -log_file abs_bert_lsum