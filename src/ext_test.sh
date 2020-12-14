#!/usr/bin/env bash



#########################
######### Data #########
#########################

BERT_DIR=/disk1/sajad/datasets/sci/longsumm/bert-files/2500-segmented-seqLabelled-30/


#########################
######### MODELS#########
#########################

MODEL_PATH=/disk1/sajad/sci-trained-models/presum/LSUM-2500-segmented-sectioned-multi50-classi-v1/

CHECKPOINT=$MODEL_PATH/Recall_BEST_model_s63000_0.4910.pt





export CUDA_VISIBLE_DEVICES=0,1

MAX_POS=2500

mkdir -p ../results/$(echo $MODEL_PATH | cut -d \/ -f 6)
for ST in val
do
    RESULT_PATH=../results/$(echo $MODEL_PATH | cut -d \/ -f 6)/$ST
#    RESULT_PATH=../results/$(echo $MODEL_PATH | cut -d \/ -f 6)/abs-set/$ST.official
#    RESULT_PATH=/home/sajad/datasets/longsum/submission_files/
    python3 train.py -task ext \
                    -mode test \
                    -test_batch_size 3000 \
                    -bert_data_path $BERT_DIR \
                    -log_file ../logs/val_ext \
                    -model_path $MODEL_PATH \
                    -sep_optim true \
                    -use_interval true \
                    -visible_gpus $CUDA_VISIBLE_DEVICES \
                    -max_pos $MAX_POS \
                    -max_length 600 \
                    -alpha 0.95 \
                    -exp_set $ST \
                    -pick_top \
                    -min_length 600 \
                    -finetune_bert False \
                    -result_path $RESULT_PATH \
                    -test_from $CHECKPOINT \
                    -model_name longformer \
                    -val_pred_len 30 \
                    -saved_list_name save_lists/lsum-$ST-longformer-multi50-aftersdu.p \
                    -section_prediction \
                    -alpha_mtl 0.50

done

#for ST in test
#do
#    PRED_LEN=20
#    METHOD=_base
#    SAVED_LIST=save_lists/pubmedL-$ST-scibert-bertsum.p
#    C1=.8
#    C2=0
#    C3=0.2
#    python3 pick_mmr.py -co1 $C1 \
#                            -co2 $C2 \
#                            -co3 $C3 \
#                            -set $ST \
#                            -method $METHOD \
#                            -pred_len $PRED_LEN \
#                            -saved_list $SAVED_LIST \
#                            -end
#done
#


