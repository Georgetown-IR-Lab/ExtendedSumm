#!/usr/bin/env bash



BASE_DIR=/disk1/sajad/datasets/sci/arxiv/

#########################
######### Data #########
#########################

######### Arxiv-long #########
#BERT_DIR=/disk1/sajad/datasets/sci/arxiv-long/v1/bert-files/512-sectioned-seqAllen-real
BERT_DIR=/disk1/sajad/datasets/sci/arxiv-long/v1/bert-files/512-sectioned-test-1-chunked/

# CSP data
#BERT_DIR=/home/sajad/datasets/csp/bert-files/sectioned-512-myIndex/

###### DATA
# ArXiv data
#BERT_DIR=/disk1/sajad/datasets/sci/pubmed-dataset/bert-files/512-sectioned/



#DATA_PATH=/home/sajad/datasets/longsumm/bs-bert-data-ext/
#DATA_PATH=/home/sajad/datasets/longsumm/bs-bert-data-ext-phase2/

#DATA_PATH=/home/sajad/datasets/longsumm/new-abs-set/bert-files/
#DATA_PATH=/home/sajad/datasets/longsumm/new-abs-set/bert-files/
#BERT_DIR=/home/sajad/datasets/longsumm/new-abs-set/splits/bert-files/sectioned-labelled-512/
#BERT_DIR=/home/sajad/datasets/longsumm/new-abs-set/splits/bert-files/sectioned-true-512/
#BERT_DIR=/home/sajad/datasets/longsumm/test-set-2020/bert-files/sectioned-512/ # official test set
#BERT_DIR=/home/sajad/datasets/longsumm/test-set-2020/bert-files/sectioned-512-seqIndex/ # official test set
#BERT_DIR=/home/sajad/datasets/longsumm/new-abs-set/5-folds-section/fold-2/bert-files/whole_files/

#BERT_DIR=/home/sajad/datasets/csp/bert-files/sectioned-512-seqIndex/ #CSPubsum
#BERT_DIR=/home/sajad/datasets/longsumm/new-abs-set/splits//bert-files/seq-sec/ #Section-prediction
#BERT_DIR=$BASE_DIR/my-format-sample/bert-files/sectioned-512-seqIndex/ # arxiv


#########################
######### MODELS#########
#########################

######### Arxiv-long #########
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/arxiv-512-seqAllen-baseline/
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/arxiv-512-seqAllen-multi-al50/
MODEL_PATH=/disk1/sajad/sci-trained-models/presum/arxivl-512-sectioned-baseline-new/




# CSP models
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/pubmed-bertsum-multi-classi-al95/

# Arxiv models
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/arxiv-section-512-classi/

MAX_POS=512
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-first-phase/
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-second-phase/

#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-arxiv-first-phase/
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-first-phase-main/
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/arxiv-first-phase/
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-arxiv-first-phase/
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-arxiv-first-phase/

#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-first-phase-cnn
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-first-phase-cnn
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-first-phase-main/
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/arxiv-section-512-rg/
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-section-512-multi-classi-with-abs-seq-index/
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-section-512-multi-classi-alpha50/
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/arxiv-lsum-section-512-multi-classi/
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/arxiv-lsum-section-512-multi-classi/




## Section predictor
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/arxiv-section-512-multi-classi-alpha50-continue
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-section-512-multi-classi-with-abs-seq-index/
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-section-512-classi/




#CHECKPOINT=$MODEL_PATH/model_step_180000.pt


# CSP Checkpoints
#CHECKPOINT=$MODEL_PATH/model_step_24000.pt # classi-cspubsum
#CHECKPOINT=$MODEL_PATH/model_step_36000.pt # classi-cspubsum-multi
#CHECKPOINT=$MODEL_PATH/model_step_69000.pt # classi-cspubsum-multi-myIndex-alpha50
CHECKPOINT=$MODEL_PATH/model_step_39000.pt # classi-cspubsum-multi-myIndex-alpha75



# ARXIV CH
#CHECKPOINT=$MODEL_PATH/model_step_35000.pt # classi-cspubsum-multi-myIndex-alpha75


#CHECKPOINT=$MODEL_PATH/model_step_100000.pt
#CHECKPOINT=$MODEL_PATH/model_step_34500.pt
#CHECKPOINT=$MODEL_PATH/model_step_30000.pt
#CHECKPOINT=$MODEL_PATH/model_step_18000.pt
#CHECKPOINT=$MODEL_PATH/model_step_6000.pt
#CHECKPOINT=$MODEL_PATH/model_step_12000.pt
#CHECKPOINT=$MODEL_PATH/model_step_36000.pt # multi-seq-alpha50
#CHECKPOINT=$MODEL_PATH/model_step_18000.pt # multi-seq-alpha30
#CHECKPOINT=$MODEL_PATH/model_step_26000.pt # multi-seq-alpha70
#CHECKPOINT=$MODEL_PATH/BEST_model.pt


#

#for i in 1 2 3 4 5
#do
#    for ST in test
#    do
#        BASE_DIR=/home/sajad/datasets/longsumm/new-abs-set/5-folds-section/
#        RESULT_PATH=$BASE_DIR/fold-$i/ext-output/$(echo $MODEL_PATH | cut -d \/ -f 6)/70-nonTriBlock/
#        BERT_DIR=$BASE_DIR/fold-$i/bert-files/sectionID-with-abs/
#        python train.py -task ext \
#                        -mode test \
#                        -test_batch_size 1 \
#                        -bert_data_path $BERT_DIR \
#                        -log_file ../logs/val_ext \
#                        -model_path $MODEL_PATH \
#                        -sep_optim true \
#                        -use_interval true \
#                        -visible_gpus 0 \
#                        -batch_size 1000 \
#                        -max_pos $MAX_POS \
#                        -max_length 600 \
#                        -alpha 0.95 \
#                        -min_length 50 \
#                        -result_path $RESULT_PATH \
#                        -exp_set $ST \
#                        -test_from $CHECKPOINT \
#                        -fold_base_dir $BASE_DIR/fold-$i/\
#                        -log_folds $BASE_DIR/scores/$(echo $MODEL_PATH | cut -d \/ -f 6).txt \
#                        -section_prediction \
#                        -alpha_mtl 0.60
##                        -bart_dir_out $BASE_DIR/fold-$i/ext-output/$(echo $MODEL_PATH | cut -d \/ -f 6)/ \
#    done
#done

mkdir -p ../results/$(echo $MODEL_PATH | cut -d \/ -f 6)
for ST in test
do
    RESULT_PATH=../results/$(echo $MODEL_PATH | cut -d \/ -f 6)/$ST
#    RESULT_PATH=../results/$(echo $MODEL_PATH | cut -d \/ -f 6)/abs-set/$ST.official
#    RESULT_PATH=/home/sajad/datasets/longsum/submission_files/
    python3 train.py -task ext \
                    -mode test \
                    -test_batch_size 10000 \
                    -bert_data_path $BERT_DIR \
                    -log_file ../logs/val_ext \
                    -model_path $MODEL_PATH \
                    -sep_optim true \
                    -use_interval true \
                    -visible_gpus 0 \
                    -batch_size 1 \
                    -max_pos $MAX_POS \
                    -max_length 600 \
                    -alpha 0.95 \
                    -exp_set $ST \
                    -min_length 600 \
                    -result_path $RESULT_PATH \
                    -test_from $CHECKPOINT \
                    -alpha_mtl 0.5 \
                    -model_name scibert
#                        -section_prediction
done
#
##
