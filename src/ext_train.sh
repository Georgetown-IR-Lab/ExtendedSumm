#!/usr/bin/env bash



################################################################################
##### Data #######
################################################################################

#DATA_PATH=/disk1/sajad/datasets/sci/csabs/bert-files/5l-csabs/
#DATA_PATH=/disk1/sajad/datasets/sci/arxiv//bert-files/5l-new/
#DATA_PATH=/disk1/sajad/datasets/sci/csp/bert-files/5l-rg-labels-whole-3/
#DATA_PATH=/disk1/sajad/datasets/sci/lsum/bert-files/6labels/
#DATA_PATH=/home/sajad/datasets/longsumm/bs-bert-data-ext-phase2/
#DATA_PATH=/disk1/sajad/datasets/sci/arxiv/bert-files/5l-bin/

######### LONGSUM
#DATA_PATH=/home/sajad/datasets/longsumm/bs-data-1700/
#DATA_PATH=/home/sajad/datasets/longsumm/bert_files/
#DATA_PATH=/home/sajad/datasets/longsumm/new-abs-set/bert-files-section/
#DATA_PATH=/home/sajad/datasets/longsumm/new-abs-set/splits/bert-files/4096-ext/
#DATA_PATH=/disk1/sajad/datasets/sci/arxiv/bert-files/arxiv-4096/
#DATA_PATH=/home/sajad/datasets/longsumm/new-abs-set/splits/bert-files/sectioned-512/
#DATA_PATH=/home/sajad/datasets/longsumm/new-abs-set/splits/bert-files/sectioned-512-with-seq-index/
#DATA_PATH=/disk1/sajad/datasets/sci/arxiv/bert-files/512-section-arxiv/
#DATA_PATH=/disk1/sajad/datasets/sci/arxiv/bert-files/arxiv-512/

#BASE_DIR=/disk1/sajad/datasets/sci/arxiv/


#DATA_PATH=/home/sajad/datasets/csp/bert-files/sectioned-512-seqIndex-4k/ #CSPubsum
#DATA_PATH=/home/sajad/datasets/csp/bert-files/sectioned-512-myIndex-4k/ #CSPubsum
#DATA_PATH=/home/sajad/datasets/longsumm/new-abs-set/splits//bert-files/seq-sec/ #Section-prediction
#DATA_PATH=$BASE_DIR/my-format-sample/bert-files/sectioned-512-seqIndex/ # arxiv


######### Pubmed-Long
#DATA_PATH=/disk1/sajad/datasets/sci/pubmed-dataset/bert-files/512-sectioned/ #PubMed
#DATA_PATH=/disk1/sajad/datasets/sci/pubmed-dataset/bert-files/480-seqAllen/ #PubMed
#DATA_PATH=/disk1/sajad/datasets/sci/pubmed-dataset/bert-files-450/512-seqAllen-labels-whole-sectioned/ #PubMed
#DATA_PATH=/disk1/sajad/datasets/sci/pubmed-dataset/bert-files-450/512-plain/ #PubMed
DATA_PATH=/disk1/sajad/datasets/sci/pubmed-dataset/bert-files-450/seqAllen-longfomer-2048-labels-whole-sectioned-chunked/ #PubMed
#DATA_PATH=/disk1/sajad/datasets/sci/pubmed-dataset/bert-files-450/512-seqAllen-whole-sectioned-labels-sectionlabels-chunked/ #PubMed

######### Arxiv-long
#DATA_PATH=/disk1/sajad/datasets/sci/arxiv-long/v1/bert-files/sectioned-myIndex/ #arxiv-long
#DATA_PATH=/disk1/sajad/datasets/sci/arxiv-long/v1/bert-files/512-sectioned-seqAllen-real/ #arxiv-long
#DATA_PATH=/disk1/sajad/datasets/sci/arxiv-long/v1/bert-files/unsegmented/base/ #arxiv-long
#DATA_PATH=/disk1/sajad/datasets/sci/arxiv-long/v1/bert-files/512-sectioned-whole-seqAllen-10labels-withrg-chunked/scibert/ #arxiv-long


################################################################################
##### MODELS #######
################################################################################

#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-first-phase/
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-arxiv-new/
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/sect-prediction-lsum-3level/ # Section prediction
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/pubmed-bertsum-multi-classi-al75/
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/lsum-section-512-classi/


######## ArXiv-Long
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/arxivLong-classi-1536/ #baseline
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/arxivLong-mutli-classi-1536-al75/ #multi
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/arxiv-600-5l-seqAllen/ #multi
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/arxiv-512-multi-unc-seqAllen/ #baseline
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/arxivl-512-sectioned-whole-multi-unc-new/ #baseline


######## Pubmed-Long
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/arxivLong-classi-1536/ #baseline
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/arxivLong-mutli-classi-1536-al75/ #multi
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/arxiv-600-5l-seqAllen/ #multi
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/arxiv-512-multi-unc-seqAllen/ #baseline
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/arxivl-512-sectioned-whole-multi-unc-new/ #baseline
MODEL_PATH=/disk1/sajad/sci-trained-models/presum/pubmedl-512-sectioned-whole-multi-unc/ #baseline
MODEL_PATH=/disk1/sajad/sci-trained-models/presum/pubmedl-512-sectioned-whole-1s/ #baseline



################################################################################
##### CHECKPOINTS –to train from #######
################################################################################
#CHECKPOINT=/disk1/sajad/sci-trained-models/presum/arxiv-first-phase/model_step_30000.pt
#CHECKPOINT=/disk1/sajad/pretrained-bert/bs-ext-cnn/bertext_cnndm_transformer.pt
#CHECKPOINT=/disk1/sajad/sci-trained-models/presum/arxiv-section-512/model_step_6000.pt
#CHECKPOINT=/disk1/sajad/sci-trained-models/presum/arxiv-section-512-classi/model_step_35000.pt



MAX_POS=2048
LOG_DIR=../logs/$(echo $MODEL_PATH | cut -d \/ -f 6).log
mkdir -p ../results/$(echo $MODEL_PATH | cut -d \/ -f 6)
RESULT_PATH_TEST=../results/$(echo $MODEL_PATH | cut -d \/ -f 6)/

python train.py -task ext \
                -mode train \
                -model_name longformer \
                -bert_data_path $DATA_PATH \
                -ext_dropout 0.1 \
                -model_path $MODEL_PATH \
                -lr 2e-3 \
                -visible_gpus 0 \
                -report_every 50 \
                -log_file $LOG_DIR \
                -val_interval 3000 \
                -save_checkpoint_steps 50000 \
                -batch_size 1 \
                -test_batch_size 1000 \
                -max_length 600 \
                -train_steps 100000 \
                -alpha 0.95 \
                -use_interval true \
                -warmup_steps 10000 \
                -max_pos $MAX_POS \
                -result_path_test $RESULT_PATH_TEST \
                -accum_count 2
#                -section_prediction
#                -alpha_mtl 0.43
#                -rg_predictor \

#                -train_from $CHECKPOINT \
# -train_from $CHECKPOINT \

#                -rg_predictor