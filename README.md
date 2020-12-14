# ExtendedSumm
This repository contains the implementation details and datasets used in _[On Generating Extended Summaries of Long Documents](http://ir.cs.georgetown.eud)_ paper at the AAAI-21 Workshop on Scholar Document Understanding (SDU-21)



## Conda environment: preliminary setup

To install the required packages, please run conda yml file that you find in the root directory using the following command:

```
conda env create -f environment.yml
```

# How to run...

**IMPORTANT:** The following commands should be run under `src/` directory.

## Dataset

To start with, you first need to download the datasets that are intended to work with the code base. Use the following command to downlaod the datasets. 

```
python dl_dataset.py -dataset pubmed -destination $DESTINATION_FILE
```
Note that `$DESTINATION_FILE` is the directory to tar file, so you have to specify it. For example `$DESTINATION_FILE` can be  `/disk/sajad/datasets/sci/pubmed.tar.gz`

After downloading the dataset, you will need to uncompress it using the following command:

```
tar -xvf pubmed.tar.gz 
```
This will uncompress the tar file into the current directory. The directory will include the single json files of different sets including training, validation, and test. 

**FORMAT** Each file is a json object with keys (and structure) as below:
- "gold"  _(List <List<>>)_: the ground-truth summary of the paper, where the innter list is the tokens associated with each gold summary sentence.

- "sentences" _(List <List<>>)_: the source sentences of the full-text. The inner list contains 5 indeces, each of which represent different fields of the source sentence:
    
    * Index [0]: tokens of the sentences (i.e., list of tokens).
    * Index [1]: textual representation of the section that the sentence belongs to. 
    * Index [2]: RG-L score of the sentence with the gold summary.
    * Index [3]: textual representation of the sentences.
    * Index [4]: oracle label associated with the sentence (0, or 1). 

## Training 
Now the is the time to train the extractive model. The training scripts are inside `train.sh` bash file. To run it on your own machine, let's take a look at the items that you should probably change to fit in your needs:

```
DATA_PATH=/path/to/dataset/512-whole/
MODEL_PATH=/path/to/saved/model/

# Specifiying GPUs either single GPU, or multi-GPU
export CUDA_VISIBLE_DEVICES=0,1


# You don't need to modify these below 
LOG_DIR=../logs/$(echo $MODEL_PATH | cut -d \/ -f 6).log
mkdir -p ../results/$(echo $MODEL_PATH | cut -d \/ -f 6)
RESULT_PATH_TEST=../results/$(echo $MODEL_PATH | cut -d \/ -f 6)/
MAX_POS=512

python train.py -task ext \
                -mode train \
                -model_name scibert \
                -bert_data_path $DATA_PATH \
                -ext_dropout 0.1 \
                -model_path $MODEL_PATH \
                -lr 2e-3 \
                -visible_gpus $CUDA_VISIBLE_DEVICES \
                -val_pred_len 7 \
                -report_every 50 \
                -log_file $LOG_DIR \
                -val_interval 3000 \
                -save_checkpoint_steps 50000 \
                -batch_size 5000 \
                -test_batch_size 1000 \
                -max_length 600 \
                -train_steps 200000 \
                -alpha 0.95 \
                -use_interval true \
                -warmup_steps 10000 \
                -max_pos $MAX_POS \
                -result_path_test $RESULT_PATH_TEST \
                -accum_count 2
 ```

## Inference 
The inference scripts are inside `test.sh` bash file. To run it on your own machine, let's take a look at the items that you should probably change to fit in your needs:

```
#########################
######### Data #########
#########################

# path to the data directory
BERT_DIR=/disk1/sajad/datasets/sci/longsumm/bert-files/2500-segmented-seqLabelled-30/


#########################
######### MODELS#########
#########################

# path to the trained model directory
MODEL_PATH=/disk1/sajad/sci-trained-models/presum/LSUM-2500-segmented-sectioned-multi50-classi-v1/

# path to the best trained model (or the checkpoint that you want to run inference on)
CHECKPOINT=$MODEL_PATH/Recall_BEST_model_s63000_0.4910.pt


export CUDA_VISIBLE_DEVICES=0,1
MAX_POS=2500

mkdir -p ../results/$(echo $MODEL_PATH | cut -d \/ -f 6)
for ST in test
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
 ```
