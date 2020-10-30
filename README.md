# SciPreSum
Scientific Summarization System

# How to run...

**IMPORTANT:** The following commands should be run under `src/` directory.

## Conda environment

To install the required packages, please run conda yml file that you find in the root directory using the following command:

```
conda env create -f environment.yml
```

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
This will uncompress the tar file into the current directory. 

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

```
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
