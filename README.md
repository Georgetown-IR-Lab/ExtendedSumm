# ExtendedSumm
This repository contains the implementation details and datasets used in _[On Generating Extended Summaries of Long Documents](http://ir.cs.georgetown.eud)_ paper at the AAAI-21 Workshop on Scholar Document Understanding (SDU-21).



### Conda environment: preliminary setup

To install the required packages, please run conda yml file that you find in the root directory using the following command:

```
conda env create -f environment.yml
```

## How to run...

**IMPORTANT:** The following commands should be run under `src/` directory.
 
### Dataset

To start with, you first need to download the datasets that are intended to work with the code base. You can download them from following links: 

| Dataset  | Download Link |
| :-------- | :-------- |
| arXiv-Long  |  [Download](https://drive.google.com/file/d/1p1lb-Urcpds1Bo9piEYwi1DKS9aXxtIv/view?usp=sharing)  |
| PubMed-Long  | [Download](https://drive.google.com/file/d/1T5xbzE_Y_kfxAPzROVbjTz4L_FFQ9EEu/view?usp=sharing) |

After downloading the dataset, you will need to uncompress it using the following command:

```
tar -xvf pubmedL.tar.gz 
```
This will uncompress the `pubmedL` tar file into the current directory. The directory will include the single json files of different sets including training, validation, and test.

**FORMAT** Each paper file is structured within a a json object with the following keys:


- `"id"` _(String)_:  the paper ID
- `"abstract"` _(String)_: the abstract text of the paper. This field is different from "gold" field for the datasets that have different ground-truth than the abstract. 
- `"gold"`  _(List <List<>>)_: the ground-truth summary of the paper, where the inner list is the tokens associated with each gold summary sentence.
- `"sentences"` _(List <List<>>)_: the source sentences of the full-text. The inner list contains 5 indices, each of which represents different fields of the source sentence:
    * Index [0]: tokens of the sentences (i.e., list of tokens).
    * Index [1]: textual representation of the section that the sentence belongs to. 
    * Index [2]: Rouge-L score of the sentence with the gold summary.
    * Index [3]: textual representation of the sentences.
    * Index [4]: oracle label associated with the sentence (0, or 1). 

### Preparing Data

Simply run the `prep.sh` bash script with providing the dataset directory. This script will use two functions to first create aggregated json files, and then preparing them for pretrained language models' usage. 


### Training 
The full training scripts are inside `train.sh` bash file. To run it on your own machine, let's take a look at some items that you should probably change to fit in your needs:
 
```
... 

DATA_PATH=/path/to/dataset/torch-files/
MODEL_PATH=/path/to/saved/model/

# Specifiying GPUs either single GPU, or multi-GPU
export CUDA_VISIBLE_DEVICES=0,1


# You don't need to modify these below 
LOG_DIR=../logs/$(echo $MODEL_PATH | cut -d \/ -f 6).log
mkdir -p ../results/$(echo $MODEL_PATH | cut -d \/ -f 6)
RESULT_PATH_TEST=../results/$(echo $MODEL_PATH | cut -d \/ -f 6)/

MAX_POS=2500

...

 ```

### Inference 
The inference scripts are inside `test.sh` bash file. To run it on your own machine, let's take a look at the items that you should probably change to fit in your needs:

```
...
# path to the data directory
BERT_DIR=/path/to/dataset/torch-files/

# path to the trained model directory
MODEL_PATH=/disk1/sajad/sci-trained-models/presum/LSUM-2500-segmented-sectioned-multi50-classi-v1/

# path to the best trained model (or the checkpoint that you want to run inference on)
CHECKPOINT=$MODEL_PATH/Recall_BEST_model_s63000_0.4910.pt

# GPU machines, either multi or single GPU
export CUDA_VISIBLE_DEVICES=0,1

MAX_POS=2500

...
 ```


## Citation

If you plan to use this work, please cite the following papers:

````
@inproceedings{Sotudeh2021ExtendedSumm,
  title={On Generating Extended Summaries of Long Documents},
  author={Sajad Sotudeh and Arman Cohan and Nazli Goharian},
  booktitle={The AAAI-21 Workshop on Scientific Document Understanding (SDU 2021)},
  year={2021}
}
````

````
@inproceedings{Sotudeh2020LongSumm,
  title={GUIR @ LongSumm 2020: Learning to Generate Long Summaries from Scientific Documents},
  author={Sajad Sotudeh and Arman Cohan and Nazli Goharian},
  booktitle={First Workshop on Scholarly Document Processing (SDP 2020)},
  year={2020}
}
````