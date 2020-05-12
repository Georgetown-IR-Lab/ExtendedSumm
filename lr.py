# import glob
# import json
# from tqdm import tqdm
# from nltk.tokenize import word_tokenize
# from nltk.tokenize import sent_tokenize
#
# files = glob.glob('/disk1/sajad/datasets/sci/arxiv/human-abstracts/train/*.txt')
# avg = [0, 0]
# num=0
# for file in tqdm(files, total=len(files)):
#     with open(file) as f:
#         for l in f:
#             if len(l.strip())>0:
#                 total_toks = len(word_tokenize(l.strip()))
#                 total_sents = len(sent_tokenize(l.strip()))
#                 avg[0] += total_toks
#                 avg[1] += total_sents
#                 num+=1
# print(num)
# print(f'avg toks= {avg[0]/num}, and avg sents= {avg[1] /num}')

import torch.nn as nn
import torch

loss = nn.MSELoss(reduction='mean')
loss1 = nn.MSELoss(reduction='none')
input = torch.randn(2, 2, requires_grad=True)
target = torch.randn(2, 2)
output = loss(input, target)
output1 = loss1(input, target)
s=0