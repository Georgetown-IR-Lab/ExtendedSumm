import glob
import json
import operator
import os

import torch
from tqdm import tqdm

import statistics

from utils.rouge_score import evaluate_rouge

PT_DIRS = "/disk1/sajad/datasets/sci/pubmed-dataset/bert-files-450/512-seqAllen-whole-sectioned-labels-sectionlabels-chunked/"

def check_path_existense(dir):
    if os.path.exists(dir):
        return
    os.makedirs(dir)

for se in ["val", "test"]:

    oracles = {}
    golds = {}
    avg_sents_len = {}
    for j, f in tqdm(enumerate(glob.glob(PT_DIRS + se + '*.pt')), total=len(glob.glob(PT_DIRS + se + '*.pt'))):
        instances = torch.load(f)
        for inst_idx, instance in enumerate(instances):
            sentences = instance['src_txt']
            sent_labels = instance['sent_labels']
            rg_scores = instance['src_sent_labels']
            gold_summary = instance['tgt_txt'].replace('<q>','')
            paper_id = instance['paper_id'].split('__')[0]
            new_labels = []
            instance_picked_up = 0

            for j, s in enumerate(sentences):

                if sent_labels[j] == 1:
                    instance_picked_up +=1
                    if paper_id not in oracles:
                        oracles[paper_id] = s + ' '
                    else:
                        oracles[paper_id] += s
                        oracles[paper_id] += ' '

            if paper_id not in avg_sents_len:
                avg_sents_len[paper_id] = instance_picked_up
            else:
                avg_sents_len[paper_id] += instance_picked_up
            # avg_sents_len[paper_id]=sent_labels.count(1)
            golds[paper_id] = gold_summary

    oracles = dict(sorted(oracles.items()))
    golds = dict(sorted(golds.items()))
    avg_sents_len = dict(sorted(avg_sents_len.items()))
    print('avg oracle sentence number: {}'.format(statistics.mean(avg_sents_len.values())))
    print('median oracle sentence number: {}'.format(statistics.median(avg_sents_len.values())))

    r1, r2, rl = evaluate_rouge(oracles.values(), golds.values())

    print('r1: {}, r2: {}, rl: {}'.format(r1, r2, rl))
#

# our = 20.34
# # ss = [48.99, 48.03, 46.46, 49.16, 12.32]
# # ss = [15.06, 14.76, 14.61, 12.8, 6.30]
# ss = [20.13, 18.04, 19.58, 18.31, 5.49]
#
# for s in ss:
#     print('{}'.format((our-s)/s*100))