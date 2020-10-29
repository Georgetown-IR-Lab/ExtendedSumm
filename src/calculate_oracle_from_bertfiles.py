import argparse
import collections
import glob
import json
import os
import statistics

import torch
from tqdm import tqdm

from utils.rouge_score import evaluate_rouge, evaluate_rouge_avg

parser = argparse.ArgumentParser()
parser.add_argument("-pt_dirs_src", default='')
parser.add_argument("-set", default='')

args = parser.parse_args()

PT_DIRS = args.pt_dirs_src


def check_path_existense(dir):
    if os.path.exists(dir):
        return
    os.makedirs(dir)


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

def Diff(li1, li2):
    return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))

for se in [args.set]:

    oracles = {}
    golds = {}
    avg_sents_len = {}
    for j, f in tqdm(enumerate(glob.glob(PT_DIRS + se + '*.pt')), total=len(glob.glob(PT_DIRS + se + '*.pt'))):
        instances = torch.load(f)
        for inst_idx, instance in enumerate(instances):
            sentences = instance['src_txt']
            sent_labels = instance['sent_labels']
            rg_scores = instance['src_sent_labels']
            gold_summary = instance['tgt_txt'].replace('<q>', '')
            paper_id = instance['paper_id'].split('__')[0]
            new_labels = []
            instance_picked_up = 0

            for j, s in enumerate(sentences):

                if sent_labels[j] == 1:
                    instance_picked_up += 1
                    if paper_id not in oracles:
                        oracles[paper_id] = s + ' '
                    else:
                        oracles[paper_id] += s
                        oracles[paper_id] += ' '
                # else:

            if paper_id not in avg_sents_len:
                avg_sents_len[paper_id] = instance_picked_up
            else:
                avg_sents_len[paper_id] += instance_picked_up
            # avg_sents_len[paper_id]=sent_labels.count(1)
            golds[paper_id] = gold_summary
    # import pdb;pdb.set_trace()
    # oracles['PMC3387377']=''
    for diff in Diff(oracles.keys(), golds.keys()):
        oracles[diff] = ''
        print(diff)
    oracles = dict(sorted(oracles.items()))
    golds = dict(sorted(golds.items()))
    avg_sents_len = dict(sorted(avg_sents_len.items()))
    print('avg oracle sentence number: {}'.format(statistics.mean(avg_sents_len.values())))
    print('median oracle sentence number: {}'.format(statistics.median(avg_sents_len.values())))

    r1, r2, rl = evaluate_rouge_avg(oracles.values(), golds.values())
    # r1, r2, rl = 1,1,1
    print('r1: {}, r2: {}, rl: {}'.format(r1, r2, rl))

    if not is_non_zero_file(PT_DIRS + '/' + 'config.json'):
        config = collections.defaultdict(dict)
        for metric, score in zip(["RG-1", "RG-2", "RG-L"], [r1, r2, rl]):
            config[se][metric] = score

        config[se]["Avg oracle sentence length"] = statistics.mean(avg_sents_len.values())
        config[se]["Median oracle sentence length"] = statistics.median(avg_sents_len.values())
        with open(PT_DIRS + '/' + 'config.json', mode='w') as F:
            json.dump(config, F, indent=4)
    else:
        config = json.load(open(PT_DIRS + '/' + 'config.json'))
        config_all = collections.defaultdict(dict)
        for key, val in config.items():
            for k, v in val.items():
                config_all[key][k] = v

        for metric, score in zip(["RG-1", "RG-2", "RG-L"], [r1, r2, rl]):
            config_all[se][metric] = score
        config_all[se]["Avg oracle sentence length"] = statistics.mean(avg_sents_len.values())
        config_all[se]["Median oracle sentence length"] = statistics.median(avg_sents_len.values())
        with open(PT_DIRS + '/' + 'config.json', mode='w') as F:
            json.dump(config_all, F, indent=4)
