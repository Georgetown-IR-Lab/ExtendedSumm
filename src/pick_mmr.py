import argparse
import csv
import json
import os
import pickle
from datetime import datetime
from multiprocessing.pool import Pool
from operator import itemgetter

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.rouge_score import evaluate_rouge
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_path_existence(dir):
    if os.path.exists(dir):
        return
    else:
        os.makedirs(dir)


parser = argparse.ArgumentParser()
parser.add_argument("-co1", type=float)
parser.add_argument("-co2", type=float)
parser.add_argument("-co3", type=float)
parser.add_argument("-cos", type=str2bool, nargs='?', const=True, default=False)

args = parser.parse_args()


def _report_rouge(predictions, references):
    r1, r2, rl = evaluate_rouge(predictions, references)

    print("Metric\tScore\t95% CI")
    print("ROUGE-1\t{:.4f}\t({:.4f},{:.4f})".format(r1, 0, 0))
    print("ROUGE-2\t{:.4f}\t({:.4f},{:.4f})".format(r2, 0, 0))
    print("ROUGE-L\t{:.4f}\t({:.4f},{:.4f})".format(rl, 0, 0))
    return r1, r2, rl


def get_cosine_sim(embedding_a, embedding_b):
    try:
        embedding_a = torch.from_numpy(embedding_a)
    except:
        embedding_a = embedding_a

    try:
        embedding_b = torch.from_numpy(embedding_b)
    except:
        embedding_b = embedding_b

    cosine_scores = util.pytorch_cos_sim(embedding_a, embedding_b)

    if cosine_scores < 0:
        return 0
    else:
        return cosine_scores.item()

def get_cosine_sim_from_txt(sent_a, sent_b, model):
    embeddings1 = model.encode(sent_a, convert_to_tensor=True)
    embeddings2 = model.encode(sent_b, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    if cosine_scores<0:
        return 0
    else:
        return cosine_scores.item()

def cal_mmr(source_sents, partial_summary, partial_summary_idx, sentence_encodings, partial_summary_sects,
            sent_scores, sent_sects_whole_true, section_contributions,  co1, co2, co3, cos, model, beta=0.1):
    current_mmrs = []
    deductions =[]
    # MMR formula:

    ## MMR = argmax (\alpha Sim(si, D) - \beta max SimSent(si, sj) - \theta max SimSect(sj, sj))
    for idx, sent in enumerate(source_sents):
        cosine_vals = []
        if idx in partial_summary_idx:
            current_mmrs.append(-100000)
            continue

        section_contrib = section_contributions[idx]
        sent_txt = sent
        sent_section = sent_sects_whole_true[idx]

        ######################
        ## calculate first term
        ######################

        first_subterm1 = sent_scores[idx]
        # if cos:
        #     first_subterm2 = get_cosine_sim(sentence_encodings, idx)
        #     first_term = (.95 * first_subterm1) + (.05 * first_subterm2)
        # else:
        #     first_term = first_subterm1

        first_term = first_subterm1

        ######################
        # calculate second term
        ######################
        if co2 > 0:
            # max_rg_score = 0
            # for sent in partial_summary:
            #     rg_score = evaluate_rouge([sent], [sent_txt], type='p')[2]
            #     if rg_score > max_rg_score:
            #         max_rg_score = rg_score
            # second_term = max_rg_score

            for summary_idx, sent in enumerate(partial_summary):
                # cosine_vals.append(get_cosine_sim_from_txt(sent, sent_txt, model))
                cosine_vals.append(get_cosine_sim(sentence_encodings[partial_summary_idx[summary_idx]], sentence_encodings[idx]))

            max_cos = max(cosine_vals)
            second_term = max_cos

        else:
            second_term = 0

        ######################
        # calculate third term
        ######################

        partial_summary_sects_counter = {}
        for sect in partial_summary_sects:
            if sect not in partial_summary_sects_counter:
                partial_summary_sects_counter[sect] = 1
            else:
                partial_summary_sects_counter[sect] += 1

        for sect in partial_summary_sects_counter:
            if sect == sent_section:
                partial_summary_sects_counter[sect] = ((partial_summary_sects_counter[sect] + 1) / 30) * (1/section_contrib) * beta
                # partial_summary_sects_counter[sect] = ((partial_summary_sects_counter[sect] + 1) / 30)
            else:
                partial_summary_sects_counter[sect] = (partial_summary_sects_counter[sect] / 30) * (1/section_contrib) * beta
                # partial_summary_sects_counter[sect] = (partial_summary_sects_counter[sect] / 30)
        third_term = max(partial_summary_sects_counter.values())
        # print(co1, co2, co3)
        mmr_sent = co1 * first_term - co2 * second_term - co3 * third_term
        # mmr_sent = co1 * first_term - co3 * third_term
        # mmr_sent = co1 * first_term - co3 * third_term
        current_mmrs.append(mmr_sent)
        # deductions.append((co2 * second_term,co3 * third_term))
        deductions.append((co3 * third_term))
    return current_mmrs, deductions

def intersection(lst1, lst2):
    lent = 0
    for l in lst1:
        if l in lst2:
            lent+=1
            lst2.remove(l)
    return lent

def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

def _multi_mmr(params):
    p_id, sent_scores, paper_srcs, paper_tgt, sent_sects_whole_true, source_sent_encodings, \
    sent_sects_whole_true, section_textual, sent_true_labels, sent_sectwise_rg, co1, co2, co3, cos, model = params

    section_textual = np.array(section_textual)
    sent_sectwise_rg = np.array(sent_sectwise_rg)
    sent_true_labels = np.array(sent_true_labels)
    PRED_LEN = 30
    oracle_sects = [s for idx, s in enumerate(section_textual) if sent_true_labels[idx] == 1]
    # keep the eligible ids by checking conditions on the sentences
    keep_ids = [idx for idx, s in enumerate(paper_srcs) if len(
        paper_srcs[idx].replace('.', '').replace(',', '').replace('(', '').replace(')', '').replace('-',
                                                                                                    '').replace(
            ':', '').replace(';', '').replace('*', '').split()) > 5 and len(
        paper_srcs[idx].replace('.', '').replace(',', '').replace('(', '').replace(')', '').replace('-',
                                                                                                    '').replace(
            ':', '').replace(';', '').replace('*', '').split()) < 100]

    # filter out ids based on the scores (top50 scores) --> should update keep ids

    source_sents = [s for idx, s in enumerate(paper_srcs) if idx in keep_ids]
    # return len(source_sents)
    sent_scores = sent_scores[keep_ids]
    sent_sects_whole_true = sent_sects_whole_true[keep_ids]
    section_textual = section_textual[keep_ids]
    sent_sectwise_rg = sent_sectwise_rg[keep_ids]
    source_sent_encodings = source_sent_encodings[keep_ids]

    sent_scores = np.asarray([s - 1.00 for s in sent_scores])

    # sent_scores = [s / np.max(sent_scores) for s in sent_scores]
    # sent_scores = np.array(sent_scores)

    top_score_ids = np.argsort(-sent_scores, 0)

    if len(sent_sects_whole_true) == 0:
        return


    pruned = False
    if pruned:
        # keep top 100 sents
        top_score_ids = top_score_ids[:100]
        top_score_ids = [sorted(top_score_ids[:100]).index(s) for s in top_score_ids]
        # only keep if it's above threshold
        # top_score_ids = [t for t in top_score_ids if sent_scores[t] > 0.01]

        sent_scores = sent_scores[top_score_ids]
        sent_sects_whole_true = sent_sects_whole_true[top_score_ids]
        section_textual = section_textual[top_score_ids]
        source_sent_encodings = source_sent_encodings[top_score_ids]
        source_sents = np.asarray(source_sents)
        source_sents = source_sents[top_score_ids]

    section_sent_contrib = [((s / sum(set(sent_sectwise_rg)))+0.001) for s in sent_sectwise_rg]

    summary = []
    summary_sects = []

    # pick the first top-score sentence to start with...
    summary_sects += [section_textual[top_score_ids[0]]]
    summary += [source_sents[top_score_ids[0]]]
    sent_scores_model = sent_scores

    summary_idx = [top_score_ids[0]]

    # augment the summary with MMR until the pred length reach.
    for summary_num in range(1, PRED_LEN):

        MMRs_score, deductions = cal_mmr(source_sents, summary, summary_idx, source_sent_encodings,
                                     summary_sects, sent_scores_model, section_textual, section_sent_contrib, co1, co2, co3, cos, model)


        sent_scores = np.multiply(sent_scores_model, MMRs_score)
        # sent_scores = np.asarray(MMRs_score)

        # autment summary with the updated sent scores
        top_score_ids = np.argsort(-sent_scores, 0)
        summary_idx += [top_score_ids[0]]
        summary += [source_sents[top_score_ids[0]]]
        summary_sects += [section_textual[top_score_ids[0]]]

    summary = [s[1] for s in sorted(zip(summary_idx, summary, summary_sects), key=lambda x: x[0])]
    summary_sects = [s[2] for s in sorted(zip(summary_idx, summary, summary_sects), key=lambda x: x[0])]

    with open('sections.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        if not is_non_zero_file("sections.csv"):
            writer.writerow(["paper_id", "Ours", "Oracle", "Overlap"])
        writer.writerow([str(p_id), str(summary_sects), str(oracle_sects), intersection(summary_sects, oracle_sects) / 10])

    return summary, paper_tgt, p_id
    # except:
    #
    #     with open('not_gotten.txt', mode='a') as ff:
    #         ff.write(str(p_id))
    #         ff.write('\n')
    #     return None


def _bertsum_baseline_1(params):
    # Set model in validating mode.
    def _get_ngrams(n, text):
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    def _block_tri(c, p):
        tri_c = _get_ngrams(3, c.split())
        for s in p:
            tri_s = _get_ngrams(3, s.split())
            if len(tri_c.intersection(tri_s)) > 0:
                return True
        return False

    p_id, sent_scores, paper_srcs, paper_tgt, sent_sects_whole_true, source_sent_encodings, \
    sent_sects_whole_true, section_textual, _,_,co1, co2, co3, cos = params
    # LENGTH_LIMIT= 100
    # print(sent_scores)
    sent_scores = np.asarray([s - 1.00 for s in sent_scores])

    selected_ids_unsorted = np.argsort(-sent_scores, 0)

    try:
        selected_ids_unsorted = [s for s in selected_ids_unsorted if
                                 len(paper_srcs[s].replace('.', '').replace(',', '').replace('(', '').replace(
                                     ')', '').
                                     replace('-', '').replace(':', '').replace(';', '').replace('*',
                                                                                                '').split()) > 5 and
                                 len(paper_srcs[s].replace('.', '').replace(',', '').replace('(', '').replace(
                                     ')', '').
                                     replace('-', '').replace(':', '').replace(';', '').replace('*',
                                                                                                '').split()) < 100]
        _pred = []
        for j in selected_ids_unsorted[1:len(paper_srcs)]:
            if (j >= len(paper_srcs)):
                continue
            candidate = paper_srcs[j].strip()
            # if True:
            if (not _block_tri(candidate, _pred)):
                _pred.append(candidate)
            if (len(_pred) == 30):
                break

        return _pred, paper_tgt, p_id
    except:

        return None


def _bertsum_baseline(params):
    # Set model in validating mode.
    def _get_ngrams(n, text):
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    def _block_tri(c, p):
        tri_c = _get_ngrams(3, c.split())
        for s in p:
            tri_s = _get_ngrams(3, s.split())
            if len(tri_c.intersection(tri_s)) > 0:
                return True
        return False

    def _insert(list, pred, pos):

        # Searching for the position
        for i in range(len(list)):
            if list[i][1] > pos:
                break

        # Inserting pred in the list
        list = list[:i] + [(pred, pos)] + list[i:]
        return list

    p_id, sent_scores, paper_srcs, paper_tgt, sent_sects_whole_true, source_sent_encodings, \
    sent_sects_whole_true, section_textual, _,_,co1, co2, co3, cos = params
    PRED_LEN = 30
    # LENGTH_LIMIT= 100
    # print(sent_scores)
    sent_scores = np.asarray([s - 1.00 for s in sent_scores])

    selected_ids_unsorted = np.argsort(-sent_scores, 0)

    try:
        selected_ids_unsorted = [s for s in selected_ids_unsorted if
                                 len(paper_srcs[s].replace('.', '').replace(',', '').replace('(', '').replace(
                                     ')', '').
                                     replace('-', '').replace(':', '').replace(';', '').replace('*',
                                                                                                '').split()) > 5 and
                                 len(paper_srcs[s].replace('.', '').replace(',', '').replace('(', '').replace(
                                     ')', '').
                                     replace('-', '').replace(':', '').replace(';', '').replace('*',
                                                                                                '').split()) < 100]
        selected_ids_top_10 = selected_ids_unsorted[:PRED_LEN]
        selected_ids_sorted_top10scored = sorted(selected_ids_top_10, reverse=False)
        _pred_final = [(paper_srcs[selected_ids_unsorted[0]].strip(), selected_ids_unsorted[0])]

        picked_up = 1
        picked_up_word_count = len(paper_srcs[selected_ids_unsorted[0]].strip().split())
        for j in selected_ids_sorted_top10scored[1:len(paper_srcs)]:
            if (j >= len(paper_srcs[0])):
                continue
            candidate = paper_srcs[j].strip()
            if True:
                if (not _block_tri(candidate, [x[0] for x in _pred_final])):
                    _pred_final = _insert(_pred_final, candidate, j)
                    # if self.is_joint:
                    #     summary_sect_pred.append(sent_sects_whole[p_id][j])
                    #
                    # try:
                    #     summary_sect_true.append(sent_sects_whole_true[p_id][j])
                    # except:
                    #     import pdb;pdb.set_trace()
                    picked_up += 1
                    picked_up_word_count += len(candidate.split())

            if (picked_up == PRED_LEN):
                break

        _pred_final = sorted(_pred_final, key=itemgetter(1))

        if picked_up < PRED_LEN:
            # if picked_up_word_count < LENGTH_LIMIT:
            #     # it means that some sentences are just skept, sample from the rest of the list
            selected_ids_rest = selected_ids_unsorted[PRED_LEN:]

            for k in selected_ids_rest:
                candidate = paper_srcs[k].strip()
                if (True):
                    if (not _block_tri(candidate, [x[0] for x in _pred_final])):
                        _pred_final = _insert(_pred_final, candidate, k)
                        # if self.is_joint:
                        #     summary_sect_pred.append(sent_sects_whole[p_id][k])
                        # summary_sect_true.append(sent_sects_whole_true[p_id][k])
                        picked_up += 1
                        picked_up_word_count += len(candidate.split())

                if (picked_up == PRED_LEN):
                    break

        _pred_final = sorted(_pred_final, key=itemgetter(1))
        return [x[0] for x in _pred_final], paper_tgt, p_id
    except:

        return None
    # if self.is_joint:
    #     preds_sects[p_id] = summary_sect_pred
    # preds_sects_true[p_id] = summary_sect_true


# saved_list = pickle.load(open("save_list_lsum_val.p", "rb"))
saved_list = pickle.load(open("save_list_arxiv_test_rg.p", "rb"))

a_lst = []

for s, val in saved_list.items():
    val = val + (args.co1, args.co2, args.co3, args.cos, model)
    a_lst.append((val))
    _multi_mmr((val))

pool = Pool(23)
preds = {}
golds = {}
sent_len = []
for d in tqdm(pool.imap(_multi_mmr, a_lst), total=len(a_lst)):
    if d is not None:
        # sent_len.append(d)
        p_id = d[2]
        preds[p_id] = d[0]
        golds[p_id] = d[1]
pool.close()
pool.join()

# import statistics
# print(f'mean: {statistics.mean(sent_len)} and median {statistics.median(sent_len)}')

setting = {"mmr": f"argmax [({args.co1} term1 - {args.co2} max term2 - {args.co3} max term3)]",
           "description": f"with cosine similarity? {args.cos}; .75, .25 (first term)"}

print(f'Calculating RG scores for {len(preds)} papers...')
r1, r2, rl = _report_rouge([' '.join(p) for p in preds.values()], golds.values())

MODEL = 'bertsum_results'
timestamp = datetime.now().strftime("%Y_%m_%d-%I_%p")

check_path_existence("{}/{:4.4f}_{:4.4f}_{:4.4f}__{}/".format(MODEL, r1, r2, rl, timestamp))
with open("{}/{:4.4f}_{:4.4f}_{:4.4f}__{}/info.json".format(MODEL, r1, r2, rl, timestamp), mode='w') as F:
    json.dump(setting, F, indent=4)

can_path = '{}/{:4.4f}_{:4.4f}_{:4.4f}__{}/val-arxivl.source'.format(MODEL, r1, r2, rl, timestamp)
gold_path = '{}/{:4.4f}_{:4.4f}_{:4.4f}__{}/val-arxivl.target'.format(MODEL, r1, r2, rl, timestamp)
save_pred = open(can_path, 'w')
save_gold = open(gold_path, 'w')
for id, pred in preds.items():
    save_pred.write(' '.join(pred).strip().replace('<q>', ' ') + '\n')
    save_gold.write(golds[id].replace('<q>', ' ').strip() + '\n')
