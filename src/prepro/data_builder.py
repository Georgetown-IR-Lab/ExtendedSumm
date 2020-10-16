import gc
import glob
import hashlib
import json
import operator
import os
import os.path
import pickle
import re
import subprocess
import xml.etree.ElementTree as ET
from os.path import join as pjoin

import pandas as pd
import torch
from multiprocess import Pool
from tqdm import tqdm

from others.logging import logger
from others.tokenization import BertTokenizer, LongformerTokenizer, LongformerTokenizerMine
from others.utils import clean, clean_upper
from prepro.utils import _get_word_ngrams
from utils.rouge_score import evaluate_rouge

nyt_remove_words = ["photo", "graph", "chart", "map", "table", "drawing"]


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()



# Tokenization classes

class LongformerData():
    def __init__(self, args):
        self.args = args
        self.CHUNK_LIMIT=512

        self.tokenizer = LongformerTokenizerMine.from_pretrained('longformer-based-uncased', do_lower_case=True)

        self.sep_token = '</s>'
        self.cls_token = '<s>'
        self.pad_token = '<pad>'
        self.tgt_bos = 'madeupword0000'
        self.tgt_eos = 'madeupword0001'
        self.tgt_sent_split = 'madeupword0002'

        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def cal_token_len(self, src):
        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]
        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        src = src[:self.args.max_src_nsents]
        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]

        return len(src_subtokens)

    def make_chunks(self, src, tgt, sent_labels=None, sent_sections=None, sent_rg_scores=None, chunk_size=512, paper_idx=0):

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]
        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]

        sent_labels = [_sent_labels[i] for i in idxs]
        sent_sections = [sent_sections[i] for i in idxs]
        sent_rg_scores = [sent_rg_scores[i] for i in idxs]

        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]
        sent_rg_scores = sent_rg_scores[:self.args.max_src_nsents]

        sent_sections = sent_sections[:self.args.max_src_nsents]

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)

        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]
        sent_sections = sent_sections[:len(cls_ids)]
        sent_rg_scores = sent_rg_scores[:len(cls_ids)]

        out_sents_labels = []
        out_sents_sections = []
        out_sents_rg_scores = []
        cur_len = 0
        out_src = []
        rg_score = evaluate_rouge([tgt_txt.replace('<q>', ' ')], [' '.join([s for s in src])])[2]
        j = 0
        last_chunk = False
        while j < len(cls_ids):
            if cur_len < chunk_size:
                out_src.append(src[j])
                out_sents_labels.append(sent_labels[j])
                out_sents_sections.append(sent_sections[j])
                out_sents_rg_scores.append(sent_rg_scores[j])
                if j != 0:
                    cur_len += len(src_subtokens[cls_ids[j - 1]:cls_ids[j]])
                else:
                    cur_len += len(src_subtokens[:cls_ids[j]])
                j += 1

            else:
                j = j - 1
                out_src = out_src[:-1]
                out_sents_labels = out_sents_labels[:-1]
                out_sents_sections = out_sents_sections[:-1]
                out_sents_rg_scores = out_sents_rg_scores[:-1]
                out_src1 = out_src.copy()
                out_sents_labels1 = out_sents_labels.copy()
                out_sents_sections1 = out_sents_sections.copy()
                out_sents_rg_scores1 = out_sents_rg_scores.copy()
                out_src.clear()
                out_sents_labels.clear()
                out_sents_sections.clear()
                out_sents_rg_scores.clear()
                cur_len1 = cur_len
                cur_len = 0
                last_chunk = False
                if len(out_src1) == 0:
                    j += 1
                    continue
                yield out_src1, out_sents_labels1, out_sents_sections1, out_sents_rg_scores1, cur_len1, last_chunk, rg_score

            if j == len(cls_ids) - 1:
                out_src1 = out_src.copy()
                out_sents_labels1 = out_sents_labels.copy()
                out_sents_sections1 = out_sents_sections.copy()
                out_sents_rg_scores1 = out_sents_rg_scores.copy()
                out_src.clear()
                out_sents_labels.clear()
                out_sents_sections.clear()
                out_sents_rg_scores.clear()
                cur_len1 = cur_len
                cur_len = 0
                last_chunk = True
                if len(out_src1) == 0:
                    j += 1
                    continue
                yield out_src1, out_sents_labels1, out_sents_sections1, out_sents_rg_scores1, cur_len1, last_chunk, rg_score

    def preprocess_single(self, src, tgt, sent_rg_scores=None, sent_labels=None, sent_sections=None, use_bert_basic_tokenizer=False, is_test=False, debug=False):

        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]
        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            try:
                _sent_labels[l] = 1
            except Exception as e:
                print(e)
                import pdb;pdb.set_trace()
                print(sent_labels)
                print(len(src))

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]

        sent_labels = [_sent_labels[i] for i in idxs]
        sent_sections = [sent_sections[i] for i in idxs]
        if sent_rg_scores is not None:
            sent_rg_scores = [sent_rg_scores[i] for i in idxs]

        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]
        sent_sections = sent_sections[:self.args.max_src_nsents]
        if sent_rg_scores is not None:
            sent_rg_scores = sent_rg_scores[:self.args.max_src_nsents]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []

        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_indxes = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]

        sent_labels = sent_labels[:len(cls_indxes)]
        sent_sections = sent_sections[:len(cls_indxes)]
        if sent_rg_scores is not None:
            sent_rg_scores = sent_rg_scores[:len(cls_indxes)]

        tgt_subtokens_str = 'madeupword0000 ' + ' madeupword0002 '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt
             in tgt]) + ' madeupword0001'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]

        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)
        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        # if debug:
        #     import pdb;pdb.set_trace()

        return src_subtoken_idxs, sent_rg_scores, sent_labels, sent_sections, tgt_subtoken_idxs, segments_ids, cls_indxes, src_txt, tgt_txt

class BertData():
    def __init__(self, args):
        self.CHUNK_LIMIT = 512
        self.args = args
        # self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased',
        #                                                do_lower_case=True)
        #

        if args.model_name == 'scibert':
            self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)

        elif 'bert-base' in args.model_name:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'

        # self.sep_token = '</s>'
        # self.cls_token = '<s>'
        # self.pad_token = '<pad>'
        # self.tgt_bos = 'madeupword0000'
        # self.tgt_eos = 'madeupword0001'
        # self.tgt_sent_split = 'madeupword0002'


        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def make_chunks(self, src, tgt, sent_labels=None, sent_sections=None, sent_rg_scores=None, chunk_size=512, paper_idx=0):

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]
        tgt_txt = ' '.join([' '.join(tt) for tt in tgt])

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]

        sent_labels = [_sent_labels[i] for i in idxs]
        sent_sections = [sent_sections[i] for i in idxs]
        sent_rg_scores = [sent_rg_scores[i] for i in idxs]

        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]
        sent_rg_scores = sent_rg_scores[:self.args.max_src_nsents]

        sent_sections = sent_sections[:self.args.max_src_nsents]

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]
        sent_sections = sent_sections[:len(cls_ids)]
        sent_rg_scores = sent_rg_scores[:len(cls_ids)]

        out_sents_labels = []
        out_sents_sections = []
        out_sents_rg_scores = []
        cur_len=0
        out_src = []
        j=0
        last_chunk = False
        rg_score = evaluate_rouge([tgt_txt.replace('<q>', ' ')], [' '.join([' '.join(s) for s in src])])[2]
        while j < len(cls_ids):
            if cur_len < chunk_size:
                out_src.append(src[j])
                out_sents_labels.append(sent_labels[j])
                out_sents_sections.append(sent_sections[j])
                out_sents_rg_scores.append(sent_rg_scores[j])
                if j!=0:
                    cur_len += len(src_subtokens[cls_ids[j-1]:cls_ids[j]])
                else:
                    cur_len += len(src_subtokens[:cls_ids[j]])
                j+=1

            else:
                j=j-1
                out_src = out_src[:-1]
                out_sents_labels = out_sents_labels[:-1]
                out_sents_sections = out_sents_sections[:-1]
                out_sents_rg_scores = out_sents_rg_scores[:-1]
                out_src1 = out_src.copy()
                out_sents_labels1 = out_sents_labels.copy()
                out_sents_sections1 = out_sents_sections.copy()
                out_sents_rg_scores1 = out_sents_rg_scores.copy()
                out_src.clear()
                out_sents_labels.clear()
                out_sents_sections.clear()
                out_sents_rg_scores.clear()
                cur_len1=cur_len
                cur_len=0
                last_chunk = False
                if len(out_src1) == 0:
                    j+=1
                    continue
                yield out_src1, out_sents_labels1, out_sents_sections1, out_sents_rg_scores1, cur_len1, last_chunk, rg_score

            if j == len(cls_ids) - 1:
                out_src1 = out_src.copy()
                out_sents_labels1 = out_sents_labels.copy()
                out_sents_sections1 = out_sents_sections.copy()
                out_sents_rg_scores1 = out_sents_rg_scores.copy()
                out_src.clear()
                out_sents_labels.clear()
                out_sents_sections.clear()
                out_sents_rg_scores.clear()
                cur_len1 = cur_len
                cur_len = 0
                last_chunk = True
                if len(out_src1) == 0:
                    j+=1
                    continue
                yield out_src1, out_sents_labels1, out_sents_sections1, out_sents_rg_scores1, cur_len1, last_chunk, rg_score


    def preprocess_single(self, src, tgt, sent_rg_scores=None, sent_labels=None, sent_sections=None, use_bert_basic_tokenizer=False, is_test=False, debug=False):

        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]
        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            try:
                _sent_labels[l] = 1
            except Exception as e:
                print(e)
                import pdb;pdb.set_trace()
                print(sent_labels)
                print(len(src))

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]

        sent_labels = [_sent_labels[i] for i in idxs]
        sent_sections = [sent_sections[i] for i in idxs]
        if sent_rg_scores is not None:
            sent_rg_scores = [sent_rg_scores[i] for i in idxs]

        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]
        sent_sections = sent_sections[:self.args.max_src_nsents]
        if sent_rg_scores is not None:
            sent_rg_scores = sent_rg_scores[:self.args.max_src_nsents]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []

        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_indxes = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]

        sent_labels = sent_labels[:len(cls_indxes)]
        sent_sections = sent_sections[:len(cls_indxes)]
        if sent_rg_scores is not None:
            sent_rg_scores = sent_rg_scores[:len(cls_indxes)]

        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt
             in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]

        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)
        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        # if debug:
        #     import pdb;pdb.set_trace()

        return src_subtoken_idxs, sent_rg_scores, sent_labels, sent_sections, tgt_subtoken_idxs, segments_ids, cls_indxes, src_txt, tgt_txt

    def cal_token_len(self, src):
        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]
        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        src = src[:self.args.max_src_nsents]
        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]

        return len(src_subtokens)



# bert-function
def format_to_bert_longsumm(args):
    test_kws = pd.read_csv('train_papers_sects.csv')

    kws = {
        'intro': [kw.strip() for kw in test_kws['intro'].dropna()],
        'related': [kw.strip() for kw in test_kws['related work'].dropna()],
        'exp': [kw.strip() for kw in test_kws['experiments'].dropna()],
        'res': [kw.strip() for kw in test_kws['results'].dropna()],
        'conclusion': [kw.strip() for kw in test_kws['conclusion'].dropna()]
    }

    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['test']

    # labels = pickle.load(open(args.id_files_src + '/sentence_labels/' + str(datasets[0]) + '_labels_csp.p', "rb"))
    labels = []

    # ARXIVIZATION
    bart = args.bart
    check_path_existence(args.save_path)
    for corpus_type in datasets:
        a_lst = []
        c = 0
        for json_f in glob.glob(pjoin(args.raw_path, corpus_type + '.*.json')):
        # for json_f in ['/disk1/sajad/datasets/sci/arxiv-long/v1/bert-files/512-sectioned-myIndex/train.10.bert.pt']:
            real_name = json_f.split('/')[-1]
            c += 1
            a_lst.append(
                (corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt')), kws, bart,
                 labels))
        print("Number of files: " + str(c))


        ##########################
        ###### <DEBUGGING> #######
        ##########################

        for a in a_lst:
            _format_to_bert_origin_sectioned_(a)

        ##########################
        ###### <DEBUGGING> #######
        ##########################

        pool = Pool(args.n_cpus)
        for d in tqdm(pool.imap(_format_to_bert_origin_sectioned_, a_lst), total=len(a_lst)):
            pass

        pool.close()
        pool.join()

    # data = []
    # with open(args.raw_path + '/' + str(datasets[0]) + '.0.json') as f:
    #     for li in f:
    #         data.append(json.loads(li.strip()))

    # pt = []
    # ptctr = 0
    # part = 0
    #
    # bart = args.bart
    #
    # check_path_existence(args.save_path)
    #
    # for i, d in enumerate(data):
    #     pt.append(d)
    #     ptctr += 1
    #     if ptctr > 500:
    #         _fomat_to_bert_section_based_to_text(
    #             (
    #             datasets[0], pt, args, args.save_path + '/' + datasets[0] + '.' + str(part) + '.pt', kws, bart))
    #         part += 1
    #         ptctr = 0
    #         pt.clear()
    # if len(pt) > 0:
    #     _fomat_to_bert_section_based_to_text((datasets[0], pt, args,
    #                                   args.save_path + '/' + datasets[0] + '.' + str(part) + '.pt', kws, bart))

def _format_to_bert_origin_sectioned_(params):
    corpus_type, json_file, args, save_file, kws, bart, labels = params
    is_test = corpus_type == 'test'
    # if (os.path.exists(save_file)):
    #     logger.info('Ignore %s' % save_file)
    #     return


    model_name = args.model_name

    if model_name == 'bert-based' or model_name=='scibert':
        bert = BertData(args)
    elif model_name == 'longformer':
        bert = LongformerData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for j, data in tqdm(enumerate(jobs), total=len(jobs)):
        try:
            paper_id, source, tgt = data['id'], data['src'], data['tgt']
            import pdb;pdb.set_trace()
        except:
            try:
                paper_id, source, tgt = data[0]['id'], data[0]['src'], data[0]['tgt']
            except:
                import pdb;pdb.set_trace()

        source_sents = [s[0] for s in source]
        source_sents = [s for s in source_sents if
                        len(s) > args.min_src_ntokens_per_sent and len(s) < args.max_src_ntokens_per_sent]
        source= [s for s in source if
                        len(s[0]) > args.min_src_ntokens_per_sent and len(s[0]) < args.max_src_ntokens_per_sent]
        # sent_labels = greedy_selection(source_sents, tgt, 10)
        sent_labels = [i for i, s in enumerate(source) if s[-1]==1 and len(s[0]) > args.min_src_ntokens_per_sent and len(s[0]) < args.max_src_ntokens_per_sent]
        sent_rg_scores = [s[2] for i, s in enumerate(source) if len(s[0]) > args.min_src_ntokens_per_sent and len(s[0]) < args.max_src_ntokens_per_sent]

        if (args.lower):

            source_sents = [' '.join(s[0]).lower().split() for s in source]
            # tgt = [' '.join(s[0]).lower().split() for s in tgt] #pubmed
            # import pdb;pdb.set_trace()
            tgt = [' '.join(s).lower().split() for s in tgt] #arxiv
            # tgt = tgt.lower()

        sent_sections = [s[1] for s in source]
        # print(j)
        if True:
            tkn_len = bert.cal_token_len(source_sents)
            debug = False
            if tkn_len > 512:
                for chunk_num, chunk in enumerate(bert.make_chunks(source_sents, tgt, sent_labels=sent_labels, sent_sections=sent_sections, sent_rg_scores=sent_rg_scores, chunk_size=512)):
                    src_chunk, sent_labels_chunk, sent_sections_chunk, sent_rg_scores_chunk,  curlen, is_last_chunk, rg_score = chunk

                    src_label_paralel = zip(src_chunk, sent_labels_chunk)
                    sent_labels_chunk = [i for i, s in enumerate(src_label_paralel) if s[1] == 1]

                    b_data = bert.preprocess_single(src_chunk, tgt, sent_labels=sent_labels_chunk, sent_rg_scores=sent_rg_scores_chunk, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer, is_test=is_test, sent_sections=sent_sections_chunk, debug=debug)



                    if (b_data is None):
                        continue

                    src_subtoken_idxs, sent_rg_scores, sent_labels_chunk, sent_sections_chunk, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
                    sent_sections_chunk = [0 for _ in range(len(sent_sections_chunk))]
                    try:
                        rg_score = evaluate_rouge([tgt_txt.replace('<q>', ' ')], [' '.join(src_txt)])[2] / 100
                        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                                       "src_sent_labels": sent_rg_scores.copy() if sent_rg_scores is not None else sent_sections_chunk,
                                       "sent_labels": sent_labels_chunk.copy(),
                                       "segs": segments_ids, 'clss': cls_ids,
                                       'src_txt': src_txt, "tgt_txt": tgt_txt, "paper_id": paper_id + '___' +str(chunk_num),
                                       "sent_sect_labels": sent_sections_chunk.copy(), "segment_rg_score": rg_score}
                        datasets.append(b_data_dict)

                    except:
                        with open('not_parsed.txt', mode='a') as F:
                            F.write(save_file + ' idx: ' + str(j) + '\n')
                        continue

            else:
                s=0
                # non-sectionized
                b_data = bert.preprocess_single(source_sents, tgt, sent_labels=sent_labels,
                                                use_bert_basic_tokenizer=args.use_bert_basic_tokenizer, is_test=is_test,
                                                sent_sections=sent_sections, debug=False)

                if b_data == None:
                    continue

                src_subtoken_idxs, sent_rg_scores, sent_labels, sent_sections, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
                # sent_rg_scores = [0 for _ in range(len(sent_labels))]
                sent_sections = [0 for _ in range(len(sent_labels))]
                try:
                    b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                                   "src_sent_labels": sent_rg_scores.copy(),
                                   "sent_labels": sent_labels.copy(),
                                   "segs": segments_ids, 'clss': cls_ids,
                                   'src_txt': src_txt, "tgt_txt": tgt_txt, "paper_id": paper_id + '___' + 'single',
                                   "sent_sect_labels": sent_sections.copy()}
                    datasets.append(b_data_dict)
                    # if len(datasets) >= args.shard_size:
                    #     written_files = glob.glob('/'.join(save_file.split('/')[:-1]) + '/' + corpus_type + '*.pt')
                    #     if len(written_files)> 0:
                    #         idxs = [int(w.split('/')[-1].split('.')[1]) for w in written_files]
                    #         indx = sorted(idxs, reverse=True)[0]
                    #         torch.save(datasets, '/'.join(save_file.split('/')[:-1]) + '/' + corpus_type + '.' + str(
                    #             indx + 1) + '.pt')
                    #         datasets = []
                    #     else:
                    #         torch.save(datasets, '/'.join(save_file.split('/')[:-1] + '/' + corpus_type + '.' + str(0) + '.pt'))
                    #
                    #     gc.collect()


                except:
                    with open('not_parsed.txt', mode='a') as F:
                        F.write(save_file + ' idx: ' + str(j) + '\n')
                    continue

    logger.info('Processed instances %data' % len(datasets))
    logger.info('Saving to %s' % save_file)
    # written_files = glob.glob('/'.join(save_file.split('/')[:-1]) + '/' + corpus_type + '*.pt')
    logger.info('Processed instances %data' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    # if len(written_files) > 0:
    #     idxs = [int(w.split('/')[-1].split('.')[1]) for w in written_files]
    #     indx = sorted(idxs, reverse=True)[0]
    #     torch.save(datasets, '/'.join(save_file.split('/')[:-1]) + '/' + corpus_type + '.' + str(indx+1) + '.pt')
    #     datasets = []
    # else:
    #     torch.save(datasets, '/'.join(save_file.split('/')[:-1]) + '/' + corpus_type + '.' + str(0) + '.pt')

    datasets = []
    gc.collect()

def _format_to_bert_origin(params):
    corpus_type, json_file, args, save_file, kws, bart, labels = params
    is_test = corpus_type == 'test'
    # if (os.path.exists(save_file)):
    #     logger.info('Ignore %s' % save_file)
    #     return

    # bert = LongformerData(args)
    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for j, d in tqdm(enumerate(jobs), total=len(jobs)):
        paper_id, source, tgt = d['id'], d['src'], d['tgt']
        source_sents = [s[0] for s in source[:args.max_src_nsents]]
        sent_labels = greedy_selection(source_sents, tgt, 10)
        if (args.lower):
            source_sents = [' '.join(s[0]).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]

        sent_sections = [s[1] for s in source[:args.max_src_nsents]]
        b_data = bert.preprocess(source_sents[:args.max_src_nsents], tgt, sent_labels=sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer, is_test=is_test, sent_sections=sent_sections)
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)
        if (b_data is None):
            continue

        src_subtoken_idxs, sent_rg_scores, sent_labels, sent_sections, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data

        sent_rg_scores = [0 for _ in range(len(sent_labels))]
        sent_sections = [0 for _ in range(len(sent_labels))]
        try:
            b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                           "src_sent_labels": sent_rg_scores.copy(),
                           "sent_labels": sent_labels.copy(),
                           "segs": segments_ids, 'clss': cls_ids,
                           'src_txt': src_txt, "tgt_txt": tgt_txt, "paper_id": paper_id,
                           "sent_sect_labels": sent_sections.copy()}
        except:
            with open('not_parsed.txt', mode='a') as F:
                F.write(save_file + ' idx: ' + str(j) + '\n')
            continue

        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


# line function
def format_longsum_to_lines(args):
    if args.dataset != '':
        corpuses_type = [args.dataset]
    else:
        corpuses_type = ['train', 'val', 'test']

    sections = {}
    for corpus_type in corpuses_type:
        files = []
        for f in glob.glob(args.raw_path + corpus_type + '-10/*.json'):
            files.append(f)
        corpora = {corpus_type: files}
        for corpus_type in corpora.keys():
            a_lst = [(f, args.keep_sect_num) for f in corpora[corpus_type]]
            pool = Pool(7)
            dataset = []
            p_ct = 0

            ##########################
            ###### <DEBUGGING> #######
            ##########################

            # for a in a_lst:
            #     _format_longsum_to_lines_section_based(a)

            ###########################
            ###### </DEBUGGING> #######
            ###########################

            check_path_existence(args.save_path)


            for d in tqdm(pool.imap_unordered(_format_longsum_to_lines_section_based, a_lst), total=len(a_lst)):
                # d_1 = d[1]
                if d is not None:
                    dataset.extend(d[0])
                    # dataset.append(d)

                    if (len(dataset) > args.shard_size):
                        pt_file = "{:s}{:s}.{:d}.json".format(args.save_path + '', corpus_type, p_ct)
                        print(pt_file)
                        with open(pt_file, 'w') as save:
                            # save.write('\n'.join(dataset))
                            save.write(json.dumps(dataset))
                            print('data len: {}'.format(len(dataset)))
                            p_ct += 1
                            dataset = []

            pool.close()
            pool.join()

            if (len(dataset) > 0):
                pt_file = "{:s}{:s}.{:d}.json".format(args.save_path + '', corpus_type, p_ct)
                print(pt_file)
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

    # sections = sorted(sections.items(), key=lambda x: x[1], reverse=True)
    # sections = dict(sections)
    # with open('sect_stat.txt', mode='a') as F:
    #     for s, c in sections.items():
    #         F.write(s + ' --> '+ str(c))
    #         F.write('\n')

def _format_longsum_to_lines(params):
    src_path, keep_sect_num = params

    def load_json(src_json):
        # print(src_json)
        paper = json.load(open(src_json))


        if len(paper['sentences']) < 10 or sum([len(sent) for sent in paper['gold']]) < 10:
            return -1, 0, 0
        id = paper['filename']
        # for sent in paper['sentences']:
        #     tokens = sent[0]
            # if (lower):
            #     tokens = [t.lower() for t in tokens]
            #     sent[0] = tokens

        # for i, sent in enumerate(paper['gold']):
        #     tokens = sent
            # if (lower):
            #     tokens = [t.lower() for t in tokens]
            #     paper['gold'][i] = tokens

        return paper['sentences'], paper['gold'], id

    paper_sents, paper_tgt, id = load_json(src_path)
    if paper_sents == -1:
        return None

    return {'id': id, 'src': paper_sents, 'tgt': paper_tgt}

def _format_longsum_to_lines_section_based(params):
    src_path, keep_sect_num = params

    def load_json(src_json, lower=True):
        paper = json.load(open(src_json))

        # main_sections = _get_main_sections(paper['sentences'])
        # try:
            # main_sections = _get_main_sections(paper['sentences'])
        # except:
        #     main_sections = _get_main_sections(paper['sentences'])
        # sections_text = [''.join([s for s in v if not s.isdigit()]).replace('.','').strip().lower() for v in main_sections.values()]

        if len(paper['sentences']) < 10 or sum([len(sent) for sent in paper['gold']]) < 10:
            return -1, 0, 0

        id = paper['filename']
        # for sent in paper['sentences']:
        #     tokens = sent[0]
            # if keep_sect_num:
            #     sent[1] = _get_section_text(sent[1], main_sections) if 'Abstract' not in sent[1] else sent[1]

        sections = []
        cur_sect = ''
        cur_sect_sents = []
        sections_textual = []
        for i, sent in enumerate(paper['sentences']):
            if i == 0:
                cur_sect = sent[1]
                sections_textual.append(sent[1])
                cur_sect_sents.append(sent)
                continue
            else:
                if cur_sect == sent[1]:
                    cur_sect_sents.append(sent)
                else:
                    cur_sect = sent[1]
                    sections_textual.append(sent[1])
                    sections.append(cur_sect_sents.copy())
                    cur_sect_sents.clear()

        tgts = []
        ids = []
        for j, _ in enumerate(sections):
            tgts.append(paper['gold'])
            ids.append(id + "__" +str(sections_textual[j]))
        return sections, tgts, ids, sections_textual

    paper_sect_sents, paper_tgts, ids, sections_text = load_json(src_path)

    if paper_sect_sents == -1:
        return None

    out = []
    for j, sect_sents in enumerate(paper_sect_sents):
        o = {}
        o['id'] = ids[j]
        o['src'] = sect_sents
        o['tgt'] = paper_tgts[j]
        out.append(o)

    return out, sections_text


## Other utils

def count_dots(txt):
    result = 0
    for char in txt:
        if char == '.':
            result += 1
    return result

def _get_section_id(sect, main_sections):
    if 'abstract' in sect.lower() or 'conclusion' in sect.lower() or 'summary' in sect.lower():
        return sect
    base_sect = sect
    sect_num = sect.split(' ')[0].rstrip('.')
    try:
        int(sect_num)
        return str(int(sect_num))
    except:
        try:
            float(sect_num)
            return str(int(float(sect_num)))
        except:
            if count_dots(sect_num) >= 2:
                sect_num = sect_num.split('.')[0]
                return str(sect_num)
            else:
                return base_sect

def _get_main_sections(sentences):
    out = {}

    for sent in sentences:
        sect_num = sent[1].split(' ')[0].rstrip('.')
        try:
            int(sect_num)
            out[str(sect_num)] = sent[1]
        except:
            pass
    return out

def _get_main_sections_textual(sentences):
    out = {}

    for sent in sentences:
        sect_first_term = sent[1].split(' ')[0].rstrip('.')
        try:
            int(sect_first_term)
            out[str(sect_first_term)] = sent[1]
        except:
            pass

    return out

def _get_section_text(sect, main_sections):
    if 'abstract' in sect.lower() or 'conclusion' in sect.lower() or 'summary' in sect.lower():
        return sect
    base_sect = sect
    sect_num = sect.split(' ')[0].rstrip('.')
    try:
        int(sect_num)
        return sect
    except:
        try:
            float(sect_num)
            int(sect_num.split('.')[0].strip())
            if sect_num.split('.')[0].strip() in main_sections.keys():
                return main_sections[sect_num.split('.')[0].strip()]
            else:
                return base_sect
        except:
            if count_dots(sect_num) >= 2:
                try:
                    int(sect_num.split('.')[0].strip())

                    if sect_num.split('.')[0].strip() in main_sections.keys():
                        return main_sections[sect_num.split('.')[0].strip()]
                    else:
                        return base_sect
                except:
                    return base_sect
            else:
                return base_sect

def check_path_existence(dir):
    if os.path.exists(dir):
        return
    else:
        os.makedirs(dir)

# greedy rg
def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    # import pdb;pdb.set_trace()
    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = (rouge_1 + rouge_2) / 2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            # return selected
            continue
        selected.append(cur_id)
        max_rouge = 0
    return sorted(selected)




