import gc
import glob
import hashlib
import json
import os
import os.path
import pickle
import re
import sys
from os.path import join as pjoin

import pandas as pd
import torch
from multiprocess import Pool
from tqdm import tqdm

from others.logging import logger
from others.tokenization import BertTokenizer, LongformerTokenizerMine
from prepro.utils import _get_word_ngrams
from utils.rouge_score import evaluate_rouge
from datetime import datetime
from uuid import uuid4
import pdb
nyt_remove_words = ["photo", "graph", "chart", "map", "table", "drawing"]


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


# Tokenization classes

class LongformerData():
    def __init__(self, args=None):
        if args is not None:
            self.args = args
        self.CHUNK_LIMIT = 2500

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
        idxs = [i for i, s in enumerate(src) if (len(s[0]) > self.args.min_src_ntokens_per_sent)]
        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        src = src[:self.args.max_src_nsents]
        src_txt = [' '.join(sent[0]) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]

        return len(src_subtokens)

    def cal_token_len_prep(self, src):
        # idxs = [i for i, s in enumerate(src)]
        # src = [src[i] for i in idxs]
        src_txt = [sent[0] for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]

        return len(src_subtokens)

    def make_chunks(self, src_sents_tokens, tgt, sent_labels=None, section_heading_txt=None, sent_rg_scores=None, chunk_size=2500,
                    paper_idx=0, sent_sects_id=None):

        idxs = [i for i, s in enumerate(src_sents_tokens) if (len(s[0]) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src_sents_tokens)
        for l in sent_labels:
            _sent_labels[l] = 1

        src_sents_tokens = [src_sents_tokens[i][:self.args.max_src_ntokens_per_sent] for i in idxs]

        sent_labels = [_sent_labels[i] for i in idxs]
        section_heading_txt = [section_heading_txt[i] for i in idxs]
        sent_rg_scores = [sent_rg_scores[i] for i in idxs]
        sent_sects_id = [sent_sects_id[i] for i in idxs]

        src_sents_tokens = src_sents_tokens[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]
        sent_rg_scores = sent_rg_scores[:self.args.max_src_nsents]
        sent_sects_id = sent_sects_id[:self.args.max_src_nsents]

        section_heading_txt = section_heading_txt[:self.args.max_src_nsents]

        # calculate section importance
        section_rouge = {}
        section_text = ''
        # section_heading_txt = [s[1] for s in src_sents_tokens]

        for idx, sent in enumerate(src_sents_tokens):
            sect_txt = sent[1]

            if idx == 0:
                cursect = sect_txt
                section_text += ' '.join(sent[0])
                section_text += ' '

            if sect_txt == cursect:
                section_text += ' '.join(sent[0])
                section_text += ' '

            else:  # sect has changed...

                # rg_score = evaluate_rouge([tgt_txt.replace('<q>', ' ')], [section_text.strip()])[2]
                rg_score = 0
                section_rouge[cursect] = rg_score
                cursect = sent[1]
                section_text = ''
                section_text += ' '.join(sent[0])

        # rg_score = evaluate_rouge([tgt_txt.replace('<q>', ' ')], [section_text.strip()])[2]
        # section_rouge[cursect] = rg_score
        section_rouge[cursect] = 0

        # if "illustrative @xmath39 matrix example of a @xmath1-symmetric hamiltonian" in section_heading_txt:
        #     print('here')
        #     import pdb;pdb.set_trace()

        src_txt = [' '.join(sent[0]) for sent in src_sents_tokens]
        # section_heading_txt = [s for s in section_heading_txt]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)

        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]
        section_heading_txt = section_heading_txt[:len(cls_ids)]
        sent_rg_scores = sent_rg_scores[:len(cls_ids)]
        sent_sects_id = sent_sects_id[:len(cls_ids)]

        out_sents_labels = []
        out_section_heading_txt = []
        out_sents_rg_scores = []
        out_sent_sects_id = []
        out_sect_sentwise_rg = []
        cur_len = 0
        out_src = []
        rg_score = 0
        j = 0
        last_chunk = False
        while j < len(cls_ids):
            if j == len(cls_ids) - 1:
                out_src1 = out_src.copy()
                out_sents_labels1 = out_sents_labels.copy()
                out_section_heading_txt1 = out_section_heading_txt.copy()
                out_sents_rg_scores1 = out_sents_rg_scores.copy()
                out_sent_sects_id1 = out_sent_sects_id.copy()
                out_src.clear()
                out_sect_sentwise_rg.clear()
                out_sents_labels.clear()
                out_section_heading_txt.clear()
                out_sents_rg_scores.clear()
                out_sent_sects_id.clear()
                cur_len1 = cur_len
                cur_len = 0
                last_chunk = True
                if len(out_src1) == 0:
                    j += 1
                    continue
                yield out_src1, out_sents_labels1, out_section_heading_txt1, out_sents_rg_scores1, cur_len1, last_chunk, rg_score, section_rouge, out_sent_sects_id1

            if cur_len < chunk_size:
                out_src.append((src_sents_tokens[j][0], src_sents_tokens[j][1], src_sents_tokens[j][2]))
                out_sect_sentwise_rg.append(section_rouge[src_sents_tokens[j][1]])
                out_sents_labels.append(sent_labels[j])
                out_section_heading_txt.append(section_heading_txt[j])
                out_sents_rg_scores.append(sent_rg_scores[j])
                out_sent_sects_id.append(sent_sects_id[j])
                if j != 0:
                    cur_len += len(src_subtokens[cls_ids[j - 1]:cls_ids[j]])
                else:
                    cur_len += len(src_subtokens[:cls_ids[j]])
                j += 1

            else:
                j = j - 1
                out_src = out_src[:-1]
                out_sect_sentwise_rg = out_sect_sentwise_rg[:-1]
                out_sents_labels = out_sents_labels[:-1]
                out_section_heading_txt = out_section_heading_txt[:-1]
                out_sents_rg_scores = out_sents_rg_scores[:-1]
                out_sent_sects_id = out_sent_sects_id[:-1]
                out_src1 = out_src.copy()
                out_sents_labels1 = out_sents_labels.copy()
                out_section_heading_txt1 = out_section_heading_txt.copy()
                out_sents_rg_scores1 = out_sents_rg_scores.copy()
                out_sent_sects_id1 = out_sent_sects_id.copy()
                out_src.clear()
                out_sents_labels.clear()
                out_section_heading_txt.clear()
                out_sents_rg_scores.clear()
                cur_len1 = cur_len
                cur_len = 0
                last_chunk = False
                if len(out_src1) == 0:
                    j += 1
                    continue

                yield out_src1, out_sents_labels1, out_section_heading_txt1, out_sents_rg_scores1, cur_len1, last_chunk, rg_score, section_rouge, out_sent_sects_id1

    def preprocess_single(self, src, tgt, sent_rg_scores=None, sent_labels=None, sent_sections=None,
                          use_bert_basic_tokenizer=False, is_test=False, section_rgs=None, debug=False):


        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s[0]) for s in src]
        idxs = [i for i, s in enumerate(src) if (len(s[0]) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            try:
                _sent_labels[l] = 1
            except Exception as e:
                print(e)
                import pdb;
                pdb.set_trace()
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

        src_txt = [' '.join(sent[0]) for sent in src]
        src_sent_token_count = [self.cal_token_len([(sent[0], 'test')]) for sent in src]

        src_sents_sections = [sent[1] for sent in src]
        src_sents_number = [sent[2] for sent in src]

        try:
            sents_sect_wise_rg = [section_rgs[sect] for sect in src_sents_sections]
        except:
            import pdb;pdb.set_trace()
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

        return src_subtoken_idxs, sent_rg_scores, sent_labels, sent_sections, tgt_subtoken_idxs, segments_ids, cls_indxes, src_txt, tgt_txt, sents_sect_wise_rg, src_sents_number, src_sent_token_count

class BertData():
    def __init__(self, args):
        self.CHUNK_LIMIT = 512
        self.args = args

        if args.model_name == 'scibert':
            self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)

        elif 'bert-base' in args.model_name or 'bert-large' in args.model_name:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'

        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def make_chunks(self, src_sents_tokens, tgt, sent_labels=None, section_heading_txt=None, sent_rg_scores=None, chunk_size=512,
                    paper_idx=0, sent_sects_id=None):

        idxs = [i for i, s in enumerate(src_sents_tokens) if (len(s[0]) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src_sents_tokens)
        for l in sent_labels:
            _sent_labels[l] = 1

        src_sents_tokens = [src_sents_tokens[i][:self.args.max_src_ntokens_per_sent] for i in idxs]

        sent_labels = [_sent_labels[i] for i in idxs]
        section_heading_txt = [section_heading_txt[i] for i in idxs]
        sent_rg_scores = [sent_rg_scores[i] for i in idxs]
        sent_sects_id = [sent_sects_id[i] for i in idxs]

        src_sents_tokens = src_sents_tokens[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]
        sent_rg_scores = sent_rg_scores[:self.args.max_src_nsents]
        sent_sects_id = sent_sects_id[:self.args.max_src_nsents]

        section_heading_txt = section_heading_txt[:self.args.max_src_nsents]

        # calculate section importance
        section_rouge = {}
        section_text = ''
        # section_heading_txt = [s[1] for s in src_sents_tokens]

        for idx, sent in enumerate(src_sents_tokens):
            sect_txt = sent[1]

            if idx == 0:
                cursect = sect_txt
                section_text += ' '.join(sent[0])
                section_text += ' '

            if sect_txt == cursect:
                section_text += ' '.join(sent[0])
                section_text += ' '

            else:  # sect has changed...

                # rg_score = evaluate_rouge([tgt_txt.replace('<q>', ' ')], [section_text.strip()])[2]
                rg_score = 0
                section_rouge[cursect] = rg_score
                cursect = sent[1]
                section_text = ''
                section_text += ' '.join(sent[0])

        # rg_score = evaluate_rouge([tgt_txt.replace('<q>', ' ')], [section_text.strip()])[2]
        # section_rouge[cursect] = rg_score
        section_rouge[cursect] = 0

        # if "illustrative @xmath39 matrix example of a @xmath1-symmetric hamiltonian" in section_heading_txt:
        #     print('here')
        #     import pdb;pdb.set_trace()

        src_txt = [' '.join(sent[0]) for sent in src_sents_tokens]
        # section_heading_txt = [s for s in section_heading_txt]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)

        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]
        section_heading_txt = section_heading_txt[:len(cls_ids)]
        sent_rg_scores = sent_rg_scores[:len(cls_ids)]
        sent_sects_id = sent_sects_id[:len(cls_ids)]

        out_sents_labels = []
        out_section_heading_txt = []
        out_sents_rg_scores = []
        out_sent_sects_id = []
        out_sect_sentwise_rg = []
        cur_len = 0
        out_src = []
        rg_score = 0
        j = 0
        last_chunk = False
        while j < len(cls_ids):
            if j == len(cls_ids) - 1:
                out_src1 = out_src.copy()
                out_sents_labels1 = out_sents_labels.copy()
                out_section_heading_txt1 = out_section_heading_txt.copy()
                out_sents_rg_scores1 = out_sents_rg_scores.copy()
                out_sent_sects_id1 = out_sent_sects_id.copy()
                out_src.clear()
                out_sect_sentwise_rg.clear()
                out_sents_labels.clear()
                out_section_heading_txt.clear()
                out_sents_rg_scores.clear()
                out_sent_sects_id.clear()
                cur_len1 = cur_len
                cur_len = 0
                last_chunk = True
                if len(out_src1) == 0:
                    j += 1
                    continue
                yield out_src1, out_sents_labels1, out_section_heading_txt1, out_sents_rg_scores1, cur_len1, last_chunk, rg_score, section_rouge, out_sent_sects_id1

            if cur_len < chunk_size:
                out_src.append((src_sents_tokens[j][0], src_sents_tokens[j][1], src_sents_tokens[j][2]))
                out_sect_sentwise_rg.append(section_rouge[src_sents_tokens[j][1]])
                out_sents_labels.append(sent_labels[j])
                out_section_heading_txt.append(section_heading_txt[j])
                out_sents_rg_scores.append(sent_rg_scores[j])
                out_sent_sects_id.append(sent_sects_id[j])
                if j != 0:
                    cur_len += len(src_subtokens[cls_ids[j - 1]:cls_ids[j]])
                else:
                    cur_len += len(src_subtokens[:cls_ids[j]])
                j += 1

            else:
                j = j - 1
                out_src = out_src[:-1]
                out_sect_sentwise_rg = out_sect_sentwise_rg[:-1]
                out_sents_labels = out_sents_labels[:-1]
                out_section_heading_txt = out_section_heading_txt[:-1]
                out_sents_rg_scores = out_sents_rg_scores[:-1]
                out_sent_sects_id = out_sent_sects_id[:-1]
                out_src1 = out_src.copy()
                out_sents_labels1 = out_sents_labels.copy()
                out_section_heading_txt1 = out_section_heading_txt.copy()
                out_sents_rg_scores1 = out_sents_rg_scores.copy()
                out_sent_sects_id1 = out_sent_sects_id.copy()
                out_src.clear()
                out_sents_labels.clear()
                out_section_heading_txt.clear()
                out_sents_rg_scores.clear()
                cur_len1 = cur_len
                cur_len = 0
                last_chunk = False
                if len(out_src1) == 0:
                    j += 1
                    continue

                yield out_src1, out_sents_labels1, out_section_heading_txt1, out_sents_rg_scores1, cur_len1, last_chunk, rg_score, section_rouge, out_sent_sects_id1


    def preprocess_single(self, src, tgt, sent_rg_scores=None, sent_labels=None, sent_sections=None,
                          use_bert_basic_tokenizer=False, is_test=False, section_rgs=None, debug=False):


        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s[0]) for s in src]
        idxs = [i for i, s in enumerate(src) if (len(s[0]) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            try:
                _sent_labels[l] = 1
            except Exception as e:
                print(e)
                import pdb;
                pdb.set_trace()
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

        src_txt = [' '.join(sent[0]) for sent in src]
        src_sent_token_count = [self.cal_token_len([(sent[0], 'test')]) for sent in src]

        src_sents_sections = [sent[1] for sent in src]
        src_sents_number = [sent[2] for sent in src]

        try:
            sents_sect_wise_rg = [section_rgs[sect] for sect in src_sents_sections]
        except:
            import pdb;pdb.set_trace()
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

        return src_subtoken_idxs, sent_rg_scores, sent_labels, sent_sections, tgt_subtoken_idxs, segments_ids, cls_indxes, src_txt, tgt_txt, sents_sect_wise_rg, src_sents_number, src_sent_token_count

    def cal_token_len(self, src):

        idxs = [i for i, s in enumerate(src) if (len(s[0]) > self.args.min_src_ntokens_per_sent)]
        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        src = src[:self.args.max_src_nsents]
        src_txt = [' '.join(sent[0]) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]

        return len(src_subtokens)


# bert-function
def format_to_bert(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['test']


    if len(args.sent_numbers_file) > 0:
        sent_numbers = pickle.load(open(args.sent_numbers_file, "rb"))
    else:
        sent_numbers = None

    check_path_existence(args.save_path)
    for corpus_type in datasets:
        a_lst = []
        c = 0
        for json_f in glob.glob(pjoin(args.raw_path, corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            c += 1
            a_lst.append(
                (corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt')),
                 sent_numbers, 1))
        print("Number of files: " + str(c))

        ##########################
        ###### <DEBUGGING> #######
        ##########################

        # for a in a_lst:
        #     _format_to_bert(a)


        # single
        # json_f = args.raw_path + '/train.6.json'
        # _format_to_bert(('val', str(json_f), args, pjoin(args.save_path, str(json_f).replace('json', 'bert.pt')), kws, bart,
        #          sent_numbers, 25))

        ##########################
        ###### <DEBUGGING> #######
        ##########################

        pool = Pool(args.n_cpus)
        print('Processing {} set with {} json files...'.format(corpus_type, len(a_lst)))
        all_papers_count = 0
        all_paper_ids = {}
        for d in tqdm(pool.imap(_format_to_bert, a_lst), total=len(a_lst), desc=''):
            all_paper_ids[d[0]] = d[1]
            all_papers_count += d[2]

        pool.close()
        pool.join()


def _format_to_bert(params):
    corpus_type, json_file, args, save_file, sent_numbers_whole, debug_idx = params
    papers_ids = set()

    def remove_ack(source, debug=False):
        out = []
        sect_idx = 2

        for sent in source:
            if 'acknowledgment' in sent[sect_idx].lower().split() or 'acknowledgments' in sent[sect_idx].lower().split() or 'acknowledgements' in sent[sect_idx].lower().split() \
                    or 'fund' in sent[sect_idx].lower().split() or 'funding' in sent[sect_idx].lower().split() \
                    or 'appendices' in sent[sect_idx].lower().split() or 'proof of' in sent[sect_idx].lower() or \
                    'related work' in sent[sect_idx].lower() or 'previous works' in sent[sect_idx].lower() or 'references' in sent[sect_idx].lower().split() \
                    or 'figure captions' in sent[sect_idx].lower() or 'acknowledgement' in sent[sect_idx].lower().split() or 'appendix' in sent[sect_idx].lower().split():
                continue

            else:
                out.append(sent)

        return out

    is_test = corpus_type == 'test'

    model_name = args.model_name

    CHUNK_SIZE_CONST=-1
    if model_name == 'bert-based' or model_name == 'scibert' or model_name == 'bert-large':
        bert = BertData(args)
        CHUNK_SIZE_CONST = 512
    elif model_name == 'longformer':
        bert = LongformerData(args)
        CHUNK_SIZE_CONST = 2500

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []

    for j, data in enumerate(jobs[debug_idx-1:]):
        paper_id, data_src, data_tgt = data['id'], data['src'], data['tgt']

        if len(data_src) < 2:
           continue

        if not isinstance(data_src[0][0], int):
            data_src = [[idx] + s for idx, s in enumerate(data_src)]

        data_src = remove_ack(data_src)

        if sent_numbers_whole is not None:
            data_src = [d for d in data_src if d[0] in sent_numbers_whole[paper_id]]

        data_src = [s for s in data_src if len(s[1]) > args.min_src_ntokens_per_sent and len(s[1]) < args.max_src_ntokens_per_sent]
        # sent_labels = greedy_selection(source_sents, tgt, 10)
        sent_labels = [i for i, s in enumerate(data_src) if
                       s[-2] == 1 and len(s[1]) > args.min_src_ntokens_per_sent and len(
                           s[1]) < args.max_src_ntokens_per_sent]
        sent_rg_scores = [s[3] for i, s in enumerate(data_src) if len(s[1]) > args.min_src_ntokens_per_sent and len(s[1]) < args.max_src_ntokens_per_sent]
        sent_sects_id = [s[-1] for i, s in enumerate(data_src) if len(s[1]) > args.min_src_ntokens_per_sent and len(s[1]) < args.max_src_ntokens_per_sent]


        if (args.lower):
            source_sents = [([tkn.lower() for tkn in s[1]], s[2], s[0]) for s in data_src]
            data_tgt = [[tkn.lower() for tkn in s] for s in data_tgt]
        else:
            source_sents = [(s[1], s[2], s[0]) for s in data_src]

        sent_sections_textual = [s[2] for s in data_src]

        if True:
            tkn_len = bert.cal_token_len(source_sents)
            debug = False

            if tkn_len > CHUNK_SIZE_CONST:
                try:
                    for chunk_num, chunk in enumerate(
                            bert.make_chunks(source_sents, data_tgt, sent_labels=sent_labels, section_heading_txt=sent_sections_textual,
                                             sent_rg_scores=sent_rg_scores, chunk_size=CHUNK_SIZE_CONST, sent_sects_id=sent_sects_id)):
                        src_chunk, sent_labels_chunk, sent_sections_chunk, sent_rg_scores_chunk, curlen, is_last_chunk, rg_score, section_rgs, sent_sects_id_chunk = chunk
                        b_data = bert.preprocess_single(src_chunk, data_tgt, sent_labels=[0],
                                                        sent_rg_scores=sent_rg_scores_chunk,
                                                        use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                                        is_test=is_test, sent_sections=sent_sections_chunk, section_rgs=section_rgs, debug=debug)


                        if (b_data is None):
                            # import pdb;pdb.set_trace()
                            with open('not_parsed_chunk_multi_processing.txt', mode='a') as F:
                                F.write(save_file + ' idx: ' + str(j) + ' paper_id: ' + str(paper_id) + '-' +str(chunk_num)+ '\n')
                            # print(paper_id)
                            continue

                        src_subtoken_idxs, sent_rg_scores, sent_labels_chunk, sent_sections_chunk, tgt_subtoken_idxs, \
                        segments_ids, cls_ids, src_txt, tgt_txt, sents_sect_wise_rg, src_sent_number, src_sent_token_count = b_data

                        try:
                            b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                                           "src_sent_labels": sent_rg_scores.copy() if sent_rg_scores is not None else sent_sections_chunk,
                                           "sent_labels": sent_labels_chunk.copy(),
                                           "segs": segments_ids, 'clss': cls_ids,
                                           'src_txt': src_txt, "tgt_txt": tgt_txt,
                                           "paper_id": paper_id + '___' + str(chunk_num) + '___' + datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4()),
                                           "sent_sect_labels": sent_sects_id_chunk.copy(), "segment_rg_score": rg_score,
                                           "sent_sections_txt": sent_sections_chunk,  "sent_sect_wise_rg": sents_sect_wise_rg, "sent_numbers":src_sent_number,
                                           "sent_token_count":src_sent_token_count}

                            papers_ids.add(paper_id.split('___')[0])
                            datasets.append(b_data_dict)
                        except:

                            with open('not_parsed_chunk_multi.txt', mode='a') as F:
                                F.write(save_file + ' idx: ' + str(j) + ' paper_id: ' + str(paper_id) + '-' +str(chunk_num)+ '\n')
                            # print('{} with {} sentences'.format(paper_id, len(src_chunk)))

                            continue
                except Exception:
                    with open('not_parsed_chunks_function.txt', mode='a') as F:
                        F.write(
                            save_file + ' idx: ' + str(j) + ' paper_id: ' + str(paper_id) + '-' + '\n')

            else:
                try:
                    for chunk_num, chunk in enumerate(
                            bert.make_chunks(source_sents, data_tgt, sent_labels=sent_labels, section_heading_txt=sent_sections_textual,
                                             sent_rg_scores=sent_rg_scores, chunk_size=CHUNK_SIZE_CONST)):

                        src_chunk, sent_labels_chunk, sent_sections_chunk, sent_rg_scores_chunk, curlen, is_last_chunk, rg_score, section_rgs = chunk

                        b_data = bert.preprocess_single(src_chunk, data_tgt, sent_labels=[0],
                                                        sent_rg_scores=sent_rg_scores_chunk,
                                                        use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                                        is_test=is_test, sent_sections=sent_sections_chunk, section_rgs=section_rgs, debug=debug)

                        if b_data == None:
                            with open('not_parsed_processing.txt', mode='a') as F:
                                F.write(save_file + ' idx: ' + str(j) + ' paper_id: ' + str(paper_id) + '\n')
                            print(paper_id)
                            continue

                        src_subtoken_idxs, sent_rg_scores, sent_labels_chunk, sent_sections_chunk, tgt_subtoken_idxs, \
                        segments_ids, cls_ids, src_txt, tgt_txt, sents_sect_wise_rg, src_sent_number, src_sent_token_count = b_data

                        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                                       "src_sent_labels": sent_rg_scores.copy() if sent_rg_scores is not None else sent_sections_textual,
                                       "sent_labels": sent_labels.copy(),
                                       "segs": segments_ids, 'clss': cls_ids,
                                       'src_txt': src_txt, "tgt_txt": tgt_txt, "paper_id": paper_id + '___' + 'single' + '___' + datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4()),
                                       "sent_sect_labels": [0 for _ in range(len(sent_sections_textual))], "segment_rg_score": 0,
                                       "sent_sections_txt": sent_sections_textual, "sent_sect_wise_rg": sents_sect_wise_rg, "sent_numbers":src_sent_number,
                                       "sent_token_count":src_sent_token_count}
                        papers_ids.add(paper_id.split('___')[0])
                        datasets.append(b_data_dict)
                except:

                    with open('not_parsed.txt', mode='a') as F:
                        F.write(save_file + ' idx: ' + str(j) + ' paper_id: ' + str(paper_id) + '\n')

                    continue

    print('Processed instances %d data' % len(datasets))
    print('Saving to %s' % save_file)
    # written_files = glob.glob('/'.join(save_file.split('/')[:-1]) + '/' + corpus_type + '*.pt')
    torch.save(datasets, save_file)
    with open('papers_' + args.model_name + '_' +corpus_type +'.txt', mode='a') as F:
        for p in papers_ids:
            F.write(str(p))
            F.write('\n')

    gc.collect()
    return save_file, papers_ids, len(papers_ids)

# line function
def format_longsum_to_lines(args):
    if args.dataset != '':
        corpuses_type = [args.dataset]
    else:
        corpuses_type = ['train', 'val', 'test']

    sections = {}
    for corpus_type in corpuses_type:
        files = []
        for f in glob.glob(args.raw_path +'/*.json'):
            files.append(f)
            # import pdb;pdb.set_trace()
        corpora = {corpus_type: files}
        for corpus_type in corpora.keys():
            a_lst = [(f, args.keep_sect_num) for f in corpora[corpus_type]]
            pool = Pool(args.n_cpus)
            dataset = []
            p_ct = 0
            all_papers_count = 0
            curr_paper_count = 0
            ##########################
            ###### <DEBUGGING> #######
            ##########################

            # for a in tqdm(a_lst, total=len(a_lst)):
            #     d = _format_longsum_to_lines_section_based(a)
            #     if d is not None:
            #         # dataset.extend(d[0])
            #         dataset.append(d)
            #         if (len(dataset) > args.shard_size):
            #             pt_file = "{:s}{:s}.{:d}.json".format(args.save_path + '', corpus_type, p_ct)
            #             check_path_existence(args.save_path)
            #             print(pt_file)
            #             with open(pt_file, 'w') as save:
            #                 # save.write('\n'.join(dataset))
            #                 save.write(json.dumps(dataset))
            #                 print('data len: {}'.format(len(dataset)))
            #                 p_ct += 1
            #                 all_papers_count += len(dataset)
            #                 dataset = []
            # if (len(dataset) > 0):
            #
            #     pt_file = "{:s}{:s}.{:d}.json".format(args.save_path + '', corpus_type, p_ct)
            #     print(pt_file)
            #     with open(pt_file, 'w') as save:
            #         # save.write('\n'.join(dataset))
            #         save.write(json.dumps(dataset))
            #         p_ct += 1
            #         all_papers_count += len(dataset)
            #         dataset = []
            #
            # print('Processed {} papers for {} set'.format(all_papers_count, corpus_type))
            ###########################
            ###### </DEBUGGING> #######
            ###########################

            check_path_existence(args.save_path)

            # for d in tqdm(pool.imap_unordered(_format_longsum_to_lines_section_based, a_lst), total=len(a_lst)):
            for d in tqdm(pool.imap_unordered(_format_longsum_to_lines, a_lst), total=len(a_lst)):
                # d_1 = d[1]
                if d is not None:
                    all_papers_count+=1
                    curr_paper_count+=1

                    # dataset.extend(d[0])
                    dataset.append(d)
                    # import pdb;pdb.set_trace()
                    # if (len(dataset) > args.shard_size):
                    if (curr_paper_count > args.shard_size):
                        pt_file = "{:s}{:s}.{:d}.json".format(args.save_path + '', corpus_type, p_ct)
                        print(pt_file)
                        with open(pt_file, 'w') as save:
                            # save.write('\n'.join(dataset))
                            save.write(json.dumps(dataset))
                            print('data len: {}'.format(len(dataset)))
                            p_ct += 1
                            dataset = []
                        curr_paper_count = 0


            pool.close()
            pool.join()

            if (len(dataset) > 0):
                pt_file = "{:s}{:s}.{:d}.json".format(args.save_path + '', corpus_type, p_ct)
                print(pt_file)
                # all_papers_count += len(dataset)
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    p_ct += 1

                    dataset = []
            print('Processed {} papers for {} set'.format(all_papers_count, corpus_type))

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

        # if len(paper['sentences']) < 10 or sum([len(sent) for sent in paper['gold']]) < 10:
        #     return -1, 0, 0
        try:
            id = paper['filename']
        except:
            id = paper['id']

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

        # if len(paper['sentences']) < 10 or sum([len(sent) for sent in paper['gold']]) < 10:
        #     return -1, 0, 0

        try:
            id = paper['filename']
        except:
            id = paper['id']

        # for sent in paper['sentences']:
        #     tokens = sent[0]
        # if keep_sect_num:
        #     sent[1] = _get_section_text(sent[1], main_sections) if 'Abstract' not in sent[1] else sent[1]

        sections = []
        cur_sect = ''
        cur_sect_sents = []
        sections_textual = []
        for i, sent in enumerate(paper['sentences']):
            if not str(sent[0]).isdigit():
                sent = [0] + sent
            if i == 0:
                cur_sect = sent[2]
                sections_textual.append(sent[2])
                cur_sect_sents.append(sent)
                continue
            else:
                if cur_sect == sent[2]:
                    cur_sect_sents.append(sent)
                    if i == len(paper['sentences']) - 1:
                        sections.append(cur_sect_sents.copy())
                        cur_sect_sents.clear()
                        break
                else:
                    cur_sect = sent[2]
                    sections_textual.append(sent[2])
                    sections.append(cur_sect_sents.copy())
                    cur_sect_sents.clear()


        tgts = []
        ids = []
        for j, _ in enumerate(sections):
            tgts.append(paper['gold'])
            ids.append(id + "___" + str(sections_textual[j]))

        return sections, tgts, ids, sections_textual

    paper_sect_sents, paper_tgts, ids, sections_text = load_json(src_path)

    # if paper_sect_sents == -1:
    #     return None


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
