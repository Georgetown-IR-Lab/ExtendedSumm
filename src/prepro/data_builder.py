import gc
import glob
import hashlib
import json
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
from others.tokenization import BertTokenizer
from others.utils import clean, clean_upper
from prepro.utils import _get_word_ngrams

nyt_remove_words = ["photo", "graph", "chart", "map", "table", "drawing"]


def recover_from_corenlp(s):
    s = re.sub(r' \'{\w}', '\'\g<1>', s)
    s = re.sub(r'\'\' {\w}', '\'\'\g<1>', s)


def load_json(src_json, lower):
    source = []
    tgt = []
    # flag = False
    id = json.load(open(src_json))['docId']
    for sent in json.load(open(src_json))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if (lower):
            tokens = [t.lower() for t in tokens]
        source.append(tokens)

    for sent in json.load(open(src_json.replace('src', 'tgt')))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if (lower):
            tokens = [t.lower() for t in tokens]
        tgt.append(tokens)

    source = [clean_upper(' '.join(sent)).split() for sent in source]
    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean_upper(' '.join(sent)).split() for sent in tgt]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]

    return source, tgt, id


def load_xml(p):
    tree = ET.parse(p)
    root = tree.getroot()
    title, byline, abs, paras = [], [], [], []
    title_node = list(root.iter('hedline'))
    if (len(title_node) > 0):
        try:
            title = [p.text.lower().split() for p in list(title_node[0].iter('hl1'))][0]
        except:
            print(p)

    else:
        return None, None
    byline_node = list(root.iter('byline'))
    byline_node = [n for n in byline_node if n.attrib['class'] == 'normalized_byline']
    if (len(byline_node) > 0):
        byline = byline_node[0].text.lower().split()
    abs_node = list(root.iter('abstract'))
    if (len(abs_node) > 0):
        try:
            abs = [p.text.lower().split() for p in list(abs_node[0].iter('p'))][0]
        except:
            print(p)

    else:
        return None, None
    abs = ' '.join(abs).split(';')
    abs[-1] = abs[-1].replace('(m)', '')
    abs[-1] = abs[-1].replace('(s)', '')

    for ww in nyt_remove_words:
        abs[-1] = abs[-1].replace('(' + ww + ')', '')
    abs = [p.split() for p in abs]
    abs = [p for p in abs if len(p) > 2]

    for doc_node in root.iter('block'):
        att = doc_node.get('class')
        # if(att == 'abstract'):
        #     abs = [p.text for p in list(f.iter('p'))]
        if (att == 'full_text'):
            paras = [p.text.lower().split() for p in list(doc_node.iter('p'))]
            break
    if (len(paras) > 0):
        if (len(byline) > 0):
            paras = [title + ['[unused3]'] + byline + ['[unused4]']] + paras
        else:
            paras = [title + ['[unused3]']] + paras

        return paras, abs
    else:
        return None, None


def arxiv_labels(args):
    corpura = ['train', 'val', 'test']
    files = []
    json_dir = '/disk1/sajad/datasets/download_google_drive/arxiv/json/'
    for corpus_type in corpura:
        papers = {}
        files = []
        for f in glob.glob('/disk1/sajad/datasets/download_google_drive/arxiv/labels/' + corpus_type + '/*.json'):
            files.append(f)
        corpora = {corpus_type: files}
        for corpus_type in corpora.keys():
            a_lst = [(f, corpus_type, args) for f in corpora[corpus_type]]
            papers = {}
            c = 0
            pool = Pool(10)
            for data in tqdm(pool.imap_unordered(_arxiv_labels, a_lst), total=len(a_lst)):
                papers[data['paper_id']] = data['labels']

            # for a in a_lst:
            #     _sent_sect_arxiv(a)

            pool.close()
            pool.join()

            with open(json_dir + corpus_type + '-labels' + '.pkl', 'wb') as handle:
                pickle.dump(papers, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _arxiv_labels(params):
    f, set, args = params
    with open(f, 'r') as f:
        paper = json.loads(f.read())
        paper_id = paper['id']
        lables = paper['labels']
    return {'paper_id': paper_id, 'labels': lables}


def sent_sect_arxiv(args):
    corpura = ['test']
    files = []
    json_dir = '/disk1/sajad/datasets/download_google_drive/arxiv/json/'
    for corpus_type in corpura:
        papers = {}
        files = []
        for f in glob.glob('/disk1/sajad/datasets/download_google_drive/arxiv/inputs/' + corpus_type + '/*.json'):
            files.append(f)
        corpora = {corpus_type: files}
        for corpus_type in corpora.keys():
            a_lst = [(f, corpus_type, args) for f in corpora[corpus_type]]
            papers = {}
            c = 0
            pool = Pool(10)
            for data in tqdm(pool.imap_unordered(_sent_sect_arxiv, a_lst), total=len(a_lst)):
                if data is not None:
                    papers[data['paper_id']] = data['sent_sect_labels']
                else:
                    c += 1

            # for a in a_lst:
            #     _sent_sect_arxiv(a)

            pool.close()
            pool.join()

            print(f'{c} papers w/o intro in {corpus_type} set')
            with open(json_dir + corpus_type + '-sect' + '.pkl', 'wb') as handle:
                pickle.dump(papers, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _sent_sect_arxiv(params):
    f, set, args = params
    test_kws = pd.read_csv('arxiv_sects_info.csv')

    kws = {
        'intro': [kw.strip() for kw in test_kws['intro'].dropna()],
        'related': [kw.strip() for kw in test_kws['related work'].dropna()],
        # 'experiments': [kw.strip() for kw in test_kws['experiments'].dropna()],
        'results': [kw.strip() for kw in test_kws['results'].dropna()],
        'conclusion': [kw.strip() for kw in test_kws['conclusion'].dropna()]
    }

    with open(f, 'r') as f:
        paper = json.loads(f.read())
        paper_id = paper['id']
        paper_sect_labels = []
        for i, sect in enumerate(paper['section_names']):
            sect_main_title = sect.lower().strip()
            sentence_num = paper['section_lengths'][i]
            if len(sect_main_title.strip()) > 0:
                if 'introduction' in sect_main_title.split()[0] or sect_main_title in kws['intro']:
                    paper_sect_labels.extend([0] * sentence_num)

                elif sect_main_title in kws['related']:
                    paper_sect_labels.extend([1] * sentence_num)

                # elif sect_main_title in kws['experiments']:
                #     paper_sect_labels.extend([3] * sentence_num)

                elif 'result' in sect_main_title \
                        or 'discussion' in sect_main_title \
                        or sect_main_title in kws['results']:
                    paper_sect_labels.extend([3] * sentence_num)

                elif 'conclusion' in sect_main_title or 'summary' in sect_main_title or sect_main_title in kws[
                    'conclusion']:
                    paper_sect_labels.extend([4] * sentence_num)

                else:
                    paper_sect_labels.extend([2] * sentence_num)

        if 5 in paper_sect_labels:
            for i, e in reversed(list(enumerate(paper_sect_labels))):
                if e == 4:
                    paper_sect_labels = paper_sect_labels[:i + 1]
                    break


        elif 5 not in paper_sect_labels:
            for i, e in reversed(list(enumerate(paper_sect_labels))):
                if e == 4:
                    paper_sect_labels = paper_sect_labels[:i + 1]
                    break

        return {'paper_id': paper_id, 'sent_sect_labels': paper_sect_labels}


def sent_sect_mine(args):
    json_dirs = []
    # for c_type in ['train', 'val', 'test']:
    for c_type in ['test']:
        json_dirs.append(args.raw_path + c_type + '.json')
        # _sent_sect(args.raw_path + c_type + '.json')

    for j in json_dirs:
        _sent_sect_mine(j)


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


def _sent_sect_mine(json_dir):
    def arraySortedOrNot(arr):
        return sorted(arr) == arr

    global sentence_num
    test_kws = pd.read_csv('train_papers_sects_longsum.csv')
    outf = open('lst.txt', mode='a')
    kws = {
        'intro': [kw.lower().strip() for kw in test_kws['intro'].dropna()],
        'related': [kw.lower().strip() for kw in test_kws['related work'].dropna()],
        'experiments': [kw.lower().strip() for kw in test_kws['experiments'].dropna()],
        'results': [kw.lower().strip() for kw in test_kws['results'].dropna()],
        'conclusion': [kw.lower().strip() for kw in test_kws['conclusion'].dropna()]
    }

    papers = {}
    print(f'Reading {json_dir}')
    line_num = sum(1 for line in open(json_dir, 'r'))
    with open(json_dir) as f:
        for line in tqdm(f, total=line_num):
            try:
                total_sent = 0
                paper = json.loads(line)
                if 'Unsupervised Learning of Contextual Role Knowledge for Coreference Resolution'.lower() in paper[
                    "title"].lower().strip():
                    continue
                paper_id = hashhex(paper["title"].lower().strip())

                papers[paper_id] = []
                for sect in paper["body"]:

                    if 'section_body' in sect.keys():
                        body_key = 'section_body'
                        title_key = 'section_title'
                    else:
                        body_key = 'section body'
                        title_key = 'section title'

                    if sect[body_key] != 'null':
                        sect_body = sect[body_key]
                    else:
                        sect_body = []
                    try:
                        if 'sub' in sect.keys() and len(sect['sub']) > 0:
                            for i, s in enumerate(sect['sub']):
                                if s[body_key] != 'null':
                                    sect_body.extend(s[body_key])
                                else:
                                    continue
                        sentence_num = len(sect_body)
                        total_sent += sentence_num
                    except:
                        print('here train')
                        import pdb;
                        pdb.set_trace()
                    try:
                        sect_main_title = sect[title_key].lower().strip()
                        if 'introduction' in sect_main_title.split()[0] or sect_main_title in kws['intro']:
                            papers[paper_id].extend([0] * sentence_num)

                        elif sect_main_title in kws['related']:
                            papers[paper_id].extend([1] * sentence_num)

                        elif sect_main_title in kws['experiments']:
                            papers[paper_id].extend([3] * sentence_num)

                        elif sect_main_title in kws['results'] or 'results' in sect_main_title:
                            papers[paper_id].extend([4] * sentence_num)

                        elif sect_main_title in kws[
                            'conclusion'] or 'conclusions' in sect_main_title or 'conclusion' in sect_main_title:
                            papers[paper_id].extend([5] * sentence_num)

                        # elif 'result' in sect_main_title \
                        #         or 'discussion' in sect_main_title \
                        #         or sect_main_title in kws['results']:
                        #     papers[paper_id].extend([4] * sentence_num)
                        #
                        # elif 'conclusion' in sect_main_title or 'summary' in sect_main_title or sect_main_title in kws[
                        #     'conclusion']:
                        #     papers[paper_id].extend([5] * sentence_num)

                        else:
                            # Methodology
                            papers[paper_id].extend([2] * sentence_num)
                    except:
                        import pdb;
                        pdb.set_trace()



            except:
                try:
                    del papers[paper_id]
                except:
                    import pdb;
                    pdb.set_trace()
                continue

            if 5 in papers[paper_id]:
                for i, e in reversed(list(enumerate(papers[paper_id]))):
                    if e == 4:
                        papers[paper_id] = papers[paper_id][:i + 1]
                        break


            elif 5 not in papers[paper_id]:
                for i, e in reversed(list(enumerate(papers[paper_id]))):
                    if e == 4:
                        papers[paper_id] = papers[paper_id][:i + 1]
                        break

                try:
                    writee = paper["body"][-1]["section_title"]
                except:
                    writee = paper["body"][-1]["section title"]

                outf.write(writee)
                outf.write('\n')

            # if not arraySortedOrNot(papers[paper_id]):
            #     seen0=False
            #     seen3 = False
            #     for i, e in enumerate(papers[paper_id]):
            #         if e==0:
            #             seen0=True
            #             seen3= False
            #
            #         if e==3:
            #             seen3= True
            #
            #         if seen0 and not seen3 and e==2:
            #             papers[paper_id][i] = 0
            #
            #         if seen3 and e == 2:
            #             papers[paper_id][i] = 3
            #
            #         if e==1:
            #             break

        # assert len(papers[paper_id]) == total_sent, "Mismtch between sentence num"
    print(len(papers))
    with open(json_dir.replace('.json', '') + '-sect-5' + '.pkl', 'wb') as handle:
        pickle.dump(papers, handle, protocol=pickle.HIGHEST_PROTOCOL)


def tokenize(args):
    src_dir = os.path.abspath(args.raw_path + '/src/')
    tokenized_src_dir = os.path.abspath(args.save_path + '/tokenized/src/')

    tgt_dir = os.path.abspath(args.raw_path + '/tgt/')
    tokenized_tgt_dir = os.path.abspath(args.save_path + '/tokenized/tgt/')

    print("Preparing to tokenize %s to %s..." % (src_dir, tokenized_src_dir))
    srcs = os.listdir(src_dir)
    tgts = os.listdir(tgt_dir)
    # make IO list file

    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in srcs:
            # if (not s.endswith('story')):
            #     continue
            f.write("%s\n" % (os.path.join(src_dir, s)))
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tokenized_src_dir]
    print("Tokenizing %i files in %s and saving in %s..." % (len(srcs), src_dir, tokenized_src_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for t in tgts:
            # if (not s.endswith('story')):
            #     continue
            f.write("%s\n" % (os.path.join(tgt_dir, t)))
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tokenized_tgt_dir]
    print("Tokenizing %i files in %s and saving in %s..." % (len(tgts), tgt_dir, tokenized_tgt_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized srcs directory contains the same number of files as the original directory
    num_orig = len(os.listdir(src_dir))
    num_tokenized = len(os.listdir(tokenized_src_dir))
    # import pdb;pdb.set_trace()
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized srcs directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_src_dir, num_tokenized, src_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (src_dir, tokenized_src_dir))


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def identify_sent_sects(paper_sent_sect, segment_sent_ids, lst_segment=False):
    """
    :param sent:
    :type sent:
    :return:
        6 labels:
            "intro": 0
            "related": 1
            "methodology": 2
            "experimental": 3
            "results" : 4
            "conclusion": 5
    :rtype:
    """

    sects = []
    for id in segment_sent_ids:
        try:
            sects.append(paper_sent_sect[id])
        except:
            try:
                lst = paper_sent_sect[-1]
                sects.append(lst)
            except:
                pass
                # import pdb;
                # pdb.set_trace()
    # import pdb;pdb.set_trace()
    assert len(segment_sent_ids) == len(
        sects), "Number of sents in segment should be the same with sect labels assigned to"
    return sects


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


def segment_rg_scores(doc_sent_list, abstract_sent_list, oracle=False):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for i in range(len(sents)):
        candidates_1 = [evaluated_1grams[i]]
        candidates_1 = set.union(*map(set, candidates_1))
        candidates_2 = [evaluated_2grams[i]]
        candidates_2 = set.union(*map(set, candidates_2))
        rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
        rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
        rouge_score = (rouge_1 + rouge_2) / 2
        selected.append(float("%.4f" % rouge_score))

    if oracle:
        sent_rg = list(zip(doc_sent_list, selected))
        sent_rg.sort(key=lambda x: x[1], reverse=True)
        # import pdb;pdb.set_trace()
        return [s[0] for s in sent_rg[:4]]

    return selected


def tokenize_with_corenlp(input_text, id, source_folder='/home/sajad/input/', out_folder='/home/sajad/tokenized/',
                          options='tokenize,ssplit', section_title=''):
    in_file = open(source_folder + str(id) + '-' + section_title + '.txt', mode='w')
    in_file.write(input_text.strip())
    in_file.close()

    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', options,
               '-ssplit.newlineIsSentenceBreak', 'always', '-file',
               source_folder + str(id) + '-' + section_title + '.txt', '-outputFormat',
               'json', '-outputDirectory', out_folder]
    subprocess.call(command)
    with open(out_folder + str(id) + '-' + section_title + '.txt.json') as f:
        tokenized = json.loads(f.read())

    out_tokenized = []

    for sent in tokenized['sentences']:
        sent_tokens = []
        for i, tkn in enumerate(sent['tokens']):
            sent_tokens.append(tkn['word'])
        out_tokenized.append(sent_tokens.copy())

    # os.remove('input.txt')
    # os.remove('tokenized/input.txt.json')

    return out_tokenized


class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased',
                                                       do_lower_case=True)

        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
        #                                                do_lower_case=True)
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt, sent_rg_scores, sent_labels=None, use_bert_basic_tokenizer=False, is_test=False):
        if ((not is_test) and len(src) == 0):
            return None
        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src)]

        # _sent_labels = [0] * len(src)
        # for l in sent_labels:
        #     _sent_labels[l] = 1

        _sent_rg_scores = sent_rg_scores
        _sent_labels = sent_labels
        # _sent_rg_scores = sent_rg_scores

        src = [src[i] for i in idxs]
        sent_rg_scores = [_sent_rg_scores[i] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.args.max_src_nsents]
        # sent_rg_scores = sent_rg_scores
        # sent_labels = sent_labels

        # if ((not is_test)):
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
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_rg_scores = sent_rg_scores[:len(cls_ids)]
        sent_labels = sent_labels[:len(cls_ids)]
        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt
             in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)
        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, sent_rg_scores, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt


def format_to_bert(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['test']

    test_kws = pd.read_csv('train_papers_sects.csv')

    kws = {
        'intro': [kw.strip() for kw in test_kws['intro'].dropna()],
        'related': [kw.strip() for kw in test_kws['related work'].dropna()],
        'exp': [kw.strip() for kw in test_kws['experiments'].dropna()],
        'res': [kw.strip() for kw in test_kws['results'].dropna()],
        'conclusion': [kw.strip() for kw in test_kws['conclusion'].dropna()]
    }

    for corpus_type in datasets:
        a_lst = []

        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append(
                (corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt')), kws))
        # for a in a_lst:
        #     if not path.exists(pjoin(args.save_path, a[-2])):
        #         print(pjoin(args.save_path, a[-2]))
        #         _format_to_bert_cspubsum(a)
        pool = Pool(6)
        # for d in pool.imap(_format_to_bert, a_lst):
        #     pass

        for a in a_lst:
            _format_to_bert(a)

        pool.close()
        pool.join()


def format_to_bert_csAbs(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['Test']

    for corpus_type in datasets:
        a_lst = []

        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append(
                (corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        for a in a_lst:
            # if not path.exists(pjoin(args.save_path, a[-2])):
            # print(pjoin(args.save_path, a[-2]))
            _format_to_bert_csabs(a)
        # pool = Pool(6)
        # for d in pool.imap(_format_to_bert_csabs, a_lst):
        #     pass

        # pool.close()
        # pool.join()


#

def format_to_bert_arxiv_custom(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['test']

    for corpus_type in datasets:
        a_lst = []

        for json_f in glob.glob(pjoin(args.raw_path, 'json', '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append(
                (corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        for a in a_lst:
            _format_to_bert_cspubsum_only_src(a)
        # pool = Pool(15)
        # for d in pool.imap(_format_to_bert, a_lst):
        #     pass
        #
        # pool.close()
        # pool.join()


def format_to_bert_arxiv(args):
    test_kws = pd.read_csv('arxiv_sections_6l.csv')

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

    for corpus_type in datasets:
        a_lst = []
        c = 0
        for json_f in glob.glob(pjoin('/disk1/sajad/datasets/sci/arxiv/json/', corpus_type + '.*.json')):
            # for json_f in tfiles:
            real_name = json_f.split('/')[-1]
            if not os.path.exists(pjoin(args.save_path, real_name.replace('json', 'bert.pt'))):
                c += 1
            a_lst.append(
                (corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt')), kws))
        print("Number of files: " + str(c))

        # for a in a_lst:
        #     _format_to_bert_arxiv(a)

        pool = Pool(8)
        for d in pool.imap(_format_to_bert_cspubsum_only_src, a_lst):
            pass

        # pool.close()
        # pool.join()


def _format_to_bert_arxiv(param):
    corpus_type, json_file, args, save_file, kws = param

    def _get_section_id(sect):
        try:
            sect = sect.lower()
        except:
            return 2

        if sect in kws['intro']:
            return 0
        elif sect in kws['related']:
            return 1

        elif sect in kws['exp']:
            return 3

        elif sect in kws['conclusion']:
            return 4

        else:
            return 2

    emp = 0
    is_test = corpus_type == 'test'
    # # if (os.path.exists(save_file)):
    # if (os.path.exists(save_file)):
    #     logger.info('Ignore %s' % save_file)
    #     return

    bert = BertData(args)
    logger.info('Processing %s' % json_file)
    print(f"Reading {json_file}")
    jobs = json.load(open(json_file))
    datasets = []
    # if '88' in save_file:
    #     jobs = jobs[:1874] + jobs[1875:]
    #     jobs = jobs[:1876] + jobs[1877:]

    # jobs = jobs[:1013] + jobs[1014:]
    # jobs = jobs[:1013] + jobs[1014:]
    for j, data in tqdm(enumerate(jobs), total=len(jobs)):
        source, tgt, id = data['src'], data['tgt'], data['id']
        paper_id = hashhex(id)

        if len(source) > 1 and len(tgt) > 0:
            segment = []
            segment_labels = []
            # segment_labels=  []
            segment_sent_num = []
            segment_section = []
            token_ctr = 0
            i = 0
            # try:
            while i < len(source):
                sent = source[i]
                if len(sent[0]) + token_ctr < 1024:

                    segment.append(sent[1])
                    segment_section.append(_get_section_id(sent[0]))
                    segment_labels.append(0)

                    segment_sent_num.append(i)
                    token_ctr += len(sent[0])
                    # print(i)
                    if i == len(source) - 1:
                        token_ctr = 0
                        # sent_labels = greedy_selection(segment, tgt, 3)
                        sent_labels = segment_rg_scores(segment, tgt)
                        # segment_labels.append(sent[2])
                        # segment_section.append(sent[1])
                        if (args.lower):
                            segment = [' '.join(s).lower().split() for s in segment]
                            tgt = [' '.join(s).lower().split() for s in tgt]

                        try:
                            b_data = bert.preprocess(segment.copy(), tgt, sent_labels.copy(),
                                                     use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                                     is_test=is_test)
                        except:
                            import pdb;
                            pdb.set_trace()
                        if (b_data is None):
                            continue
                        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
                        try:
                            assert len(segment_section) == len(
                                sent_labels), "Number of segment_sent and section_sents should be the same"
                            assert len(cls_ids) == len(
                                sent_labels), "Number of segment_sent and section_sents should be the same"
                        except:
                            import pdb;
                            pdb.set_trace()

                        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                                       "src_sent_labels": sent_labels.copy(), "segs": segments_ids, 'clss': cls_ids,
                                       'src_txt': src_txt, "tgt_txt": tgt_txt, "paper_id": paper_id,
                                       "sent_sect_labels": segment_section.copy()}

                        datasets.append(b_data_dict.copy())
                        segment_sent_num.clear()
                        segment.clear()
                        segment_labels.clear()
                        segment_section.clear()


                else:
                    if len(sent[0]) >= 1024 and len(segment) == 0:
                        source = source[:i] + source[i + 1:]
                        # i = i + 1
                        continue
                    # import pdb;pdb.set_trace()
                    i = i - 1
                    token_ctr = 0
                    # sent_labels = greedy_selection(segment, tgt, 3)
                    sent_labels = segment_rg_scores(segment, tgt)
                    # segment_labels.append(sent[2])
                    # segment_section.append(_get_section_id(sent[1]))
                    if (args.lower):
                        segment = [' '.join(s).lower().split() for s in segment]
                        tgt = [' '.join(s).lower().split() for s in tgt]

                    # if len(segment_section) != len(segment_labels):
                    #     import pdb;
                    #     pdb.set_trace()
                    #
                    # if len(segment) != len(segment_labels):
                    #     import pdb;
                    #     pdb.set_trace()
                    #
                    # if len(segment) != len(segment_section):
                    #     import pdb;
                    #     pdb.set_trace()

                    b_data = bert.preprocess(segment.copy(), tgt, sent_labels.copy(),
                                             use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                             is_test=is_test)

                    # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)
                    if (b_data is None):
                        continue
                    src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data

                    if len(cls_ids) != len(sent_labels):
                        import pdb;
                        pdb.set_trace()

                    assert len(segment_section) == len(
                        segment_labels), "Number of segment_section and segment_labels should be the same"
                    assert len(cls_ids) == len(
                        sent_labels), "Number of cls_ids and sent_labels should be the same"
                    assert len(cls_ids) == len(
                        segment_section), "Number of cls_ids and segment_section should be the same"

                    b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                                   "src_sent_labels": sent_labels.copy(), "segs": segments_ids, 'clss': cls_ids,
                                   'src_txt': src_txt, "tgt_txt": tgt_txt, "paper_id": paper_id,
                                   "sent_sect_labels": segment_section.copy()}
                    datasets.append(b_data_dict.copy())
                    segment_sent_num.clear()
                    segment.clear()
                    segment_section.clear()
                    segment_labels.clear()
                i += 1
            # except:
            #     emp += 1
            #     continue
        else:

            emp += 1
            continue
    if emp > 0: print(f'Empty: {emp}')
    if emp > 0: print(f'datasets: {len(datasets)}')
    if emp > 0: print(f'Saving: {save_file}')
    # logger.info('Processed instances %d' % len(datasets))
    # logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_to_bert_cspubsum(args):
    test_kws = pd.read_csv('train_papers_sects.csv')

    kws = {
        'intro': [kw.strip() for kw in test_kws['intro'].dropna()],
        'related': [kw.strip() for kw in test_kws['related work'].dropna()],
        # 'method': [kw.strip() for kw in test_kws['method'].dropna()],
        'exp': [kw.strip() for kw in test_kws['experiments'].dropna()],
        'res': [kw.strip() for kw in test_kws['results'].dropna()],
        'conclusion': [kw.strip() for kw in test_kws['conclusion'].dropna()]
    }

    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['test']

    data = []
    with open('/home/sajad/packages/sum/scientific-paper-summarisation/Data/Test_Data/all_data_' + str(
            datasets[0]) + '.json') as f:
        for li in f:
            data.append(json.loads(li.strip()))

    pt = []
    ptctr = 0
    part = 0
    for i, d in enumerate(data):
        pt.append(d)
        ptctr += 1
        if ptctr > 2000:
            _format_to_bert_cspubsum_only_src((datasets[0], pt, args,
                                               '/disk1/sajad/datasets/sci/csp/bert-files/5l-rg-labels-whole-3/' +
                                               datasets[0] + '.' + str(part) + '.pt', kws))
            part += 1
            ptctr = 0
            pt.clear()
    if len(pt) > 0:
        _format_to_bert_cspubsum_only_src((datasets[0], pt, args,
                                           '/disk1/sajad/datasets/sci/csp/bert-files/5l-rg-labels-whole-3/' + datasets[
                                               0] + '.' + str(part) + '.pt',
                                           kws))


def format_to_bert_arxiv_2(args):
    test_kws = pd.read_csv('training_sections_categories_arxiv_2.csv')

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

    data = []
    with open('/disk1/sajad/datasets/lsum/files/' + str(datasets[0]) + '.json') as f:
        for li in f:
            data.append(json.loads(li.strip()))

    pt = []
    ptctr = 0
    part = 0
    for i, d in enumerate(data):
        pt.append(d)
        ptctr += 1
        if ptctr > 500:
            _format_to_bert_cspubsum_only_src((datasets[0], pt, args,
                                               '/disk1/sajad/datasets/sci/arxiv/bert-files/6labels/' + datasets[
                                                   0] + '.' + str(part) + '.pt', kws))
            part += 1
            ptctr = 0
            pt.clear()
    if len(pt) > 0:
        _format_to_bert_cspubsum_only_src((datasets[0], pt, args,
                                           '/disk1/sajad/datasets/sci/arxiv/bert-files/6labels/' + datasets[
                                               0] + '.' + str(part) + '.pt',
                                           kws))


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

    data = []
    with open(args.raw_path + '/' + str(datasets[0]) + '.0.json') as f:
        for li in f:
            data.append(json.loads(li.strip()))

    # selected_ids = pickle.load(open(args.id_files_src + '/selected_ids_' + str(datasets[0]) + '.pkl', "rb"))
    pt = []
    ptctr = 0
    part = 0

    bart = args.bart

    for i, d in enumerate(data):
        pt.append(d)
        ptctr += 1
        if ptctr > 500:
            _fomat_to_bart_section_based(
                (datasets[0], pt, args, args.save_path + '/' + datasets[0] + '.' + str(part) + '.pt', kws, bart))
            part += 1
            ptctr = 0
            pt.clear()
    if len(pt) > 0:
        _fomat_to_bart_section_based((datasets[0], pt, args,
                                      args.save_path + '/' + datasets[0] + '.' + str(part) + '.pt', kws, bart))


def _fomat_to_bart_section_based(param):
    corpus_type, jobs, args, save_file, kws, bart = param

    def _get_section_id(sect):
        try:
            sect = sect.lower()
            sect = sect.replace('.', '').strip()
        except:
            return 2

        if sect in kws['intro']:
            return 0
        elif sect in kws['related']:
            return 1

        # elif sect in kws['method']:
        #     return 2

        elif sect in kws['exp']:
            return 3

        elif sect in kws['res']:
            return 4

        elif sect in kws['conclusion']:
            return 5
        else:
            return 2

    def remove_ack(source):
        out = []
        for sent in source:
            if 'acknowledgment' in sent[1].lower() or 'acknowledgments' in sent[1].lower() \
                    or 'fund' in sent[1].lower() or 'funding' in sent[1].lower() \
                    or 'appendices' in sent[1].lower():
                continue
            else:
                out.append(sent)
        return out

    emp = 0
    is_test = corpus_type == 'test'
    bert = BertData(args)

    datasets = []

    for j, data in tqdm(enumerate(jobs[0]), total=len(jobs)):
        wsource, tgt, id = data['src'], data['tgt'], data['id']
        paper_id = str(id)
        # selected_ids[paper_id] = selected_ids[paper_id]
        if len(wsource) > 1 and len(tgt) > 0:
            # segment = []
            wsource = remove_ack(wsource)
            i = 0
            # try:
            while i < len(wsource):

                # for j in wsource:
                #     if wsource[][1]
                cur_sect = ''
                source = []
                z = i
                for z in range(*{'start': i, 'stop': len(wsource), 'step': 1}.values()):
                    if z == i:
                        cur_sect = wsource[z][1]
                        source.append(wsource[z])
                    else:
                        if wsource[z][1] == cur_sect:
                            source.append(wsource[z])
                        else:
                            break

                # print(("Encdoing section [%s] for paper [%s]") % (cur_sect, paper_id))
                i = z + 1

                k = 0
                segment_labels = []
                sent_rg_scores = []
                segment_sent_num = []
                segment_section = []
                segment = []
                token_ctr = 0

                while k < len(source):
                    sent = source[k]
                    # if i in selected_ids[paper_id]:

                    if len(sent[0]) + token_ctr < 1698:
                        segment.append(sent[0])
                        segment_section.append(_get_section_id(sent[1]))

                        # try:
                        #     int(sent[2])
                        # except:
                        #     import pdb;pdb.set_trace()

                        sent_rg_scores.append(sent[2])
                        segment_labels.append(sent[4])
                        segment_sent_num.append(k)
                        token_ctr += len(sent[0])

                        # if paper_id=='63531440' and sent[3].startswith('At test'):
                        #     import pdb;pdb.set_trace()

                        if k == len(source) - 1 and len(segment) > 3:
                            token_ctr = 0
                            if (args.lower):
                                # segment_str = ' '.join([s.lower() for s in segment])
                                segment = [[ss.lower() for ss in s] for s in segment]
                                # tgt_str = ' '.join([' '.join(s).lower() for s in tgt])
                                tgt = [[ss.lower() for ss in s] for s in tgt]
                            # import pdb;pdb.set_trace()
                            try:
                                b_data = bert.preprocess(segment.copy(), tgt, sent_rg_scores.copy(),
                                                         segment_labels.copy(),
                                                         use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                                         is_test=is_test)
                                # import pdb;pdb.set_trace()
                            except:
                                import pdb;
                                pdb.set_trace()
                            if (b_data is None):
                                continue

                            src_subtoken_idxs, sent_rg_scores, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
                            try:
                                assert len(segment_labels) == len(
                                    sent_rg_scores), "Number of segment_sent and section_sents should be the same"
                                assert len(cls_ids) == len(
                                    sent_rg_scores), "Number of segment_sent and section_sents should be the same"
                            except:
                                import pdb;
                                pdb.set_trace()
                            b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                                           "src_sent_labels": sent_rg_scores.copy(),
                                           "sent_labels": segment_labels.copy(),
                                           "segs": segments_ids, 'clss': cls_ids,
                                           'src_txt': src_txt, "tgt_txt": tgt_txt, "paper_id": paper_id,
                                           "sent_sect_labels": segment_section.copy()}
                            datasets.append(b_data_dict.copy())
                            segment_sent_num.clear()
                            segment.clear()
                            segment_labels.clear()
                            sent_labels.clear()
                            segment_section.clear()

                        elif k == len(source) - 1 and len(source) <= 3:
                            # sentence count is less than/eq to 3, should be concatenated.
                            if (args.lower):
                                # segment_str = ' '.join([s.lower() for s in segment])
                                segment = [[ss.lower() for ss in s] for s in segment]
                                # tgt_str = ' '.join([' '.join(s).lower() for s in tgt])
                                tgt = [[ss.lower() for ss in s] for s in tgt]

                            if len(datasets) == 0:
                                break
                            last_dataset = datasets[-1]

                            segment_labels1 = last_dataset['sent_labels'] + segment_labels
                            sent_rg_scores1 = last_dataset['src_sent_labels'] + sent_rg_scores
                            # segment_sent_num = last_dataset['sent_labels'] + segment_labels
                            segment_section1 = last_dataset['sent_labels'] + segment_section
                            segment1 = [s.split() for s in last_dataset['src_txt']] + segment

                            try:
                                b_data = bert.preprocess(segment1.copy(), tgt, sent_rg_scores1.copy(),
                                                         segment_labels1.copy(),
                                                         use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                                         is_test=is_test)
                            except:
                                import pdb;
                                pdb.set_trace()
                            if (b_data is None):
                                continue

                            src_subtoken_idxs, sent_rg_scores, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
                            # try:
                            assert len(segment_labels1) == len(sent_rg_scores1), "Number of segment_sent and section_sents should be the same"
                            assert len(cls_ids) == len(sent_rg_scores), "Number of segment_sent and section_sents should be the same"
                            # except:
                            #     print('a')
                            #     import pdb;
                            #     pdb.set_trace()

                            b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                                           "src_sent_labels": sent_rg_scores1.copy(),
                                           "sent_labels": segment_labels1.copy(),
                                           "segs": segments_ids, 'clss': cls_ids,
                                           'src_txt': src_txt, "tgt_txt": tgt_txt, "paper_id": paper_id,
                                           "sent_sect_labels": segment_section1.copy()}

                            datasets[-1] = b_data_dict
                            segment_sent_num.clear()
                            segment1.clear()
                            segment_labels1.clear()
                            segment_section1.clear()

                    else:
                        if len(sent[0]) >= 1698 and len(segment) == 0:
                            source = source[:k] + source[k + 1:]
                            continue

                        k = k - 1
                        token_ctr = 0

                        if (args.lower):
                            # segment_str = ' '.join([s.lower() for s in segment])
                            segment = [[ss.lower() for ss in s] for s in segment]
                            # tgt_str = ' '.join([' '.join(s).lower() for s in tgt])
                            tgt = [[ss.lower() for ss in s] for s in tgt]

                        if len(segment_section) != len(segment_labels):
                            import pdb;
                            pdb.set_trace()

                        if len(segment) != len(segment_labels):
                            import pdb;
                            pdb.set_trace()

                        if len(segment) != len(segment_section):
                            import pdb;
                            pdb.set_trace()

                        # import pdb;pdb.set_trace()
                        b_data = bert.preprocess(segment.copy(), tgt, sent_rg_scores.copy(), segment_labels.copy(),
                                                 use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                                 is_test=is_test)

                        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)
                        if (b_data is None):
                            continue
                        src_subtoken_idxs, sent_rg_scores, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data

                        if len(cls_ids) != len(sent_rg_scores):
                            import pdb;
                            pdb.set_trace()
                            continue

                        assert len(segment_section) == len(
                            segment_labels), "Number of segment_section and segment_labels should be the same"
                        assert len(cls_ids) == len(
                            sent_rg_scores), "Number of cls_ids and sent_labels should be the same"

                        if len(cls_ids) != len(segment_section):
                            import pdb;
                            pdb.set_trace()
                            continue

                        assert len(cls_ids) == len(
                            segment_section), "Number of cls_ids and segment_section should be the same"

                        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                                       "src_sent_labels": sent_rg_scores.copy(), "sent_labels": segment_labels.copy(),
                                       "segs": segments_ids, 'clss': cls_ids,
                                       'src_txt': src_txt, "tgt_txt": tgt_txt, "paper_id": paper_id,
                                       "sent_sect_labels": segment_section.copy()}

                        datasets.append(b_data_dict.copy())
                        segment_sent_num.clear()
                        segment.clear()
                        sent_labels.clear()
                        segment_section.clear()
                        segment_labels.clear()

                    k += 1
        else:
            emp += 1
            continue

    # for i, (src, tgt) in enumerate(datasets):
    #     with open('/home/sajad/packages/sum/transformers/examples/seq2seq/lsum-1100/' + corpus_type + '.source', mode='a') as sf:
    #         sf.write(src)
    #         sf.write('\n')
    #     with open('/home/sajad/packages/sum/transformers/examples/seq2seq/lsum-1100/' + corpus_type + '.target', mode='a') as tf:
    #         tf.write(tgt)
    #         tf.write('\n')

    if emp > 0: print(f'Empty: {emp}')
    # if emp > 0: print(f'datasets: {len(datasets)}')
    print(f'datasets: {len(datasets)}')
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def _fomat_to_bart_oracle(param):
    corpus_type, jobs, args, save_file, kws, bart = param

    def _get_section_id(sect):
        try:
            sect = sect.lower()
            sect = sect.replace('.', '').strip()
        except:
            return 2

        if sect in kws['intro']:
            return 0
        elif sect in kws['related']:
            return 1

        # elif sect in kws['method']:
        #     return 2

        elif sect in kws['exp']:
            return 3

        elif sect in kws['res']:
            return 4

        elif sect in kws['conclusion']:
            return 5
        else:
            return 2

    def remove_ack(source):
        out = []
        for sent in source:
            if 'acknowledgment' in sent[1].lower() or 'acknowledgments' in sent[1].lower() \
                    or 'fund' in sent[1].lower() or 'funding' in sent[1].lower() \
                    or 'appendices' in sent[1].lower():
                continue
            else:
                out.append(sent)
        return out

    emp = 0
    is_test = corpus_type == 'test'
    bert = BertData(args)

    datasets = []
    datasets_bart = []

    for j, data in tqdm(enumerate(jobs[0]), total=len(jobs)):
        source, tgt, id = data['src'], data['tgt'], data['id']
        paper_id = str(id)
        # selected_ids[paper_id] = selected_ids[paper_id]
        if len(source) > 1 and len(tgt) > 0:
            segment = []
            source = remove_ack(source)
            segment_labels = []
            sent_rg_scores = []
            segment_sent_num = []
            segment_section = []
            token_ctr = 0

            src_oracle_str = ''
            tgt_oracle_str = ''

            i = 0
            # try:
            while i < len(source):
                sent = source[i]

                if sent[-1] == 1:
                    src_oracle_str += sent[-2]
                    src_oracle_str += ' '
                    segment.append(sent[0])
                    segment_labels.append(sent[-1])
                    sent_rg_scores.append(sent[-3])
                    segment_section.append(_get_section_id(sent[1]))
                    segment_sent_num.append(i)
                    token_ctr += len(sent[0])

                if i == len(source) - 1:
                    token_ctr = 0
                    if (args.lower):
                        segment = [[ss.lower() for ss in s] for s in segment]
                        src_oracle_str = src_oracle_str.lower()
                        tgt = [[ss.lower() for ss in s] for s in tgt]
                        tgt_oracle_str = ' '.join([' '.join(s).lower() for s in tgt])

                    try:
                        b_data = bert.preprocess(segment.copy(), tgt, sent_rg_scores.copy(), segment_labels.copy(),
                                                 use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                                 is_test=is_test)
                    except:
                        import pdb;
                        pdb.set_trace()
                    if (b_data is None):
                        continue

                    src_subtoken_idxs, sent_rg_scores, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
                    try:
                        assert len(segment_labels) == len(
                            sent_rg_scores), "Number of segment_sent and section_sents should be the same"
                        assert len(cls_ids) == len(
                            sent_rg_scores), "Number of segment_sent and section_sents should be the same"
                    except:
                        import pdb;
                        pdb.set_trace()
                    b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                                   "src_sent_labels": sent_rg_scores.copy(), "sent_labels": sent_labels.copy(),
                                   "segs": segments_ids, 'clss': cls_ids,
                                   'src_txt': src_txt, "tgt_txt": tgt_txt, "paper_id": paper_id,
                                   "sent_sect_labels": segment_section.copy()}
                    datasets.append(b_data_dict.copy())

                    if bart:
                        datasets_bart.append((src_oracle_str, tgt_oracle_str))

                    segment_sent_num.clear()
                    segment.clear()
                    segment_labels.clear()
                    sent_labels.clear()
                    segment_section.clear()
                i += 1
        else:
            emp += 1
            continue

    if emp > 0: print(f'Empty: {emp}')
    # if emp > 0: print(f'datasets: {len(datasets)}')
    print(f'datasets: {len(datasets)}')

    if bart:
        srcFile = open('/home/sajad/packages/sum/transformers/examples/seq2seq/lsum-oracle/' + corpus_type + '.source',
                       mode='a')
        tgtFile = open('/home/sajad/packages/sum/transformers/examples/seq2seq/lsum-oracle/' + corpus_type + '.target',
                       mode='a')
        for (src, tgt) in datasets_bart:
            srcFile.write(src.strip())
            srcFile.write('\n')
            tgtFile.write(tgt.strip())
            tgtFile.write('\n')

    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def _format_to_bert_cspubsum(param):
    corpus_type, jobs, args, save_file, kws, selected_ids = param

    def _get_section_id(sect):
        try:
            sect = sect.lower()
            sect = sect.replace('.', '').strip()
        except:
            return 2

        if sect in kws['intro']:
            return 0
        elif sect in kws['related']:
            return 1

        # elif sect in kws['method']:
        #     return 2

        elif sect in kws['exp']:
            return 3

        elif sect in kws['res']:
            return 4

        elif sect in kws['conclusion']:
            return 5
        else:
            return 2

    def remove_ack(source):
        out = []
        for sent in source:
            if 'acknowledgment' in sent[1].lower() or 'acknowledgments' in sent[1].lower() \
                    or 'fund' in sent[1].lower() or 'funding' in sent[1].lower() \
                    or 'appendices' in sent[1].lower():
                continue
            else:
                out.append(sent)
        return out

    emp = 0
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return
    bert = BertData(args)
    datasets = []

    for j, data in tqdm(enumerate(jobs[0]), total=len(jobs)):
        source, tgt, id = data['src'], data['tgt'], data['id']
        paper_id = str(id)
        # selected_ids[paper_id] = selected_ids[paper_id][:60]
        if len(source) > 1 and len(tgt) > 0:
            segment = []
            source = remove_ack(source)
            segment_labels = []
            sent_rg_scores = []
            segment_sent_num = []
            segment_section = []
            token_ctr = 0
            i = 0
            # try:
            while i < len(source):
                sent = source[i]
                # if i in selected_ids[paper_id]:
                if len(sent[0]) + token_ctr < 1097:
                    segment.append(sent[0])
                    segment_section.append(_get_section_id(sent[1]))
                    segment_labels.append(sent[-1])
                    sent_rg_scores.append(sent[-3])
                    segment_sent_num.append(i)
                    token_ctr += len(sent[0])
                    import pdb;
                    pdb.set_trace()
                    if i == len(source) - 1:
                        # if i == selected_ids[paper_id][-1]:
                        token_ctr = 0
                        if (args.lower):
                            segment = [' '.join(s).lower().split() for s in segment]
                            tgt = [' '.join(s).lower().split() for s in tgt]
                        try:
                            b_data = bert.preprocess(segment.copy(), tgt, sent_rg_scores.copy(), segment_labels.copy(),
                                                     use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                                     is_test=is_test)
                        except:
                            import pdb;
                            pdb.set_trace()
                        if (b_data is None):
                            continue

                        src_subtoken_idxs, sent_rg_scores, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
                        try:
                            assert len(segment_labels) == len(
                                sent_rg_scores), "Number of segment_sent and section_sents should be the same"
                            assert len(cls_ids) == len(
                                sent_rg_scores), "Number of segment_sent and section_sents should be the same"
                        except:
                            import pdb;
                            pdb.set_trace()
                        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                                       "src_sent_labels": sent_rg_scores.copy(), "sent_labels": sent_labels.copy(),
                                       "segs": segments_ids, 'clss': cls_ids,
                                       'src_txt': src_txt, "tgt_txt": tgt_txt, "paper_id": paper_id,
                                       "sent_sect_labels": segment_section.copy()}

                        datasets.append(b_data_dict.copy())
                        segment_sent_num.clear()
                        segment.clear()
                        segment_labels.clear()
                        sent_labels.clear()
                        segment_section.clear()

                else:
                    if len(sent[0]) >= 1097 and len(segment) == 0:
                        source = source[:i] + source[i + 1:]
                        # i = i + 1
                        continue
                    # import pdb;pdb.set_trace()
                    i = i - 1
                    token_ctr = 0

                    # sent_labels = greedy_selection(segment, tgt, 2)
                    # sent_rg_scores = segment_rg_scores(segment, tgt)
                    # sent_rg_scores.append(sent[-2])
                    # segment_labels.append(sent[2])
                    # segment_section.append(_get_section_id(sent[1]))

                    if (args.lower):
                        segment = [' '.join(s).lower().split() for s in segment]
                        tgt = [' '.join(s).lower().split() for s in tgt]

                    if len(segment_section) != len(segment_labels):
                        import pdb;
                        pdb.set_trace()

                    if len(segment) != len(segment_labels):
                        import pdb;
                        pdb.set_trace()

                    if len(segment) != len(segment_section):
                        import pdb;
                        pdb.set_trace()

                    # import pdb;pdb.set_trace()
                    b_data = bert.preprocess(segment.copy(), tgt, sent_rg_scores.copy(), segment_labels.copy(),
                                             use_bert_basic_tokenizer=args.use_bert_basic_tokenizer, is_test=is_test)

                    # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)
                    if (b_data is None):
                        continue
                    src_subtoken_idxs, sent_rg_scores, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data

                    if len(cls_ids) != len(sent_rg_scores):
                        import pdb;
                        pdb.set_trace()
                        continue

                    assert len(segment_section) == len(
                        segment_labels), "Number of segment_section and segment_labels should be the same"
                    assert len(cls_ids) == len(
                        sent_rg_scores), "Number of cls_ids and sent_labels should be the same"

                    if len(cls_ids) != len(segment_section):
                        import pdb;
                        pdb.set_trace()
                        continue

                    assert len(cls_ids) == len(
                        segment_section), "Number of cls_ids and segment_section should be the same"

                    b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                                   "src_sent_labels": sent_rg_scores.copy(), "sent_labels": sent_labels.copy(),
                                   "segs": segments_ids, 'clss': cls_ids,
                                   'src_txt': src_txt, "tgt_txt": tgt_txt, "paper_id": paper_id,
                                   "sent_sect_labels": segment_section.copy()}

                    datasets.append(b_data_dict.copy())
                    segment_sent_num.clear()
                    segment.clear()
                    sent_labels.clear()
                    segment_section.clear()
                    segment_labels.clear()
                i += 1
                # except:
                #     emp += 1
                #     continue
        else:
            emp += 1
            continue
    if emp > 0: print(f'Empty: {emp}')
    # if emp > 0: print(f'datasets: {len(datasets)}')
    print(f'datasets: {len(datasets)}')
    if emp > 0: print(f'Saving: {save_file}')
    # logger.info('Processed instances %d' % len(datasets))
    # logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def _format_to_bert_cspubsum_segmented(param):
    corpus_type, jobs, args, save_file, kws, selected_ids = param

    def _get_section_id(sect):
        try:
            sect = sect.lower()
            sect = sect.replace('.', '').strip()
        except:
            return 2

        if sect in kws['intro']:
            return 0
        elif sect in kws['related']:
            return 1

        # elif sect in kws['method']:
        #     return 2

        elif sect in kws['exp']:
            return 3

        elif sect in kws['res']:
            return 4

        elif sect in kws['conclusion']:
            return 5
        else:
            return 2

    def remove_ack(source):
        out = []
        for sent in source:
            if 'acknowledgment' in sent[1].lower() or 'acknowledgments' in sent[1].lower() \
                    or 'fund' in sent[1].lower() or 'funding' in sent[1].lower() \
                    or 'appendices' in sent[1].lower():
                continue
            else:
                out.append(sent)
        return out

    emp = 0
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return
    bert = BertData(args)
    datasets = []

    for j, data in tqdm(enumerate(jobs[0]), total=len(jobs)):
        source, tgt, id = data['src'], data['tgt'], data['id']
        paper_id = str(id)
        # selected_ids[paper_id] = selected_ids[paper_id][:60]
        if len(source) > 1 and len(tgt) > 0:
            segment = []
            source = remove_ack(source)
            segment_labels = []
            sent_rg_scores = []
            segment_sent_num = []
            segment_section = []
            token_ctr = 0
            whole_segments = list()
            i = 0
            # try:
            while i < len(source):
                sent = source[i]
                # if i in selected_ids[paper_id]:
                if len(sent[0]) + token_ctr < 1097:
                    segment.append(sent[0])
                    segment_section.append(_get_section_id(sent[1]))
                    segment_labels.append(sent[-1])
                    sent_rg_scores.append(sent[-3])
                    segment_sent_num.append(i)
                    token_ctr += len(sent[0])
                    if i == len(source) - 1:
                        # if i == selected_ids[paper_id][-1]:
                        token_ctr = 0
                        if (args.lower):
                            segment = [' '.join(s).lower().split() for s in segment]
                            tgt = [' '.join(s).lower().split() for s in tgt]
                        try:
                            b_data = bert.preprocess(segment.copy(), tgt, sent_rg_scores.copy(), segment_labels.copy(),
                                                     use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                                     is_test=is_test)
                        except:
                            import pdb;
                            pdb.set_trace()
                        if (b_data is None):
                            continue

                        src_subtoken_idxs, sent_rg_scores, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
                        try:
                            assert len(segment_labels) == len(
                                sent_rg_scores), "Number of segment_sent and section_sents should be the same"
                            assert len(cls_ids) == len(
                                sent_rg_scores), "Number of segment_sent and section_sents should be the same"
                        except:
                            import pdb;
                            pdb.set_trace()
                        # b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                        #                "src_sent_labels": sent_rg_scores.copy(), "sent_labels": sent_labels.copy(), "segs": segments_ids, 'clss': cls_ids,
                        #                'src_txt': src_txt, "tgt_txt": tgt_txt, "paper_id": paper_id,
                        #                "sent_sect_labels": segment_section.copy()}

                        whole_segments.append((src_subtoken_idxs, sent_rg_scores.copy(), sent_labels.copy(),
                                               segments_ids, cls_ids, src_txt, segment_section.copy()))
                        # datasets.append(b_data_dict.copy())
                        segment_sent_num.clear()
                        segment.clear()
                        segment_labels.clear()
                        sent_labels.clear()
                        segment_section.clear()

                else:
                    if len(sent[0]) >= 1097 and len(segment) == 0:
                        source = source[:i] + source[i + 1:]
                        # i = i + 1
                        continue
                    # import pdb;pdb.set_trace()
                    i = i - 1
                    token_ctr = 0

                    # sent_labels = greedy_selection(segment, tgt, 2)
                    # sent_rg_scores = segment_rg_scores(segment, tgt)
                    # sent_rg_scores.append(sent[-2])
                    # segment_labels.append(sent[2])
                    # segment_section.append(_get_section_id(sent[1]))

                    if (args.lower):
                        segment = [' '.join(s).lower().split() for s in segment]
                        tgt = [' '.join(s).lower().split() for s in tgt]

                    if len(segment_section) != len(segment_labels):
                        import pdb;
                        pdb.set_trace()

                    if len(segment) != len(segment_labels):
                        import pdb;
                        pdb.set_trace()

                    if len(segment) != len(segment_section):
                        import pdb;
                        pdb.set_trace()

                    # import pdb;pdb.set_trace()
                    b_data = bert.preprocess(segment.copy(), tgt, sent_rg_scores.copy(), segment_labels.copy(),
                                             use_bert_basic_tokenizer=args.use_bert_basic_tokenizer, is_test=is_test)

                    # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)
                    if (b_data is None):
                        continue
                    src_subtoken_idxs, sent_rg_scores, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data

                    if len(cls_ids) != len(sent_rg_scores):
                        import pdb;
                        pdb.set_trace()
                        continue

                    assert len(segment_section) == len(
                        segment_labels), "Number of segment_section and segment_labels should be the same"
                    assert len(cls_ids) == len(
                        sent_rg_scores), "Number of cls_ids and sent_labels should be the same"

                    if len(cls_ids) != len(segment_section):
                        import pdb;
                        pdb.set_trace()
                        continue

                    assert len(cls_ids) == len(
                        segment_section), "Number of cls_ids and segment_section should be the same"

                    whole_segments.append((src_subtoken_idxs, sent_rg_scores.copy(), sent_labels.copy(),
                                           segments_ids, cls_ids, src_txt, segment_section.copy()))

                    segment_sent_num.clear()
                    segment.clear()
                    sent_labels.clear()
                    segment_section.clear()
                    segment_labels.clear()
                i += 1

            b_data_dict = {"src": [s[0] for s in whole_segments], "tgt": tgt_subtoken_idxs,
                           "src_sent_labels": [s[1] for s in whole_segments],
                           "sent_labels": [s[2] for s in whole_segments], "segs": [s[3] for s in whole_segments],
                           'clss': [s[4] for s in whole_segments],
                           'src_txt': [s[5] for s in whole_segments], "tgt_txt": tgt_txt, "paper_id": paper_id,
                           "sent_sect_labels": [s[6] for s in whole_segments]}
            datasets.append(b_data_dict.copy())
        else:
            emp += 1
            continue
    if emp > 0: print(f'Empty: {emp}')
    # if emp > 0: print(f'datasets: {len(datasets)}')
    print(f'datasets: {len(datasets)}')
    if emp > 0: print(f'Saving: {save_file}')
    # logger.info('Processed instances %d' % len(datasets))
    # logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def _format_to_bert_cspubsum_only_src(param):
    corpus_type, json_file, args, save_file, kws = param
    print(save_file)

    # if save_file=='/disk1/sajad/datasets/sci/arxiv/bert-files/5l-new/Train.87.bert.pt' or \
    #         save_file=='/disk1/sajad/datasets/sci/arxiv/bert-files/5l-new/Train.274.bert.pt':
    #     return

    def _get_section_id(sect):
        try:
            sect = sect.lower()
        except:
            return 2

        if sect in kws['intro']:
            return 0
        elif sect in kws['related']:
            return 1

        elif sect in kws['exp']:
            return 3

        elif sect in kws['res']:
            return 4

        elif sect in kws['conclusion']:
            return 5
        else:
            return 2

    # jobs = json.load(open(json_file))
    jobs = json_file
    emp = 0
    is_test = corpus_type == 'test'
    print(save_file)
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return
    # if save_file == '/disk1/sajad/datasets/sci/arxiv/bert-files/5l-new/Train.274.bert.pt':
    #     return

    print(save_file)
    bert = BertData(args)
    datasets = []
    for j, data in tqdm(enumerate(jobs), total=len(jobs)):

        source = data['source']
        paper_id = hashhex(data['id'])

        if len(source) > 1 and len(tgt) > 0:
            segment = []
            segment_labels = []
            sent_rg_scores = []
            segment_sent_num = []
            segment_section = []
            token_ctr = 0
            i = 0
            # try:
            while i < len(source):
                sent = source[i]
                if len(sent[0]) + token_ctr < 1024:
                    segment.append(sent[0])
                    segment_section.append(_get_section_id(sent[1]))
                    segment_labels.append(sent[-1])
                    sent_rg_scores.append(sent[-2])
                    segment_sent_num.append(i)
                    token_ctr += len(sent[0])
                    if i == len(source) - 1:
                        token_ctr = 0
                        if (args.lower):
                            segment = [' '.join(s).lower().split() for s in segment]
                            tgt = [' '.join(s).lower().split() for s in tgt]

                        try:
                            b_data = bert.preprocess(segment.copy(), tgt, sent_rg_scores.copy(), segment_labels.copy(),
                                                     use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                                     is_test=is_test)
                        except:
                            continue
                        if (b_data is None):
                            continue
                        src_subtoken_idxs, sent_rg_scores, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
                        try:
                            assert len(segment_labels) == len(
                                sent_rg_scores), "Number of segment_sent and section_sents should be the same"
                            assert len(cls_ids) == len(
                                sent_rg_scores), "Number of segment_sent and section_sents should be the same"
                        except:
                            continue
                        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                                       "src_sent_labels": sent_rg_scores.copy(), "sent_labels": sent_labels.copy(),
                                       "segs": segments_ids, 'clss': cls_ids,
                                       'src_txt': src_txt, "tgt_txt": tgt_txt, "paper_id": paper_id,
                                       "sent_sect_labels": segment_section.copy()}

                        datasets.append(b_data_dict.copy())
                        segment_sent_num.clear()
                        segment.clear()
                        segment_labels.clear()
                        sent_labels.clear()
                        segment_section.clear()

                else:
                    if len(sent[0]) >= 1024 and len(segment) == 0:
                        source = source[:i] + source[i + 1:]
                        # i = i + 1
                        continue
                    # import pdb;pdb.set_trace()
                    i = i - 1
                    token_ctr = 0

                    # sent_labels = greedy_selection(segment, tgt, 2)
                    # sent_rg_scores = segment_rg_scores(segment, tgt)
                    # sent_rg_scores.append(sent[-2])
                    # segment_labels.append(sent[2])
                    # segment_section.append(_get_section_id(sent[1]))
                    if (args.lower):
                        segment = [' '.join(s).lower().split() for s in segment]
                        tgt = [' '.join(s).lower().split() for s in tgt]

                    # if len(segment_section) != len(segment_labels):
                    #     import pdb;pdb.set_trace()
                    #
                    # if len(segment) != len(segment_labels):
                    #     import pdb;pdb.set_trace()
                    #
                    # if len(segment) != len(segment_section):
                    #     import pdb;pdb.set_trace()

                    b_data = bert.preprocess(segment.copy(), tgt, sent_rg_scores.copy(), segment_labels.copy(),
                                             use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                             is_test=is_test)

                    # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)
                    if (b_data is None):
                        continue
                    src_subtoken_idxs, sent_rg_scores, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data

                    if len(cls_ids) != len(sent_rg_scores):
                        import pdb;
                        pdb.set_trace()
                        continue

                    assert len(segment_section) == len(
                        segment_labels), "Number of segment_section and segment_labels should be the same"
                    assert len(cls_ids) == len(
                        sent_rg_scores), "Number of cls_ids and sent_labels should be the same"

                    if len(cls_ids) != len(segment_section):
                        import pdb;
                        pdb.set_trace()
                        continue

                    assert len(cls_ids) == len(
                        segment_section), "Number of cls_ids and segment_section should be the same"

                    b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                                   "src_sent_labels": sent_rg_scores.copy(), "sent_labels": sent_labels.copy(),
                                   "segs": segments_ids, 'clss': cls_ids,
                                   'src_txt': src_txt, "tgt_txt": tgt_txt, "paper_id": paper_id,
                                   "sent_sect_labels": segment_section.copy()}
                    datasets.append(b_data_dict.copy())
                    segment_sent_num.clear()
                    segment.clear()
                    sent_labels.clear()
                    segment_section.clear()
                    segment_labels.clear()
                i += 1
            # except:
            #     emp += 1
            #     continue
        else:
            emp += 1
            continue
    if emp > 0: print(f'Empty: {emp}')
    if emp > 0: print(f'datasets: {len(datasets)}')
    if emp > 0: print(f'Saving: {save_file}')
    # logger.info('Processed instances %d' % len(datasets))
    # logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def _format_to_bert_csabs(param):
    corpus_type, json_file, args, save_file = param

    # print(save_file)

    # if save_file=='/disk1/sajad/datasets/sci/arxiv/bert-files/5l-new/Train.87.bert.pt' or \
    #         save_file=='/disk1/sajad/datasets/sci/arxiv/bert-files/5l-new/Train.274.bert.pt':
    #     return

    def _get_section_id(sect):
        if sect == 'objective':
            return 0
        elif sect == 'background':
            return 1
        elif sect == 'method':
            return 2
        elif sect == 'result':
            return 3
        elif sect == 'other':
            return 4

    jobs = json.load(open(json_file))
    emp = 0
    is_test = corpus_type == 'test'
    print(save_file)
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return
    # if save_file == '/disk1/sajad/datasets/sci/arxiv/bert-files/5l-new/Train.274.bert.pt':
    #     return

    print(save_file)
    bert = BertData(args)
    datasets = []
    for j, data in tqdm(enumerate(jobs), total=len(jobs)):
        source = data['source']
        paper_id = hashhex(data['id'])

        if len(source) > 1:
            segment = []
            segment_labels = []
            sent_rg_scores = []
            segment_sent_num = []
            segment_section = []
            i = 0
            # try:
            while i < len(source):
                # print(i)
                sent = source[i]
                # if i == 6:
                #     import pdb;pdb.set_trace()
                segment.append(sent[0])
                segment_labels.append(_get_section_id(sent[1]))
                sent_rg_scores.append(0)
                segment_sent_num.append(i)
                # token_ctr += len(sent[0])
                if i == len(source) - 1:
                    # token_ctr = 0
                    if (args.lower):
                        segment = [' '.join(s).lower().split() for s in segment]
                    try:
                        # if i == 6:
                        #     import pdb;
                        #     pdb.set_trace()
                        b_data = bert.preprocess(segment.copy(), [['this', 'tgt', 'is', 'not', 'real', '!']],
                                                 sent_rg_scores.copy(), segment_labels.copy(),
                                                 use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                                 is_test=is_test)
                    except:
                        continue
                    if (b_data is None):
                        continue
                    src_subtoken_idxs, sent_rg_scores, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
                    try:
                        assert len(segment_labels) == len(
                            sent_rg_scores), "Number of segment_sent and section_sents should be the same"
                        assert len(cls_ids) == len(
                            sent_rg_scores), "Number of segment_sent and section_sents should be the same"
                    except:
                        continue
                    b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                                   "src_sent_labels": sent_rg_scores.copy(), "sent_labels": sent_labels.copy(),
                                   "segs": segments_ids, 'clss': cls_ids,
                                   'src_txt': src_txt, "tgt_txt": tgt_txt, "paper_id": paper_id,
                                   "sent_sect_labels": segment_labels.copy()}

                    datasets.append(b_data_dict.copy())
                    segment_sent_num.clear()
                    segment.clear()
                    segment_labels.clear()
                    sent_labels.clear()
                    segment_section.clear()
                i += 1

                # else:
                #     if len(sent[0]) >= 1024 and len(segment) == 0:
                #         source = source[:i] + source[i + 1:]
                #         # i = i + 1
                #         continue
                #     # import pdb;pdb.set_trace()
                #     i = i - 1
                #     token_ctr = 0
                #
                #     # sent_labels = greedy_selection(segment, tgt, 2)
                #     # sent_rg_scores = segment_rg_scores(segment, tgt)
                #     # sent_rg_scores.append(sent[-2])
                #     # segment_labels.append(sent[2])
                #     # segment_section.append(_get_section_id(sent[1]))
                #     if (args.lower):
                #         segment = [' '.join(s).lower().split() for s in segment]
                #         tgt = [' '.join(s).lower().split() for s in tgt]
                #
                #     # if len(segment_section) != len(segment_labels):
                #     #     import pdb;pdb.set_trace()
                #     #
                #     # if len(segment) != len(segment_labels):
                #     #     import pdb;pdb.set_trace()
                #     #
                #     # if len(segment) != len(segment_section):
                #     #     import pdb;pdb.set_trace()
                #
                #     b_data = bert.preprocess(segment.copy(), tgt, sent_rg_scores.copy(), segment_labels.copy(),
                #                              use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                #                              is_test=is_test)
                #
                #
                #     # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)
                #     if (b_data is None):
                #         continue
                #     src_subtoken_idxs, sent_rg_scores, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
                #
                #     if len(cls_ids) != len(sent_rg_scores):
                #         import pdb;pdb.set_trace()
                #         continue
                #
                #     assert len(segment_section) == len(
                #         segment_labels), "Number of segment_section and segment_labels should be the same"
                #     assert len(cls_ids) == len(
                #         sent_rg_scores), "Number of cls_ids and sent_labels should be the same"
                #
                #     if len(cls_ids) != len(segment_section):
                #         import pdb;pdb.set_trace()
                #         continue
                #
                #     assert len(cls_ids) == len(
                #         segment_section), "Number of cls_ids and segment_section should be the same"
                #
                #
                #
                #     b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                #                    "src_sent_labels": sent_rg_scores.copy(), "sent_labels":sent_labels.copy(), "segs": segments_ids, 'clss': cls_ids,
                #                    'src_txt': src_txt, "tgt_txt": tgt_txt, "paper_id": paper_id,
                #                    "sent_sect_labels": segment_section.copy()}
                #     datasets.append(b_data_dict.copy())
                #     segment_sent_num.clear()
                #     segment.clear()
                #     sent_labels.clear()
                #     segment_section.clear()
                #     segment_labels.clear()
            # except:
            #     emp += 1
            #     continue
        else:
            emp += 1
            continue
    if emp > 0: print(f'Empty: {emp}')
    if emp > 0: print(f'datasets: {len(datasets)}')
    if emp > 0: print(f'Saving: {save_file}')
    # logger.info('Processed instances %d' % len(datasets))
    # logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def _format_to_bert(param):
    import pdb;
    pdb.set_trace()
    corpus_type, json_file, args, save_file, sent_sect_dict, sent_labels_dict = param
    emp = 0
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return
    bert = BertData(args)
    logger.info('Processing %s' % json_file)
    print(f"Reading {json_file}")
    jobs = json.load(open(json_file))
    datasets = []
    for j, data in tqdm(enumerate(jobs), total=len(jobs)):
        # tgt, id, source = data['tgt'], data['id'], data['src']
        import pdb;
        pdb.set_trace()
        source, tgt, paper_id = data['src'], data['tgt'], data['id']

        if len(source) > 1 and len(tgt) > 0 and len(paper_id) > 0:
            segment = []
            segment_sent_num = []
            token_ctr = 0
            i = 0
            segment_num = -1
            try:
                sent_labels_dict[paper_id] = sent_labels_dict[paper_id][:len(sent_sect_dict[paper_id])]
                while i < len(sent_sect_dict[paper_id]):
                    # try:
                    sent = source[i]

                    # except:
                    # import pdb;pdb.set_trace()
                    # print('Sentence not found')
                    # break
                    if len(sent) + token_ctr < 1022:
                        segment.append(sent)
                        segment_sent_num.append(i)
                        token_ctr += len(sent)

                        # print(i)
                        if i == len(sent_sect_dict[paper_id]) - 1:
                            token_ctr = 0
                            lst_segment = True
                            # sent_labels = greedy_selection(segment, tgt, 3)
                            sent_labels = sent_labels_dict[paper_id][:len(segment)]
                            sent_sect_labels = identify_sent_sects(sent_sect_dict[paper_id], segment_sent_num.copy(),
                                                                   lst_segment)
                            # if i==196:
                            # import pdb;
                            # pdb.set_trace()
                            # import pdb;
                            # pdb.set_trace()
                            if (args.lower):
                                segment = [' '.join(s).lower().split() for s in segment]
                                tgt = [' '.join(s).lower().split() for s in tgt]
                            b_data = bert.preprocess(segment.copy(), tgt, sent_labels,
                                                     use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                                     is_test=is_test)
                            if (b_data is None):
                                continue
                            src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data

                            import pdb;
                            pdb.set_trace()
                            try:
                                assert len(sent_sect_labels) == len(
                                    sent_labels), "Number of segment_sent and section_sents should be the same"
                                assert len(cls_ids) == len(
                                    sent_labels), "Number of segment_sent and section_sents should be the same"
                            except:
                                import pdb;
                                pdb.set_trace()

                            b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                                           "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                                           'src_txt': src_txt, "tgt_txt": tgt_txt, "paper_id": paper_id,
                                           "sent_sect_labels": sent_sect_labels}

                            datasets.append(b_data_dict.copy())
                            segment_sent_num.clear()
                            segment.clear()


                    else:
                        if len(sent) >= 1024 and len(segment) == 0:
                            # import pdb;pdb.set_trace()
                            sent_sect_dict[paper_id] = sent_sect_dict[paper_id][:i] + sent_sect_dict[paper_id][i + 1:]
                            sent_labels_dict[paper_id] = sent_labels_dict[paper_id][:i] + sent_labels_dict[paper_id][
                                                                                          i + 1:]
                            source = source[:i] + source[i + 1:]
                            # i = i + 1
                            continue
                        # import pdb;pdb.set_trace()
                        i = i - 1
                        token_ctr = 0
                        # sent_labels = greedy_selection(segment, tgt, 3)
                        sent_labels = sent_labels_dict[paper_id][:len(segment)]
                        sent_labels_dict[paper_id] = sent_labels_dict[paper_id][len(segment):]
                        sent_sect_labels = identify_sent_sects(sent_sect_dict[paper_id], segment_sent_num.copy())
                        if (args.lower):
                            segment = [' '.join(s).lower().split() for s in segment]
                            tgt = [' '.join(s).lower().split() for s in tgt]

                        b_data = bert.preprocess(segment.copy(), tgt, sent_labels,
                                                 use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                                 is_test=is_test)

                        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)
                        if (b_data is None):
                            continue

                        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
                        import pdb;
                        pdb.set_trace()

                        assert len(sent_sect_labels) == len(
                            sent_labels), "Number of segment_sent and section_sents should be the same"
                        assert len(cls_ids) == len(
                            sent_labels), "Number of segment_sent and section_sents should be the same"
                        assert len(cls_ids) == len(
                            sent_sect_labels), "Number of segment_sent and section_sents should be the same"

                        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                                       "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                                       'src_txt': src_txt, "tgt_txt": tgt_txt, "paper_id": paper_id,
                                       "sent_sect_labels": sent_sect_labels}
                        datasets.append(b_data_dict.copy())
                        segment_sent_num.clear()
                        segment.clear()

                    i += 1
            except:
                emp += 1
                continue
        else:

            emp += 1
            continue
    if emp > 0: print(f'Empty: {emp}')
    if emp > 0: print(f'datasets: {len(datasets)}')
    if emp > 0: print(f'Saving: {save_file}')
    # logger.info('Processed instances %d' % len(datasets))
    # logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def _format_to_lines(params):
    f, args = params
    print(f)
    source, tgt, id = load_json(f, args.lower)
    return {'id': id.replace('.txt', '').replace('.json', '').strip(), 'src': source, 'tgt': tgt}


def format_arxiv_to_lines(args):
    if args.dataset != '':
        corpus_type = args.dataset
    else:
        corpus_type = 'train'
    corpus_mapping = {}
    files = []
    for f in glob.glob('/disk1/sajad/datasets/sci/arxiv/inputs/' + corpus_type + '/*.json'):
        files.append(f)
    corpora = {corpus_type: files}
    for corpus_type in corpora.keys():
        a_lst = [(f, corpus_type, args) for f in corpora[corpus_type]]
        pool = Pool(7)
        dataset = []
        p_ct = 0
        # for a in a_lst:
        #     _format_arxiv_to_lines(a)
        for d in tqdm(pool.imap_unordered(_format_arxiv_to_lines, a_lst), total=len(a_lst)):
            dataset.append(d)
            # print(args.shard_size)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            print(args.save_path)
            pt_file = "{:s}{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            print(pt_file)
            with open(pt_file, 'w') as save:
                # save.write('\n'.join(dataset))
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _format_arxiv_to_lines(params):
    def load_arxiv_json(src_json, set, lower=True):
        source = []
        # flag = False

        sect_lens = json.load(open(src_json))['section_lengths']
        sect_names = json.load(open(src_json))['section_names']
        sent_sects = []

        for i, len in enumerate(sect_lens):
            for _ in range(int(len)):
                sent_sects.append(sect_names[i])

        id = json.load(open(src_json))['id']
        for i, sent_info in enumerate(json.load(open(src_json))['inputs']):
            tokens = sent_info['tokens']
            if (lower):
                tokens = [t.lower() for t in tokens]
            source.append((sent_sects[i], tokens))

        tgt_txt_path = src_json.split('arxiv/')[0] + 'arxiv/' + 'human-abstracts/' + set + '/' + id + '.txt'
        with open(tgt_txt_path, mode='r') as f:
            abs_text = f.read()
        abs_text = abs_text.strip()

        tknized_abs = tokenize_with_corenlp(abs_text, id)

        return id, source, tknized_abs

    f, corpus_type, args = params
    id, source, target = load_arxiv_json(f, corpus_type, args.lower)
    return {'id': id, 'src': source, 'tgt': target, 'section': ''}


def format_xsum_to_lines(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'test', 'valid']

    corpus_mapping = json.load(open(pjoin(args.raw_path, 'XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json')))

    for corpus_type in datasets:
        mapped_fnames = corpus_mapping[corpus_type]
        root_src = pjoin(args.raw_path, 'restbody')
        root_tgt = pjoin(args.raw_path, 'firstsentence')
        # realnames = [fname.split('.')[0] for fname in os.listdir(root_src)]
        realnames = mapped_fnames

        a_lst = [(root_src, root_tgt, n) for n in realnames]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_xsum_to_lines, a_lst):
            if (d is None):
                continue
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _format_xsum_to_lines(params):
    src_path, root_tgt, name = params
    f_src = pjoin(src_path, name + '.restbody')
    f_tgt = pjoin(root_tgt, name + '.fs')
    if (os.path.exists(f_src) and os.path.exists(f_tgt)):
        print(name)
        source = []
        for sent in open(f_src):
            source.append(sent.split())
        tgt = []
        for sent in open(f_tgt):
            tgt.append(sent.split())
        return {'src': source, 'tgt': tgt}
    return None


def format_longsum_to_lines(args):
    if args.dataset != '':
        corpus_type = args.dataset
    else:
        corpus_type = 'train'
    corpus_mapping = {}
    files = []
    for f in glob.glob(args.raw_path + '/' + corpus_type + '/*.json'):
        files.append(f)
    corpora = {corpus_type: files}
    for corpus_type in corpora.keys():
        a_lst = [(f) for f in corpora[corpus_type]]
        pool = Pool(7)
        dataset = []
        p_ct = 0

        # for a in a_lst:
        #     _format_longsum_to_lines(a)

        for d in tqdm(pool.imap_unordered(_format_longsum_to_lines, a_lst), total=len(a_lst)):

            if d is not None:
                dataset.append(d)
                # print(args.shard_size)
                if (len(dataset) > args.shard_size):
                    pt_file = "{:s}{:s}.{:d}.json".format(args.save_path + '', corpus_type, p_ct)
                    with open(pt_file, 'w') as save:
                        # save.write('\n'.join(dataset))
                        save.write(json.dumps(dataset))
                        p_ct += 1
                        dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            print(args.save_path)
            pt_file = "{:s}{:s}.{:d}.json".format(args.save_path + '', corpus_type, p_ct)
            print(pt_file)
            with open(pt_file, 'w') as save:
                # save.write('\n'.join(dataset))
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _format_longsum_to_lines(params):
    src_path = params

    def load_json(src_json, lower=True):
        print(src_json)
        paper = json.load(open(src_json))

        if len(paper['sentences']) < 10 or sum([len(sent) for sent in paper['gold']]) < 10:
            return -1, 0, 0
        id = paper['filename']
        for sent in paper['sentences']:
            tokens = sent[0]
            if (lower):
                tokens = [t.lower() for t in tokens]
                sent[0] = tokens

        for i, sent in enumerate(paper['gold']):
            tokens = sent
            if (lower):
                tokens = [t.lower() for t in tokens]
                paper['gold'][i] = tokens

        return paper['sentences'], paper['gold'], id

    paper_sents, paper_tgt, id = load_json(src_path)

    if paper_sents == -1:
        return None

    return {'id': id, 'src': paper_sents, 'tgt': paper_tgt}
