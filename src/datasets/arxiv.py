import gc
import os
import glob
import json
from multiprocessing.pool import Pool

import pandas
import torch
from tqdm import tqdm
from os.path import join as pjoin
import spacy
from others.tokenization import BertTokenizer, logger

nlp = spacy.load("en_core_sci_md", disable=['tagger', 'ner'])


class BertData():
    def __init__(self, args):
        self.args = args
        # self.tokenizer = BertTokenizer.from_pretrained('/disk1/sajad/pretrained-bert/scibert_scivocab_uncased/',
        #                                                do_lower_case=True)

        self.tokenizer = BertTokenizer.from_pretrained('/disk1/sajad/pretrained-bert/scibert_scivocab_uncased/',
                                                       do_lower_case=True)
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

def processor(params):
    fi, c_type = params
    with open(fi) as f:
        for l in f:
            paper_src = json.loads(l)
            abstract = get_summary(paper_src['id'], c_type)
            labels = get_labels(paper_src['id'], c_type)
            if len(abstract) == 0:
                return
            else:

                sent_sections = []

                for section, count in zip(paper_src['section_names'], paper_src['section_lengths']):
                    sent_sections.extend(count * [section])

                final_sents = list()

                for sent, label, section_name in zip(paper_src['inputs'], labels, sent_sections):
                    final_sents.append((sent['tokens'], sent['text'], section_name, label))

                # The data item that will be written for this paper
                data_item = {
                    "filename": paper_src['id'],
                    "gold": abstract,
                    "sentences": final_sents
                }
                if len(data_item['sentences']) < 10:
                    return

                with open('/disk1/sajad/datasets/sci/arxiv/json/' + c_type + '/' + paper_src['id'] + ".json", "w") as f:
                    json.dump(data_item, f)


def reform_papers(params):
    fi, c_type = params

    with open(fi) as f:
        for l in f:
            paper = json.loads(l)
            gold_str = paper['gold']

            doc_src = nlp(gold_str)
            doc_src = list(doc_src.sents)
            sents_tgt = []
            for sent in doc_src:
                toks = []
                doc_src = nlp(sent.text)
                for tok in doc_src:
                    if len(tok.text.strip()) > 0:
                        toks.append(tok.text)
                sents_tgt.append(toks)

            paper['gold'] = sents_tgt
            with open('/disk1/sajad/datasets/sci/arxiv/json/' + c_type + '-reformed' + '/' + paper['filename'] + ".json", "w") as f:
                json.dump(paper, f)



def get_summary(id, corpus_type):
    summary = ''
    with open('/disk1/sajad/datasets/sci/arxiv/human-abstracts/' + corpus_type + '/' + id + '.txt') as f:
        for l in f:
            summary = l.strip()
    return summary


def get_labels(id, corpus_type):
    labels = []
    with open('/disk1/sajad/datasets/sci/arxiv/labels/' + corpus_type + '/' + id + '.json') as f:
        for l in f:
          labels = json.loads(l)['labels']
    return list(labels)


def run_reform():
    pool = Pool(8)
    for set in ['train', 'val', 'test']:
    # for set in ['train']:
        files = []
        for in_file in glob.glob('/disk1/sajad/datasets/sci/arxiv/json/' + set + '/*.json'):
            files.append((in_file, set))
        for d in tqdm(pool.imap_unordered(reform_papers, files), total=len(files)):
            pass
        # for f in files:
        #     reform_papers(f)

def run():
    pool = Pool(8)
    # for set in ['train', 'val', 'test']:
    for set in ['train']:
        files = []
        for in_file in glob.glob('/disk1/sajad/datasets/sci/arxiv/inputs/' + set + '/*.json'):
            files.append((in_file, set))
        # for d in tqdm(pool.imap_unordered(processor, files), total=len(files)):
        #     pass



"""
    BERTSUM-related functions
    JSON and BERTIZE the dataset
"""

def format_arxiv_to_lines(args):
    if args.dataset != '':
        corpus_type = args.dataset
    else:
        corpus_type = 'train'
    files = []
    for f in glob.glob('/disk1/sajad/datasets/sci/arxiv/json/' + corpus_type + '-reformed'+ '/*.json'):
        files.append(f)
    corpora = {corpus_type: files}

    for corpus_type in corpora.keys():
        a_lst = [(f, corpus_type, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        # allowed_sampling = len(files) * args.sample_ratio
        for d in tqdm(pool.imap_unordered(_format_arxiv_to_lines, a_lst), total=len(a_lst)):
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}{:s}.{:d}.json".format(args.save_path.replace('json', 'json-aggregated'), corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

                # if j > allowed_sampling:
                #     jj = j
                #     break

        pool.close()
        pool.join()

        if (len(dataset) > 0):
            print(args.save_path.replace('json', 'json-aggregated'))
            pt_file = "{:s}{:s}.{:d}.json".format(args.save_path.replace('json', 'json-aggregated'), corpus_type, p_ct)
            print(pt_file)
            with open(pt_file, 'w') as save:
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []

        # print('Sampled %d instances for %s set' % (jj, corpus_type))


def _format_arxiv_to_lines(params):
    def load_arxiv_json(src_json, set, lower=True):
        with open(src_json) as f:
            for l in f:
                paper = json.loads(l)
        return paper['filename'], paper['sentences'], paper['gold']
    f, corpus_type, args = params
    id, source, target = load_arxiv_json(f, corpus_type, args.lower)
    if len(source) == 0:
        print('0 found ')
    return {'id': id, 'src': source, 'tgt': target}


def format_to_bert_arxiv(args):
    test_kws = pandas.read_csv('arxiv_sections_6l.csv')

    kws = {
        'intro': [kw.strip() for kw in test_kws['intro'].dropna()],
        'related': [kw.strip() for kw in test_kws['related work'].dropna()],
        'exp': [kw.strip() for kw in test_kws['experiments'].dropna()],
        'res':[kw.strip() for kw in test_kws['results'].dropna()],
        'conclusion': [kw.strip() for kw in test_kws['conclusion'].dropna()]
    }

    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['test']

    for corpus_type in datasets:
        a_lst = []
        c = 0
        for json_f in glob.glob(pjoin('/disk1/sajad/datasets/sci/arxiv/json-aggregated/', corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            c += 1
            a_lst.append(
                (corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt')), kws))
        print("Number of files: " + str(c))

        for a in a_lst:
            _fomat_to_bart_oracle(a)
        #
        # pool = Pool(args.n_cpus)
        # for d in tqdm(pool.imap(_fomat_to_bart_oracle, a_lst), total=len(a_lst)):
        #     pass
        #
        # pool.close()
        # pool.join()


def _fomat_to_bart(param):
    corpus_type, jobs, args, save_file, kws = param
    jobs = json.load(open(jobs))

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

    for j, data in enumerate(jobs):
        id, source, tgt = data['id'], data['src'], data['tgt']
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
                if len(sent[0]) + token_ctr < 1698:
                    segment.append(sent[0])
                    segment_labels.append(sent[-1])
                    sent_rg_scores.append(0)
                    segment_sent_num.append(i)
                    segment_section.append(_get_section_id(sent[-2]))
                    token_ctr += len(sent[0])
                    if i == len(source) - 1:
                        token_ctr = 0
                        # if (args.lower):
                        #     segment_str = ' '.join([s.lower() for s in segment])
                        #     tgt_str = ' '.join([' '.join(s).lower() for s in tgt])
                        if (args.lower):
                            segment = [[ss.lower() for ss in s] for s in segment]
                            tgt = [[ss.lower() for ss in s] for s in tgt]

                        try:
                            b_data = bert.preprocess(segment.copy(), tgt, sent_rg_scores.copy(), segment_labels.copy(),
                                                     use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                                     is_test=is_test)
                        except:
                            import pdb;pdb.set_trace()
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
                    if len(sent[0]) >= 1698 and len(segment) == 0:
                        source = source[:i] + source[i + 1:]
                        continue
                    i = i - 1
                    token_ctr = 0

                    # if (args.lower):
                    #     segment_str = ' '.join([s.lower() for s in segment])
                    #     tgt_str = ' '.join([' '.join(s).lower() for s in tgt])

                    if (args.lower):
                        segment = [[ss.lower() for ss in s] for s in segment]
                        tgt = [[ss.lower() for ss in s] for s in tgt]

                    if len(segment_section) != len(segment_labels):
                        import pdb;pdb.set_trace()

                    if len(segment) != len(segment_labels):
                        import pdb;pdb.set_trace()

                    if len(segment) != len(segment_section):
                        import pdb;pdb.set_trace()

                    # import pdb;pdb.set_trace()
                    b_data = bert.preprocess(segment.copy(), tgt, sent_rg_scores.copy(), segment_labels.copy(), use_bert_basic_tokenizer=args.use_bert_basic_tokenizer, is_test=is_test)


                    # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)
                    if (b_data is None):
                        continue
                    src_subtoken_idxs, sent_rg_scores, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data

                    if len(cls_ids) != len(sent_rg_scores):
                        import pdb;pdb.set_trace()
                        continue

                    assert len(segment_section) == len(
                        segment_labels), "Number of segment_section and segment_labels should be the same"
                    assert len(cls_ids) == len(
                        sent_rg_scores), "Number of cls_ids and sent_labels should be the same"

                    if len(cls_ids) != len(segment_section):
                        import pdb;pdb.set_trace()
                        continue

                    assert len(cls_ids) == len(
                        segment_section), "Number of cls_ids and segment_section should be the same"

                    b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                                   "src_sent_labels": sent_rg_scores.copy(), "sent_labels":sent_labels.copy(), "segs": segments_ids, 'clss': cls_ids,
                                   'src_txt': src_txt, "tgt_txt": tgt_txt, "paper_id": paper_id,
                                   "sent_sect_labels": segment_section.copy()}

                    datasets.append(b_data_dict.copy())
                    segment_sent_num.clear()
                    segment.clear()
                    sent_labels.clear()
                    segment_section.clear()
                    segment_labels.clear()

                i += 1
        else:
            emp += 1
            continue

    if emp > 0: print(f'Empty: {emp}')
    # if emp > 0: print(f'datasets: {len(datasets)}')
    # print(f'datasets: {len(datasets)}')
    # logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()



def _fomat_to_bart_oracle(param):
    corpus_type, jobs, args, save_file, kws = param
    jobs = json.load(open(jobs))

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
    for j, data in tqdm(enumerate(jobs), total=len(jobs)):
        id, source, tgt = data['id'], data['src'], data['tgt']
        paper_id = str(id)
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

                if sent[-1] == 1:
                    segment.append(sent[0])
                    segment_labels.append(sent[-1])
                    sent_rg_scores.append(0.1)
                    segment_sent_num.append(i)
                    segment_section.append(0)
                    token_ctr += len(sent[0])

                if i == len(source)-1:
                    # token_ctr = 0
                    if (args.lower):
                        segment = [[s.lower() for s in ss] for ss in segment]
                        tgt = [[s.lower() for s in ss] for ss in tgt]

                    try:
                        b_data = bert.preprocess(segment.copy(), tgt, sent_rg_scores.copy(), segment_labels.copy(), use_bert_basic_tokenizer=args.use_bert_basic_tokenizer, is_test=is_test)
                        # if j == 15:
                        #     print(i)
                            # if i==519:
                            #     import pdb;pdb.set_trace()
                    except:
                        import pdb;pdb.set_trace()
                    if (b_data is None):
                        break


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
                    # print(token_ctr)

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
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()



if __name__ == '__main__':
    run_reform()