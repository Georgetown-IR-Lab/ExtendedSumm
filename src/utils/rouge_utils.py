import logging
import re

import numpy as np
# stopwords = pkgutil.get_data(__package__, 'smart_common_words.txt')
# stopwords = stopwords.decode('ascii').split('\n')
# stopwords = {key.strip(): 1 for key in stopwords}
import pandas as pd

from utils.rouge_score import evaluate_rouge

logger = logging.getLogger('ftpuploader')

INTRO_KWs_STR = "[introduction, introduction and motivation, motivation, motivations, basics and motivations, conclusions and discussion, introduction., [sec:intro]introduction, *introduction*, i. introduction, [sec:level1]introduction, introduction and motivation, introduction[sec:intro], [intro]introduction, introduction and main results, introduction[intro], introduction and summary, [sec:introduction]introduction, overview, 1. introduction, [sec:intro] introduction, introduction[sec:introduction], introduction and results, introduction and background, [sec1]introduction, [introduction]introduction, introduction and statement of results, introduction[introduction], introduction and overview, introduction:, [intro] introduction, [sec:1]introduction, authors contributions, introduction and main result, introduction[sec1], [sec:level1] introduction, motivations, outline, introductory remarks, [sec1] introduction, introduction and motivations, 1.introduction, introduction and definitions, introduction and notation, introduction and statement of the results, i.introduction, introduction[s1], [sec:level1]introduction +,  introduction., introduction[s:intro], [i]introduction, [sec:int]introduction, introduction and observations, [introduction] introduction, [sec:1] introduction, **introduction**, [seci]introduction, introduction and conclusions, **introduction**, [seci]introduction, introduction and summary of results, introduction and outline, preliminary remarks, general introduction, [sec:intr]introduction, [s1]introduction, introduction[sec_intro], introduction and statement of main results, scientific motivation, [sec:sec1]introduction, *questions*, introduction and the model, intoduction, challenges, introduction[sec-intro], introduction and result, inroduction, [sec:intro]introduction +, introdution, 1 introduction, brief summary, motivation and introduction, [1]introduction, introduction and related work, contributions, [sec:one]introduction, [section1]introduction, [sect:intro]introduction]"

INTRO_KWs_STR = "[" + str(['' + kw.strip() + '' for kw in INTRO_KWs_STR[1:-1].split(',')]) + "]"
INTRO_KWs = eval(INTRO_KWs_STR)[0]

CONC_KWs_STR = "[conclusion, conclusions, conclusion and future work, conclusions and future work, conclusion & future work, extensions, future work, related work and discussion, discussion and related work, conclusion and future directions, summary and future work, limitations and future work, future directions, conclusion and outlook, conclusions and future directions, conclusions and discussion, discussion and future directions, conclusions and discussions, conclusion and future direction, conclusions and future research, conclusion and future works, future plans, concluding remarks, conclusions and outlook, summary and outlook, final remarks, outlook, conclusion and outlook, conclusions and future work, summary and discussions, conclusion and future work, conclusions and perspectives, summary and concluding remarks, future work, conclusions., discussion and outlook, discussion & conclusions, open problems, remarks, conclusions[sec:conclusions], conclusion and perspectives, summary and future work, conclusion., summary & conclusions, closing remarks, final comments, future prospects, open questions, *conclusions*, [sec:conclusions]conclusions, conclusions and summary, comments, conclusion[sec:conclusion], perspectives, [sec:conclusion]conclusion, conclusions and future directions, summary & discussion, conclusions and remarks, conclusions and prospects, discussions and summary, future directions, conclusions and final remarks, the future, concluding comments, conclusions and open problems, summary[sec:summary], conclusions and future prospects, summary and remarks, conclusions and further work, conclusions[conclusions], [sec:summary]summary, comments and conclusions, summary and future prospects, [conclusion]conclusion, conclusion and remarks, concluding remark, further remarks, prospects, conclusion and open problems, conclusion and summary, v. conclusions, iv. conclusions,  summary and conclusions, summary and prospects, conclusions:, conclusion[conclusion], summary and final remarks, summary and future directions, summary & conclusion, [summary]summary, iv. conclusion, further questions, conclusion and future directions,  concluding remarks, further work, [conclusions]conclusions, outlook and conclusions, v. conclusion, *summary*, concluding remarks and open problems, conclusions and future works, future, [sec:conclusions] conclusions, [sec:concl]conclusions, remarks and conclusions, concluding remarks., conclusion and future works, summary., 4. conclusions, discussion and open problems, summary and comments, final remarks and conclusions, summary and conclusions., [sec:conc]conclusions, summary[summary], conclusions and open questions, [sec:conclusion]conclusions, further directions, conclusions and implications, conclusions & outlook, review, [sec:level1]conclusion, future developments, [sec:conc] conclusions, conclusions[sec:concl], conclusions and future perspectives, summary, conclusions and outlook, conclusions & discussion, [conclusions] conclusions, future research, concluding remarks and outlook, conclusions and future research, conclusion & outlook, discussion and future directions, conclusions[sec:conc], summary & outlook, vi. conclusions, future plans, [sec:summary] summary, conclusions and comments, conclusion and further work, conclusion and open questions, conclusions & future work, 5. conclusions, [sec:conclusion] conclusion, *concluding remarks*, iv. summary, conclusions[conc], conclusion:, [concl]conclusions, summary and perspective, conclusions[sec:conclusion], [sec:level1]conclusions, open issues, [sec:conc]conclusion, [sec:concl]conclusion, [sec:sum]summary, summary of the results, implications and conclusions, conclusions[conclusion], some remarks, conclusions[concl], conclusion and future research, conclusion remarks, vi. conclusion, perspective, conclusions and future developments, [conc]conclusion, general remarks, summary and conclusions[sec:summary], summary and open questions, 4. conclusion, conclusion and future prospects, concluding remarks and perspectives, remarks and questions, remarks and questions, [conclusion] conclusion, summary and implications, conclusive remarks, comments and conclusions, summary of conclusions, [conclusion]conclusions, conclusion and perspective, conclusion[sec:conc], [sec:summary]summary and conclusions, [sec:level1]summary, [sec:con]conclusion, [sec:level4]conclusion, conclusions and outlook., [summary]summary and conclusions, conclusion[sec:concl], 5. conclusion, [conc]conclusions, outlook and conclusion, remarks and conclusion,  summary and conclusion, conlusions, conclusion and final remarks, v. summary, future outlook, future improvements, summary and open problems, conclusion[concl], summary]"

CONC_KWs_STR = "[" + str(['' + kw.strip() + '' for kw in CONC_KWs_STR[1:-1].split(',')]) + "]"
CONC_KWs = eval(CONC_KWs_STR)[0]

ABSTRACT_KWs_STR = "[abstract, 0 abstract]"
ABSTRACT_KWs_STR = "[" + str(['' + kw.strip() + '' for kw in ABSTRACT_KWs_STR[1:-1].split(',')]) + "]"
ABS_KWs = eval(ABSTRACT_KWs_STR)[0]


def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
      n: which n-grams to calculate
      text: An array of tokens
    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


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


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

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
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def greedy_selection_section_based(doc_sent_list, abstract_sent_list, section_lens, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    def get_section_idx(index):
        section = 0
        for idx, s in enumerate(section_indeces):
            if index > s:
                section = idx
            if index < s:
                break
        return section

    def is_idx_in_eligible_sections(eligible_sections_idx, checked_idx):
        for sect_idx, sect_number in enumerate(eligible_sections_idx):
            if sect_idx != len(eligible_sections_idx):
                if checked_idx > section_indeces[sect_number] \
                        and checked_idx < section_indeces[sect_number + 1]:
                    return True
            else:
                if checked_idx > section_indeces[sect_number]:
                    return True

        return False

    section_indeces = [sum(section_lens[:i]) for i in range(len(section_lens))]

    section_indeces = [section_indeces[0]] + [s - 1 for s in section_indeces[1:]] + [
        sum(section_lens[:len(section_lens)]) - 1]

    max_rouge = 0.0

    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    selected_sents_dist = {}
    for idx, len_sect in enumerate(section_lens):
        selected_sents_dist[idx] = 0

    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1

        if s == 0:
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
                rouge_score = rouge_1 + rouge_2
                if rouge_score >= cur_max_rouge:
                    cur_max_rouge = rouge_score
                    cur_id = i

            if (cur_id == -1):
                continue

            selected.append(cur_id)
            max_rouge = cur_max_rouge
            selected_sents_dist[get_section_idx(cur_id)] += 1


        else:
            # premise: select sentences from sections that are less selected...
            # remove non-eligible sections...
            count_sections = []
            for sect, count in selected_sents_dist.items():
                count_sections.append(count)
            min_count = min(count_sections)

            if count_sections.count(min_count) >= 1:
                # there are multiple sections with min_count
                eligible_sections = []
                for idx, c in enumerate(count_sections):
                    if c == min_count:
                        eligible_sections.append(idx)

                # choose sentences from the eligible sections
                for i in range(len(sents)):
                    if (i in selected or
                            not is_idx_in_eligible_sections(eligible_sections, i)):
                        continue
                    c = selected + [i]
                    candidates_1 = [evaluated_1grams[idx] for idx in c]
                    candidates_1 = set.union(*map(set, candidates_1))
                    candidates_2 = [evaluated_2grams[idx] for idx in c]
                    candidates_2 = set.union(*map(set, candidates_2))
                    rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
                    rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
                    rouge_score = rouge_1 + rouge_2

                    if rouge_score > cur_max_rouge:
                        cur_max_rouge = rouge_score
                        cur_id = i

                if (cur_id == -1):
                    continue

                selected.append(cur_id)
                max_rouge = cur_max_rouge

                selected_sents_dist[get_section_idx(cur_id)] += 1

    # import pdb;
    # pdb.set_trace()
    return sorted(selected), selected_sents_dist


def transfer_to_5label(selected_sents_dist):
    def check_existense(kws, sect):
        for kw in kws:
            if len(kw.split()) < len(sect.split()):
                if kw in sect:
                    return True
            else:
                if sect in kw:
                    return True
        return False

    test_kws = pd.read_csv('csv_files/train_papers_sect8.csv')

    kws = {
        'intro': [kw.strip() for kw in test_kws['intro'].dropna()],
        'related': [kw.strip() for kw in test_kws['related work'].dropna()],
        'exp': [kw.strip() for kw in test_kws['experiments'].dropna()],
        'res': [kw.strip() for kw in test_kws['results'].dropna()],
        'conclusion': [kw.strip() for kw in test_kws['conclusion'].dropna()]
    }

    dist_5l = {'abstract': 0, 'intro': 0, 'method': 0, 'exp': 0, 'res': 0, 'conc': 0}
    for sect in selected_sents_dist.keys():
        if sect.lower() in 'abstract' or 'abstract' in sect.lower():
            dist_5l['abstract'] = selected_sents_dist[sect]
        elif sect.lower() in kws['intro'] or check_existense(kws['intro'], sect.lower()):
            dist_5l['intro'] += selected_sents_dist[sect]
        elif sect.lower() in kws['exp'] or check_existense(kws['exp'], sect.lower()):
            dist_5l['exp'] += selected_sents_dist[sect]
        elif sect.lower() in kws['res'] or check_existense(kws['res'], sect.lower()):
            dist_5l['res'] += selected_sents_dist[sect]
        elif sect.lower() in kws['conclusion'] or check_existense(kws['conclusion'], sect.lower()):
            dist_5l['conc'] += selected_sents_dist[sect]
        else:
            dist_5l['method'] += selected_sents_dist[sect]

    return dist_5l


def greedy_selection_section_based_intro_conc(paper_id, doc_sent_list, abstract_sent_list, section_lens, sections_text,
                                              summary_size, doc_section_list=None):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    def get_section_idx(index):
        section = 0
        for idx, s in enumerate(section_indeces):
            if index > s:
                section = idx
            if index < s:
                break
        return section

    def is_idx_in_sections(section_idx, checked_idx):
        for sect_idx, sect_number in enumerate(section_idx):
            if sect_number < len(section_lens):
                if checked_idx >= section_indeces[sect_number] \
                        and checked_idx < section_indeces[sect_number + 1]:
                    return True
            else:
                if checked_idx >= section_indeces[sect_number]:
                    return True

        return False

    def is_idx_in_sectionIDs(section_ids, checked_idx):

        for sect_id in section_ids:
            if doc_section_list[checked_idx] == sect_id:
                return True
        return False

    def get_eligible_sections(fetch_sections):
        out = []
        for sect in fetch_sections:

            if sect == 'abstract':
                jump = False
                for kw in ABS_KWs:
                    for j, sect_heading in enumerate(sections_text):
                        if kw.lower() in sect_heading:
                            out.append(j)
                            jump = True
                            break
                    if jump:
                        break

            if sect == 'introduction':

                jump = False
                for kw in INTRO_KWs:
                    for j, sect_heading in enumerate(sections_text):
                        if kw.lower() in sect_heading:
                            out.append(j)
                            jump = True
                            break
                    if jump:
                        break

            if sect == 'conclusion':
                jump = False
                for kw in CONC_KWs:
                    for j, sect_heading in enumerate(sections_text):
                        if kw.lower() in sect_heading:
                            out.append(j)
                            jump = True
                            break
                    if jump:
                        break

        if len(out) == 0:
            return [0]
        return out

    def get_eligible_section_ids(fetch_sections):
        out = []
        for sect in fetch_sections:
            # if sect == 4:
            #     continue
            # else:
            out.append(sect)
        return out

    sections_text = [s.lower() for s in sections_text]
    doc_sent_list = np.array(doc_sent_list, dtype=object)
    section_indeces = [sum(section_lens[:i]) for i in range(len(section_lens))]
    section_indeces = [section_indeces[0]] + [s for s in section_indeces[1:]] + [sum(section_lens[:len(section_lens)])]
    picked_sent_intro_conc = 0
    max_rouge = 0.0

    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    reference_2grams = _get_word_ngrams(2, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]

    selected = []
    selected_sents_dist = {}
    for idx, len_sect in enumerate(section_lens):
        selected_sents_dist[idx] = 0

    if doc_section_list is not None:
        selected_sents_dist_sectionIDs = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}

    get_eligible_parts = get_eligible_sections
    eligible_parts_args = ['abstract', 'introduction', 'conclusion']
    is_idx_in_global = is_idx_in_sections
    if doc_section_list is not None:
        get_eligible_parts = get_eligible_section_ids
        eligible_parts_args = [0, 1]
        is_idx_in_global = is_idx_in_sectionIDs

    for s in range(summary_size - (5 * summary_size) // 10):
        cur_max_rouge = max_rouge
        cur_id = -1
        eligible_sections_intro_conc = get_eligible_parts(eligible_parts_args)
        for i in range(len(sents)):
            if (i in selected or not is_idx_in_global(eligible_sections_intro_conc, i)):
                continue
            c = selected + [i]
            candidates_1grams = [evaluated_1grams[idx] for idx in c]
            candidates_1grams = set.union(*map(set, candidates_1grams))
            candidates_2grams = [evaluated_2grams[idx] for idx in c]
            candidates_2grams = set.union(*map(set, candidates_2grams))
            rouge_1 = cal_rouge(candidates_1grams, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2grams, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score >= cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            break

        selected.append(cur_id)
        max_rouge = cur_max_rouge
        selected_sents_dist[get_section_idx(cur_id)] += 1
        if doc_section_list is not None:
            selected_sents_dist_sectionIDs[str(doc_section_list[cur_id])] += 1
        picked_sent_intro_conc += 1

    selected = sorted(selected)
    selected_intro_conc = selected.copy()


    if len(selected_intro_conc) < (summary_size - (8 * summary_size) // 10):
        sampling_sections = [0,1,2, 3]
    else:
        sampling_sections = [2, 3]

    if len(sampling_sections) == 0:
        sampling_sections = get_eligible_parts(['abstract', 'introduction', 'conclusion'])
    # expand the summary sentences, select 2 expandable sentences for each summary sentence

    selected_samples = list()
    for i_intro in selected_intro_conc:
        selected = list()
        max_rouge = 0.0
        reference_1grams = _get_word_ngrams(1, [doc_sent_list[i_intro]])
        reference_2grams = _get_word_ngrams(2, [doc_sent_list[i_intro]])

        for s in range(((summary_size - picked_sent_intro_conc) // picked_sent_intro_conc)):
            cur_max_rouge = max_rouge
            cur_id = -1
            for i in range(len(sents)):
                if i in selected:
                    continue
                if (i in selected_samples):
                    continue
                if (i in selected_intro_conc):
                    continue
                if (not is_idx_in_global(sampling_sections, i)):
                    continue

                c = selected + [i]
                candidates_1grams = [evaluated_1grams[idx] for idx in c]
                candidates_1grams = set.union(*map(set, candidates_1grams))

                candidates_2grams = [evaluated_2grams[idx] for idx in c]
                candidates_2grams = set.union(*map(set, candidates_2grams))

                rouge_1 = cal_rouge(candidates_1grams, reference_1grams)['f']
                rouge_2 = cal_rouge(candidates_2grams, reference_2grams)['f']

                rouge_score = rouge_1 + rouge_2
                if rouge_score >= cur_max_rouge:
                    cur_max_rouge = rouge_score
                    cur_id = i

            if cur_id == -1:
                break

            selected.append(cur_id)
            max_rouge = cur_max_rouge
            selected_sents_dist[get_section_idx(cur_id)] += 1
            if doc_section_list is not None:
                selected_sents_dist_sectionIDs[str(doc_section_list[cur_id])] += 1
        for s in selected:
            selected_samples.append(s)
            if len(selected_samples) + len(selected_intro_conc) == summary_size:
                break
        if len(selected_samples) + len(selected_intro_conc) == summary_size:
            break


    selected_final = selected_samples.copy() + selected_intro_conc.copy()

    if len(selected_final) < summary_size:
        selected_samples = list()
        for i_intro in selected_intro_conc:
            selected = list()
            max_rouge = 0.0
            reference_1grams = _get_word_ngrams(1, [doc_sent_list[i_intro]])
            reference_2grams = _get_word_ngrams(2, [doc_sent_list[i_intro]])

            for s in range(((summary_size - picked_sent_intro_conc) // picked_sent_intro_conc) + 1):
                cur_max_rouge = max_rouge
                cur_id = -1
                for i in range(len(sents)):
                    if i in selected:
                        continue
                    if (i in selected_samples):
                        continue
                    if (i in selected_final):
                        continue
                    if (not is_idx_in_global(sampling_sections, i)):
                        continue

                    c = selected + [i]
                    candidates_1grams = [evaluated_1grams[idx] for idx in c]
                    candidates_1grams = set.union(*map(set, candidates_1grams))

                    candidates_2grams = [evaluated_2grams[idx] for idx in c]
                    candidates_2grams = set.union(*map(set, candidates_2grams))

                    rouge_1 = cal_rouge(candidates_1grams, reference_1grams)['f']
                    rouge_2 = cal_rouge(candidates_2grams, reference_2grams)['f']

                    rouge_score = rouge_1 + rouge_2
                    if rouge_score >= cur_max_rouge:
                        cur_max_rouge = rouge_score
                        cur_id = i

                if cur_id == -1:
                    break

                selected.append(cur_id)
                max_rouge = cur_max_rouge
                selected_sents_dist[get_section_idx(cur_id)] += 1
                if doc_section_list is not None:
                    selected_sents_dist_sectionIDs[str(doc_section_list[cur_id])] += 1

            for s in selected:
                selected_samples.append(s)
                if len(selected_samples) + len(selected_final) == summary_size:
                    break
            if len(selected_samples) + len(selected_final) == summary_size:
                break

        selected_final += selected_samples
    heading_dict = {}
    for heading, count in zip(sections_text, selected_sents_dist.values()):
        heading_dict[heading] = count
    selected_sents_dist_5labels = transfer_to_5label(heading_dict)

    percentage = {}
    for k, v in selected_sents_dist_5labels.items():
        try:
            percentage[k] = v / sum(selected_sents_dist_5labels.values())
        except:
            percentage[k] = v / summary_size

    if doc_section_list is not None:
        percentage_section_ID = {}
        for k, v in selected_sents_dist_sectionIDs.items():
            try:
                percentage_section_ID[k] = v / sum(selected_sents_dist_sectionIDs.values())
            except Exception as e:
                # logger.error('Failed since: ' + str(e))
                percentage_section_ID[k] = v / summary_size

    # oracle = ''
    # for pred in sorted(selected_final):
    #     oracle += ' '.join(doc_sent_list[pred])
    #     oracle += ' '

    # print(evaluate_rouge([oracle.strip()], [' '.join([' '.join(g) for g in abstract_sent_list])]))
    # import pdb;pdb.set_trace()
    return sorted(selected_final), selected_sents_dist, percentage, percentage_section_ID
