import json
import re
import sys

import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm
from rouge_score import scoring


def impose_max_length(summary_text, max_tokens=600):
    # same tokenization as in rouge_score
    # https://github.com/google-research/google-research/blob/26a130831ee903cb97b7d04e71f227bbe24960b2/rouge/tokenize.py
    text = summary_text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = re.split(r"\s+", text)
    tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", x)]
    tokens = tokens[0:min(max_tokens, len(tokens))]
    return " ".join(tokens)

def evaluate_rouge(hypotheses, references, type='f'):
    metrics = ['rouge1', 'rouge2', 'rougeL']
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    results = {"rouge1_f": [], "rouge1_r": [], "rouge1_p": [], "rouge2_f": [],
               "rouge2_r": [], "rouge2_p": [], "rougeL_f": [], "rougeL_r": [], "rougeL_p": []}
    results_avg = {}

    if len(hypotheses) < len(references):
        print("Warning number of papers in submission file is smaller than ground truth file", file=sys.stderr)
    # import pdb;pdb.set_trace()
    hypotheses = list(hypotheses)
    references = list(references)
    for j, hyp in enumerate(hypotheses):
        submission_summary = hyp.replace('<q>', ' ')

        scores = scorer.score(references[j].strip(), submission_summary.strip())

        for metric in metrics:
            results[metric + "_f"].append(scores[metric].fmeasure)
            results[metric + "_r"].append(scores[metric].recall)
            results[metric + "_p"].append(scores[metric].precision)

        for rouge_metric, rouge_scores in results.items():
            results_avg[rouge_metric] = np.average(rouge_scores)

    return results_avg['rouge1_' + type], results_avg['rouge2_'+ type], results_avg['rougeL_'+ type]


def evaluate_rouge_avg(hypotheses, references, type='f', use_progress_bar=False):
    metrics = ['rouge1', 'rouge2', 'rougeL']
    scorer = {}
    scorer["rouge"] = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    aggregators_dict = {k: scoring.BootstrapAggregator() for k in scorer}

    if len(hypotheses) < len(references):
        print("Warning number of papers in submission file is smaller than ground truth file", file=sys.stderr)
    # import pdb;pdb.set_trace()
    hypotheses = list(hypotheses)
    references = list(references)

    if not use_progress_bar:
        for j, hyp in enumerate(hypotheses):
            submission_summary = hyp.replace('<q>', ' ')
            for key, scorr in scorer.items():
                scores_i = scorr.score(references[j].strip(), submission_summary)
                aggregators_dict[key].add_scores(scores_i)

        aggregates_dict = {k: v.aggregate() for k, v in aggregators_dict.items()}
        out_avg_scores = {}
        for k, v in sorted(aggregates_dict["rouge"].items()):
            out_avg_scores[k] = v.mid.fmeasure
    else:
        for j, hyp in tqdm(enumerate(hypotheses), total=len(hypotheses)):
            submission_summary = hyp.replace('<q>', ' ')
            for key, scorr in scorer.items():
                scores_i = scorr.score(references[j].strip(), submission_summary)
                aggregators_dict[key].add_scores(scores_i)

        aggregates_dict = {k: v.aggregate() for k, v in aggregators_dict.items()}
        out_avg_scores = {}
        for k, v in sorted(aggregates_dict["rouge"].items()):
            out_avg_scores[k] = v.mid.fmeasure
    return out_avg_scores['rouge1'], out_avg_scores['rouge2'], out_avg_scores['rougeL']


