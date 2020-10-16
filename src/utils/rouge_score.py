import json
import re
import sys

from rouge_score import rouge_scorer
import numpy as np

def impose_max_length(summary_text, max_tokens=600):
    # same tokenization as in rouge_score
    # https://github.com/google-research/google-research/blob/26a130831ee903cb97b7d04e71f227bbe24960b2/rouge/tokenize.py
    text = summary_text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = re.split(r"\s+", text)
    tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", x)]
    tokens = tokens[0:min(max_tokens, len(tokens))]
    return " ".join(tokens)

def evaluate_rouge(hypotheses, references):
    metrics = ['rouge1', 'rouge2', 'rougeL']
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    results = {"rouge1_f": [], "rouge1_r": [], "rouge2_f": [], "rouge2_r": [], "rougeL_f": [], "rougeL_r": []}
    results_avg = {}

    if len(hypotheses) < len(references):
        print("Warning number of papers in submission file is smaller than ground truth file", file=sys.stderr)
    # import pdb;pdb.set_trace()
    hypotheses = list(hypotheses)
    references = list(references)
    for j, hyp in enumerate(hypotheses):
        submission_summary = hyp.replace('<q>', ' ')

        # submission_summary = impose_max_length(submission_summary)
        # ground_truth_summary = impose_max_length(references[j].replace('<q>',' '))

        scores = scorer.score(references[j].strip(), submission_summary.strip())

        for metric in metrics:
            results[metric + "_f"].append(scores[metric].fmeasure)
            results[metric + "_r"].append(scores[metric].recall)

        for rouge_metric, rouge_scores in results.items():
            results_avg[rouge_metric] = np.average(rouge_scores)

    return results_avg['rouge1_f']*100, results_avg['rouge2_f']*100, results_avg['rougeL_f']*100
