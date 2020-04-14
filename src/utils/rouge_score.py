import csv
import os

from pythonrouge.pythonrouge import Pythonrouge
from progress.bar import ChargingBar


def csv_writer(filename, ref, hyp, p1, p2, p3):
    with open(filename, mode='a') as rouge_scores:
        employee_writer = csv.writer(rouge_scores, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        employee_writer.writerow([ref, hyp, p1, p2, p3])


def get_rouge(filename, hypothesis, references):

    assert len(hypothesis) == len(references)
    assert len(hypothesis) > 0
    r1_total = 0
    r2_total = 0
    rl_total = 0
    if os.path.exists(filename):
        os.remove(filename)

    csv_writer(filename, 'Gold', "Prediction", 'RG-1', 'RG-2', 'RG-l')
    bar = ChargingBar('Processing', max = len(references))
    print('\n')
    for hyp, ref in zip(hypothesis, references):
        summary = [[hyp]]
        reference = [[[ref]]]
        # initialize setting of ROUGE to eval ROUGE-1, 2, SU4
        # if you evaluate ROUGE by sentence list as above, set summary_file_exist=False
        # if recall_only=True, you can get recall scores of ROUGE
        rouge = Pythonrouge(summary_file_exist=False, summary=summary, reference=reference, \
            n_gram=2, ROUGE_SU4=False, ROUGE_L=True, recall_only=False, stemming=False, stopwords=False,\
            word_level=True, length_limit=False, use_cf=True, cf=95, scoring_formula='average', \
            resampling=True, samples=1000, favor=True, p=0.5)
        score = rouge.calc_score()
        # print(score)
        r1 = score['ROUGE-1-F'] * 100
        r2 = score['ROUGE-2-F'] * 100
        rl = score['ROUGE-L-F'] * 100
        r1_total += r1
        r2_total += r2
        rl_total += rl
        csv_writer(filename, ref, hyp, r1, r2, rl)
        bar.next()
    print('\nROUGE-1: {}'.format(r1_total / len(references)))
    print('\nROUGE-2: {}'.format(r2_total / len(references)))
    print('\nROUGE-l: {}'.format(rl_total / len(references)))
    csv_writer(filename, '', '', r1_total / len(references), r1_total / len(references), rl_total / len(references))

    bar.finish()

