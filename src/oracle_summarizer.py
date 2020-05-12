import json

import utils.rouge
from utils.rouge import get_rouge
from prepro.data_builder import greedy_selection, segment_rg_scores
from tqdm import tqdm

SRC = '/home/sajad/packages/sum/scientific-paper-summarisation/Data/Test_Data/all_data_test.json'
oracles = []
golds = []
with open(SRC) as f:
    for l in tqdm(f, total=9883):
        data = json.loads(l)

        source = data['sentences']
        gold = data['gold']
        golds.append('<q>'.join([' '.join(s) for s in gold]))
        source_sents = [s[0] for s in source]
        # source = [' '.join(s) for s in source_sents]

        # sent_labels = greedy_selection(source_sents, gold, 4)
        oracle = segment_rg_scores(source_sents, gold, oracle=True)
        oracle = '<q>'.join([' '.join(o) for o in oracle[:4]])
        # oracle = []
        # for i, s in enumerate(source_sents):
        #     if i in sent_labels:
        #         oracle.append(' '.join(s))

        oracles.append(oracle)
r1, r2, rl, r1_cf, r2_cf, rl_cf = utils.rouge.get_rouge(oracles, golds, use_cf=True)
# print("{} set results:\n".format(args.filename))
print("Metric\tScore\t95% CI")
print("ROUGE-1\t{:.2f}\t({:.2f},{:.2f})".format(r1, r1_cf[0] - r1, r1_cf[1] - r1))
print("ROUGE-2\t{:.2f}\t({:.2f},{:.2f})".format(r2, r2_cf[0] - r2, r2_cf[1] - r2))
print("ROUGE-L\t{:.2f}\t({:.2f},{:.2f})".format(rl, rl_cf[0] - rl, rl_cf[1] - rl))