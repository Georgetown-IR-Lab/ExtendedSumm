from others import utils
from others.logging import logger
from utils import rouge

arx_sents = []
arx = ''
with open('../logs/arxiv.txt') as f:
    for l in f:
        arx_sents.extend(l.split('<q>'))
        arx = l.strip()
lsum_sents = []
lsum=''
with open('../logs/lsum.txt') as f:
    for l in f:
        lsum_sents.extend([ls.strip() for ls in l.split('<q>')])
        lsum = l.strip()
gold=''
with open('../logs/gold.txt') as f:
    for l in f:
        gold = l.strip()

r1, r2, rl, r1_cf, r2_cf, rl_cf = rouge.get_rouge([arx], [gold], use_cf=True)

print("Metric\tScore\t95% CI")
print("ROUGE-1\t{:.2f}\t({:.2f},{:.2f})".format(r1, r1_cf[0] - r1, r1_cf[1] - r1))
print("ROUGE-2\t{:.2f}\t({:.2f},{:.2f})".format(r2, r2_cf[0] - r2, r2_cf[1] - r2))
print("ROUGE-L\t{:.2f}\t({:.2f},{:.2f})".format(rl, rl_cf[0] - rl, rl_cf[1] - rl))

r1, r2, rl, r1_cf, r2_cf, rl_cf = rouge.get_rouge([lsum], [gold], use_cf=True)

print("Metric\tScore\t95% CI")
print("ROUGE-1\t{:.2f}\t({:.2f},{:.2f})".format(r1, r1_cf[0] - r1, r1_cf[1] - r1))
print("ROUGE-2\t{:.2f}\t({:.2f},{:.2f})".format(r2, r2_cf[0] - r2, r2_cf[1] - r2))
print("ROUGE-L\t{:.2f}\t({:.2f},{:.2f})".format(rl, rl_cf[0] - rl, rl_cf[1] - rl))

# outf = open('logs/diff.txt', mode='w')
# for sent in arx_sents:
#     if sent.strip() not in lsum_sents:
#         outf.write(sent.strip())
#         outf.write('\n')