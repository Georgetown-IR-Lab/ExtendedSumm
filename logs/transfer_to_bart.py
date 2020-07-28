

generated = []
with open('lsum-arxiv-first-phase-postprocessed-sorted-1_step35000.candidate') as f:
    for l in f:
        generated.append(l.strip().replace('<q>', ' '))

with open('/home/sajad/packages/sum/transformers/examples/seq2seq/lsum-oracle/test.source', mode='w') as F:
    for g in generated:
        F.write(g + '\n')
