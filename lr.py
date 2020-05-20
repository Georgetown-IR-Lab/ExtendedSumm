import glob
import json
from tqdm import tqdm
import spacy
nlp = spacy.load("en_core_sci_sm", disable=['tagger', 'ner'])

files = glob.glob('/disk1/sajad/datasets/sci/arxiv/human-abstracts/train/*.txt')
avg = [0, 0]
num=0
for file in tqdm(files, total=len(files)):
    with open(file) as f:
        for l in f:
            if len(l.strip())>0:
                doc = nlp(l.strip())
                sents = doc.sents
                total_toks = 0

                for sent in sents:
                    for tok in sent:
                        total_toks += 1

                total_sents = len(list(doc.sents))
                avg[0] += total_toks
                avg[1] += total_sents
                num+=1
print(num)
print(f'avg toks= {avg[0]/num}, and avg sents= {avg[1] /num}')