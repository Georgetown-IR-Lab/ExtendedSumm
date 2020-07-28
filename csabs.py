import json

import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_sci_sm", disable=['ner', 'tagger'])
TRAINING_DATA_WRITE_LOC = '/disk1/sajad/datasets/sci/csabs/jsons/dev/'

c = 0
set = set()
with open('/home/sajad/packages/sequential_sentence_classification/data/CSAbstruct/dev.jsonl') as f:
    for l in tqdm(f, total=1668):
        abs = json.loads(l)
        sents_list = list()
        for sent, label in zip(abs['sentences'], abs['labels']):
            set.add(label)
        #     doc = nlp(sent)
        #     sents = list(doc.sents)
        #     sent_tokens = []
        #     for s in sents:
        #         for token in s:
        #             sent_tokens.append(token.text)
        #     sents_list.append((sent_tokens, label))
        #
        # data_item = {
        #     "filename": str(c),
        #     "sentences": sents_list
        # }

        # with open(TRAINING_DATA_WRITE_LOC + str(c) + ".json", "w") as fi:
        #     json.dump(data_item, fi)
        # c += 1

with open('sects_dev.txt', mode='w') as ff:
    for s in set:
        ff.write(s)
        ff.write('\n')
