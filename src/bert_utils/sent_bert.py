import collections
import pickle
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

saved_list = pickle.load(open("save_list_arxiv_test_rg.p", "rb"))


sentbert_embs = collections.defaultdict(dict)
for s, val in tqdm(saved_list.items(), total=len(saved_list)):
    p_id, sent_scores, paper_srcs, paper_tgt, sent_sects_whole_true, source_sent_encodings, \
    sent_sects_whole_true, section_textual, _, _ = val

    sentbert_embs[p_id] = []

    for sent in paper_srcs:
        sentbert_embs[p_id].append(model.encode(sent))

pickle.dump( sentbert_embs, open( "embs_save_list_arxiv_test_rg.p", "wb" ) )