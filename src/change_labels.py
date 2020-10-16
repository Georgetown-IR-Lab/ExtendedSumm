import glob
import operator
import pickle

import torch
from tqdm import tqdm


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i

    return num


# PT_DIRS = "/home/sajad/datasets/csp/bert-files/sectioned-512-seqIndex/"
# PT_DIRS = "/disk1/sajad/datasets/sci/pubmed-dataset/bert-files/512-plain/"
# PT_DIRS = "/disk1/sajad/datasets/sci/arxiv-long/v1/bert-files/sectioned-myIndex/"
PT_DIRS = "/disk1/sajad/datasets/sci/csabs/bert-files/5l/"


def reform_single_elem(A):
    k = 0
    while k < len(A):
        if k != 0 and k != len(A) - 1 and [k] != A[k + 1] and A[k] != A[k - 1] and A[k - 1] == A[k + 1]:
            A[k] = A[k + 1]
        k += 1

    return A

def longest_run(A, number):


    count = 0
    longest = 0
    i = 0

    while i < len(A):
        if A[i] == number:
            count += 1
        else:
            if count > longest:
                longest = count
            count = 0
        i += 1

    if count > longest:
        longest = count

    # for i in range(0, len(A)):
    #     if A[i] == number:
    #         count += 1
    #     else:
    #         if count > longest:
    #             longest = count
    #         count = 0

    return longest


def replace_most_frequent(new_labels, pid):
    m_freq = most_frequent(new_labels)
    out = new_labels.copy()

    # most freq
    if new_labels.count(m_freq) / len(new_labels) >= 0.508:
        out = []
        for _ in range(len(new_labels)):
            out.append(m_freq)
    else:
        # longest sequence
        # ol = new_labels.copy()
        new_labels = reform_single_elem(new_labels)

        lr0 = longest_run(new_labels, 0)
        lr1 = longest_run(new_labels, 1)
        lr2 = longest_run(new_labels, 2)

        lrs = [lr0, lr1, lr2]
        index, _ = max(enumerate(lrs), key=operator.itemgetter(1))
        out = []
        for _ in range(len(new_labels)):
            out.append(index)
    return out


for se in ["train", "val", "test"]:
    lbld = 0
    nonlbld = 0
    paper_labels = pickle.load(open('sentence_labels/' + se + '_labels_arxLong_myIndex.p', mode='rb'))
    new_instances = []
    ss = set()
    sss = set()
    for f in tqdm(glob.glob(PT_DIRS + se + '*.pt'), total=len(glob.glob(PT_DIRS + se + '*.pt'))):

        # if os.path.exists(f.replace('sectioned-512-seqIndex', 'sectioned-512-myIndex')):
        #     continue

        instances = torch.load(f)
        for instance in instances:
            new_instances.append(instance)
            paper_id = instance['paper_id']
            sentences = instance['src_txt']
            old_labels = instance['sent_sect_labels']
            for o in old_labels:
                sss.add(o)
            new_labels = []
            for j, s in enumerate(sentences):
                try:
                    sent_label = paper_labels[paper_id][s.lower().replace(' ', '')[:60]]
                    lbld += 1
                except:
                    if len(new_labels) > 0:
                        sent_label = most_frequent(new_labels)
                    else:
                        try:
                            sent_label = paper_labels[paper_id][sentences[j + 1].lower().replace(' ', '')[:60]]
                        except:
                            try:
                                sent_label = paper_labels[paper_id][sentences[j + 2].lower().replace(' ', '')[:60]]
                            except:
                                sent_label=1

                    nonlbld += 1
                ss.add(sent_label)
                new_labels.append(sent_label)

            new_labels_modified = replace_most_frequent(new_labels, paper_id)
            new_instances[-1]['sent_sect_labels'] = new_labels_modified

        # torch.save(new_instances, f)
        print('Saved file: {}'.format(f))
        new_instances.clear()
    print(se + '-->')
    print(sss)

    print('{} / {} = {:4.2f} percent'.format(nonlbld, nonlbld + lbld, nonlbld / (nonlbld + lbld)))
