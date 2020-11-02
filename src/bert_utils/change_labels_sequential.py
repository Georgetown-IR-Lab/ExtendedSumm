import argparse
import glob
import json
import operator
import os

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
# PT_DIRS = "/disk1/sajad/datasets/sci/arxiv-long/v1/bert-files/512-sectioned-seqAllen/"
# PT_DIRS = "/disk1/sajad/datasets/sci/pubmed-dataset/bert-files/480-plain/"
# PT_DIRS = "/disk1/sajad/datasets/sci/csabs/bert-files/5l/"

# PT_DIRS = "/disk1/sajad/datasets/sci/arxiv-long/v1/bert-files/512-sectioned-seqAllen/scibert/"

parser = argparse.ArgumentParser()
parser.add_argument("-read_from", default='')
parser.add_argument("-write_to", default='')
parser.add_argument("-predicted_labels", default='')
parser.add_argument("-set", default='')

args = parser.parse_args()
PT_DIRS = args.read_from
PT_DIRS_DEST = args.write_to
PREDICTED_LABELS = args.predicted_labels
se = args.set

def check_path_existense(dir):
    if os.path.exists(dir):
        return
    os.makedirs(dir)

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

    if new_labels.count(m_freq) / len(new_labels) >= 0.508:
        out = []
        for _ in range(len(new_labels)):
            out.append(m_freq)
    else:
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


def _get_sect_id(sent_label):
    if sent_label=='background_label':
        return 0
    elif sent_label=='method_label':
        return 1
    elif sent_label=='result_label':
        return 2
    elif sent_label=='objective_label':
        return 3
    elif sent_label=='other_label':
        return 4


lbld = 0
nonlbld = 0

ds_instance_labels = []
ds_instance_labels_ids = []

# with open('/home/sajad/packages/sequential_sentence_classification/pubmed_long_' + se + '.json') as F:
with open(PREDICTED_LABELS) as F:
    for l in F:
        inst = json.loads(l.strip())
        ds_instance_labels.append(inst)
        ds_instance_labels_ids.append(inst['segment_id'])
new_instances = []
prev_inst_counter = 0

for j, f in tqdm(enumerate(glob.glob(PT_DIRS + se + '*.pt')), total=len(glob.glob(PT_DIRS + se + '*.pt'))):
    instances = torch.load(f)

    for inst_idx, instance in enumerate(instances):
        new_instances.append(instance)
        paper_id = instance['paper_id']
        sentences = instance['src_txt']
        slables = instance['sent_labels']
        old_labels = instance['sent_sect_labels']
        import pdb;pdb.set_trace()
        position_in_dict = ds_instance_labels_ids.index(paper_id)

        new_labels = []
        for j, s in enumerate(sentences):
            try:
                sent_label = ds_instance_labels[position_in_dict]['labels'][j]
                lbld += 1
            except:
                import pdb;pdb.set_trace()
                sent_label = 4
                nonlbld += 1
            new_labels.append(sent_label)

        # new_labels_modified = replace_most_frequent(new_labels, paper_id)
        new_instances[-1]['sent_sect_labels'] = new_labels
    destination = f.replace('512-seqAllen-whole-sectioned-labels', '512-seqAllen-whole-sectioned-labels-sectionlabels')
    check_path_existense('/'.join(destination.split('/')[:-1]))
    # torch.save(new_instances, destination)
    prev_inst_counter += len(instances)
    print('Saved file: {}'.format(destination))
    new_instances.clear()
print(se + '-->')

print('{} / {} = {:4.2f} percent'.format(nonlbld, nonlbld + lbld, nonlbld / (nonlbld + lbld)))
