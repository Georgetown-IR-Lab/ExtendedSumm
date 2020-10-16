# import glob
# import operator
# import pickle
#
# import torch
# from tqdm import tqdm
# import random
# PT_DIRS = '/disk1/sajad/datasets/sci/csabs/bert-files/5l/'
# for se in ["train"]:
#     new_instances = []
#     whole_instances = []
#     for f in tqdm(glob.glob(PT_DIRS + se + '*.pt'), total=len(glob.glob(PT_DIRS + se + '*.pt'))):
#
#         instances = torch.load(f)
#         for instance in instances:
#             whole_instances.append(instance)
#
#
#
#
#
#         # torch.save(new_instances, f)
#         print('Saved file: {}'.format(f))
#         new_instances.clear()
#     random.seed(888)
#     random.shuffle(whole_instances)
#
#     c = 0
#     iteration_counter = 0
#     iteration_instance_list = []
#     for instance in whole_instances:
#         iteration_instance_list.append(instance)
#         iteration_counter+=1
#         if iteration_counter==2000:
#             iteration_counter=0
#             torch.save(iteration_instance_list.copy(), PT_DIRS.replace('5l','5l-comb') + se + '.' + str(c) + '.pt')
#             iteration_instance_list.clear()
#             c+=1

s=20.34
ss = [21.38, 20.50, 20.46, 19.59]

for st in ss:
    print('{}'.format(((s-st)/st ) *100))