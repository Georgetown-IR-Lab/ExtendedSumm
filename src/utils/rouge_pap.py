# import glob
#
# import numpy as np
# import os
# import shutil
#
# import pandas as pd
# import rouge_papier_v2
#
# def check_ext(dir):
#     if os.path.exists(dir):
#         return
#     os.makedirs(dir)
#
# def write_summaries_to_file(hyothesis, references):
#     check_ext('tmp_rouge/')
#
#     counter = 0
#
#     for hyp, ref in zip(hyothesis, references):
#         with open('tmp_rouge/' + str(counter) + '.hyp', mode='w') as fS:
#             fS.write(hyp)
#             fS.write('\n')
#
#         with open('tmp_rouge/' + str(counter) + '.ref', mode='w') as fT:
#             fT.write(ref)
#             fT.write('\n')
#
#         counter+=1
#
#     hyp_dir_out = glob.glob('tmp_rouge/*.hyp')
#     ref_dir_out = glob.glob('tmp_rouge/*.ref')
#     hyp_dir_out = sorted(hyp_dir_out)
#     ref_dir_out = sorted(ref_dir_out)
#     return hyp_dir_out, ref_dir_out
#
#
# def get_rouge_pap(hyothesis, references, length_limit=300,remove_stopwords=False,stemmer=False,lcs=True,):
#
#     hyp_pathlist, ref_pathlist = write_summaries_to_file(hyothesis, references)
#
#     path_data = []
#     uttnames = []
#     for i in range(len(hyp_pathlist)):
#         path_data.append([hyp_pathlist[i], [ref_pathlist[i]]])
#         uttnames.append(os.path.splitext(hyp_pathlist[i])[0].split('/')[-1])
#
#     config_text = rouge_papier_v2.util.make_simple_config_text(path_data)
#     config_path = './config'
#     of = open(config_path,'w')
#     of.write(config_text)
#     of.close()
#     uttnames.append('Average')
#     df,avgfs = rouge_papier_v2.compute_rouge(config_path, max_ngram=2, lcs=lcs, remove_stopwords=remove_stopwords,stemmer=stemmer,set_length = False, length=length_limit)
#     df['data_ids'] = pd.Series(np.array(uttnames),index =df.index)
#     avg = df.iloc[-1:].to_dict("records")[0]
#     shutil.rmtree('tmp_rouge/')
#     return avg['rouge-1-f'] * 100, avg['rouge-2-f'] * 100, avg['rouge-L-f'] * 100