import os
import pickle
# from multiprocessing.pool import Pool
from operator import itemgetter

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.multiprocessing import Pool
from tqdm import tqdm

import distributed
from models.reporter_ext import ReportMgr, Statistics
from models.uncertainty_loss import UncertaintyLoss
from others.logging import logger
from prepro.data_builder import check_path_existence
# from utils.rouge_pap import get_rouge_pap
from utils.rouge_score import evaluate_rouge, evaluate_rouge_avg
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, device_id, model, optim):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    grad_accum_count = args.accum_count
    n_gpu = args.world_size

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    print('gpu_rank %d' % gpu_rank)

    tensorboard_log_dir = args.model_path + '/stats/'

    # if not os.path.exists(tensorboard_log_dir) :
    #     os.makedirs(tensorboard_log_dir)
    # else:
    #     os.remove(tensorboard_log_dir)
    #     os.makedirs(tensorboard_log_dir)

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)

    trainer = Trainer(args, model, optim, grad_accum_count, n_gpu, gpu_rank, report_manager)

    # print(tr)
    if (model):
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)

    return trainer


class Trainer(object):
    """
    Class that controls the training process.
    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, args, model, optim,
                 grad_accum_count=1, n_gpu=1, gpu_rank=1,
                 report_manager=None):
        # Basic attributes.
        self.args = args
        self.alpha = args.alpha_mtl
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.is_joint = getattr(self.model, 'is_joint')
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager
        self.valid_trajectories = []
        self.valid_rgls = []
        self.best_val_step = 0
        # self.loss = torch.nn.BCELoss(reduction='none')
        self.rg_predictor = False

        if self.args.rg_predictor:
            self.mse_loss = torch.nn.MSELoss(reduction='none')
            self.rg_predictor = True

        self.rmse_loss = torch.nn.MSELoss(reduction='none')
        # self.loss_sect = torch.nn.CrossEntropyLoss(reduction='none')



        self.min_val_loss = 100000
        self.min_rl = -100000
        self.softmax = nn.Softmax(dim=1)
        self.softmax_acc_pred = nn.Softmax(dim=2)
        self.softmax_sent = nn.Softmax(dim=1)
        assert grad_accum_count > 0
        # Set model in training mode.
        if (model):
            self.model.train()

    def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`
        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):
        Return:
            None
        """
        logger.info('Start training...')

        # step =  self.optim._step + 1
        step = self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0
        sent_num_normalization = 0
        train_iter = train_iter_fct()

        total_stats = Statistics()
        valid_global_stats = Statistics(stat_file_dir=self.args.model_path)
        valid_global_stats.write_stat_header(self.is_joint)
        # report_stats = Statistics(print_traj=self.is_joint)
        report_stats = Statistics(print_traj=self.is_joint)
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:
            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):
                    true_batchs.append(batch)
                    normalization += batch.batch_size
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.n_gpu > 1:
                            normalization = sum(distributed
                                                .all_gather_list
                                                (normalization))

                        self._gradient_accumulation(
                            true_batchs, normalization, total_stats,
                            report_stats)

                        # if step % 120 == 0:
                        #     for name, param in self.model.named_parameters():
                        #         if name == 'sentence_encoder.ext_transformer_layer.transformer_inter.0.self_attn.linear_keys.weight':
                        #             print(param)

                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optim.learning_rate,
                            self.model.uncertainty_loss._sigmas_sq[0].item() if self.is_joint else 0,
                            self.model.uncertainty_loss._sigmas_sq[1].item() if self.is_joint else 0,
                            report_stats)

                        self._report_step(self.optim.learning_rate, step,
                                          self.model.uncertainty_loss._sigmas_sq[0] if self.is_joint else 0,
                                          self.model.uncertainty_loss._sigmas_sq[1] if self.is_joint else 0,
                                          train_stats=report_stats)

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            self._save(step)

                        if step == 60 or step % self.args.val_interval == 0:  # Validation
                        # if step % self.args.val_interval == 0:  # Validation
                            logger.info('----------------------------------------')
                            logger.info('Start evaluating on evaluation set... ')
                            val_stat, best_model_save = self.validate_rouge_baseline(valid_iter_fct, step,
                                                                            valid_gl_stats=valid_global_stats)
                            if best_model_save:
                                self._save(step)
                                logger.info(f'Best model saved sucessfully at step %d' % step)
                                self.best_val_step = step
                            logger.info('----------------------------------------')
                            self.model.train()

                        step += 1
                        if step > train_steps:
                            break
            train_iter = train_iter_fct()

        return total_stats

    def plot_confusion(self, y_pred, y_true):
        cnf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
        np.set_printoptions(precision=2)
        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=['Objective', 'Background', 'Method', 'Results', 'Other'],
                              title='Confusion matrix, without normalization')

    def validate_rouge(self, valid_iter_fct, step=0, valid_gl_stats=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """

        # Set model in validating mode.
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False

        def _insert(list, pred, pos):

            # Searching for the position
            for i in range(len(list)):
                if list[i][1] > pos:
                    break

            # Inserting pred in the list
            list = list[:i] + [(pred, pos)] + list[i:]
            return list

        preds = {}
        preds_with_idx = {}
        golds = {}
        can_path = '%s_step%d.source' % (self.args.result_path, step)
        gold_path = '%s_step%d.target' % (self.args.result_path, step)

        if step == self.best_val_step:
            can_path = '%s_step%d.source' % (self.args.result_path_test, step)
            gold_path = '%s_step%d.target' % (self.args.result_path_test, step)

        save_pred = open(can_path, 'w')
        save_gold = open(gold_path, 'w')
        sent_scores_whole = {}
        sent_sects_whole = {}
        sent_sects_whole_true = {}
        sent_labels_true = {}
        paper_srcs = {}
        paper_tgts = {}
        preds_sects = {}
        preds_sects_true = {}
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()
        best_model_saved = False
        # valid_iter = valid_iter_fct()
        # idx_num = self.args.idx_num
        # counter = 0
        # for _ in valid_iter:
        #     counter += 1

        valid_iter = valid_iter_fct()

        with torch.no_grad():
            for batch in tqdm(valid_iter):
                src = batch.src
                labels = batch.src_sent_labels
                sent_labels = batch.sent_labels

                if self.rg_predictor:
                    sent_true_rg = batch.src_sent_labels
                else:
                    sent_labels = batch.sent_labels
                segs = batch.segs
                clss = batch.clss
                # section_rg = batch.section_rg
                mask = batch.mask_src
                mask_cls = batch.mask_cls
                p_id = batch.paper_id
                segment_src = batch.src_str
                paper_tgt = batch.tgt_str

                sent_sect_labels = batch.sent_sect_labels
                if self.is_joint:
                    if not self.rg_predictor:
                        sent_scores, sent_sect_scores, mask, loss, loss_sent, loss_sect = self.model(src, segs, clss, mask, mask_cls, sent_labels, sent_sect_labels)
                    else:
                        sent_scores, sent_sect_scores, mask, loss, loss_sent, loss_sect = self.model(src, segs, clss, mask, mask_cls, sent_true_rg, sent_sect_labels)
                    acc, _ = self._get_mertrics(sent_sect_scores, sent_sect_labels, mask=mask,
                                                task='sent_sect')

                    batch_stats = Statistics(loss=float(loss.cpu().data.numpy().sum()),
                                             loss_sect=float(loss_sect.cpu().data.numpy().sum()),
                                             loss_sent=float(loss_sent.cpu().data.numpy().sum()),
                                             n_docs=len(labels),
                                             n_acc=batch.batch_size,
                                             RMSE=self._get_mertrics(sent_scores, labels, mask=mask, task='sent'),
                                             accuracy=acc)

                else:
                    if not self.rg_predictor:
                        sent_scores, mask, loss, _, _ = self.model(src, segs, clss, mask, mask_cls, sent_labels, sent_sect_labels=None, is_inference=True)
                    else:
                        sent_scores, mask, loss, _, _ = self.model(src, segs, clss, mask, mask_cls, sent_true_rg, sent_sect_labels=None, is_inference=True)

                    # sent_scores = (section_rg.unsqueeze(1).expand_as(sent_scores).to(device='cuda')*100) * sent_scores


                    batch_stats = Statistics(loss=float(loss.cpu().data.numpy().sum()),
                                             RMSE=self._get_mertrics(sent_scores, labels, mask=mask, task='sent'),
                                             n_acc=batch.batch_size,
                                             n_docs=len(labels))

                stats.update(batch_stats)

                sent_scores = sent_scores + mask.float()
                sent_scores = sent_scores.cpu().data.numpy()

                for idx, p_id in enumerate(p_id):
                    p_id = p_id.split('___')[0]

                    if p_id not in sent_scores_whole.keys():
                        masked_scores = sent_scores[idx] * mask[idx].cpu().data.numpy()
                        masked_scores = masked_scores[masked_scores.nonzero()]


                        masked_sent_labels_true = (sent_labels[idx] + 1) * mask[idx].long()
                        masked_sent_labels_true = masked_sent_labels_true[masked_sent_labels_true.nonzero()].flatten()
                        masked_sent_labels_true = (masked_sent_labels_true - 1)

                        sent_scores_whole[p_id] = masked_scores
                        sent_labels_true[p_id] = masked_sent_labels_true.cpu()

                        masked_scores_true = (sent_sect_labels[idx] + 1) * mask[idx].long()
                        masked_scores_true = masked_scores_true[masked_scores_true.nonzero()].flatten()
                        masked_scores_true = (masked_scores_true - 1)
                        sent_sects_whole_true[p_id] = masked_scores_true.cpu()

                        if self.is_joint:
                            masked_scores_sects = sent_sect_scores[idx] * mask[idx].view(-1,1).expand_as(sent_sect_scores[idx]).float()
                            masked_scores_sects = masked_scores_sects[torch.abs(masked_scores_sects).sum(dim=1) != 0]
                            sent_sects_whole[p_id] = torch.max(self.softmax(masked_scores_sects), 1)[1].cpu()


                        paper_srcs[p_id] = segment_src[idx]
                        paper_tgts[p_id] = paper_tgt[idx]
                    else:
                        masked_scores = sent_scores[idx] * mask[idx].cpu().data.numpy()
                        masked_scores = masked_scores[masked_scores.nonzero()]

                        masked_sent_labels_true = (sent_labels[idx] + 1) * mask[idx].long()
                        masked_sent_labels_true = masked_sent_labels_true[masked_sent_labels_true.nonzero()].flatten()
                        masked_sent_labels_true = (masked_sent_labels_true - 1)

                        sent_scores_whole[p_id] = np.concatenate((sent_scores_whole[p_id], masked_scores), 0)
                        sent_labels_true[p_id] = np.concatenate((sent_labels_true[p_id], masked_sent_labels_true.cpu()),
                                                                0)

                        masked_scores_true = (sent_sect_labels[idx] + 1) * mask[idx].long()
                        masked_scores_true = masked_scores_true[masked_scores_true.nonzero()].flatten()
                        masked_scores_true = (masked_scores_true - 1)
                        sent_sects_whole_true[p_id] = np.concatenate(
                            (sent_sects_whole_true[p_id], masked_scores_true.cpu()), 0)

                        if self.is_joint:
                            masked_scores_sects = sent_sect_scores[idx] * mask[idx].view(-1, 1).expand_as(
                                sent_sect_scores[idx]).float()
                            masked_scores_sects = masked_scores_sects[
                                torch.abs(masked_scores_sects).sum(dim=1) != 0]
                            sent_sects_whole[p_id] = np.concatenate((sent_sects_whole[p_id], torch.max(self.softmax(masked_scores_sects), 1)[1].cpu()), 0)

                        paper_srcs[p_id] = np.concatenate((paper_srcs[p_id], segment_src[idx]), 0)

        PRED_LEN = self.args.val_pred_len
        # LENGTH_LIMIT= 100
        for p_id, sent_scores in sent_scores_whole.items():
            selected_ids_unsorted = np.argsort(-sent_scores, 0)
            try:
                selected_ids_unsorted = [s for s in selected_ids_unsorted if
                                         len(paper_srcs[p_id][s].replace('.', '').replace(',', '').replace('(', '').replace(')', '').
                                             replace('-', '').replace(':','').replace(';','').replace('*','').split()) > 5 and
                                         len(paper_srcs[p_id][s].replace('.', '').replace(',', '').replace('(', '').replace(')', '').
                                             replace('-', '').replace(':','').replace(';','').replace('*','').split()) < 100]
            except:
                continue
            selected_ids_top = selected_ids_unsorted[:PRED_LEN]
            selected_ids = sorted(selected_ids_top, reverse=False)
            _pred_final = [(paper_srcs[p_id][selected_ids_unsorted[0]].strip(),selected_ids_unsorted[0])]
            try:
                if self.is_joint:
                    summary_sect_pred = [sent_sects_whole[p_id][selected_ids_unsorted[0]]]

                # summary_sect_true = [sent_sects_whole_true[p_id][selected_ids_unsorted[0]]]
            except:
                import pdb;
                pdb.set_trace()

                # for j in selected_ids[:len(paper_srcs[paper_id][0])]:
            #     if (j >= len(paper_srcs[paper_id][0])):
            #         continue
            #     candidate = paper_srcs[paper_id][0][j].strip()
            #     if (self.args.block_trigram):
            #         if (not _block_tri(candidate, _pred_final)):
            #             _pred_final.append(candidate)
            #     else:
            #         _pred_final.append(candidate)
            #
            #     if (len(_pred_final) == 60):
            #         break

            picked_up = 1
            picked_up_word_count = len(paper_srcs[p_id][selected_ids_unsorted[0]].strip().split())
            for j in selected_ids[1:len(paper_srcs[p_id])]:
                if (j >= len(paper_srcs[p_id][0])):
                    continue
                candidate = paper_srcs[p_id][j].strip()
                if (self.args.block_trigram):
                    if (not _block_tri(candidate, [x[0] for x in _pred_final])):
                        _pred_final = _insert(_pred_final, candidate, j)
                        # if self.is_joint:
                        #     summary_sect_pred.append(sent_sects_whole[p_id][j])
                        #
                        # try:
                        #     summary_sect_true.append(sent_sects_whole_true[p_id][j])
                        # except:
                        #     import pdb;pdb.set_trace()
                        picked_up += 1
                        picked_up_word_count += len(candidate.split())
                else:
                    _pred_final.append((candidate, j))
                    # if self.is_joint:
                    #     summary_sect_pred.append(sent_sects_whole[p_id][j])
                    #     summary_sect_true.append(sent_sects_whole_true[p_id][j])
                    picked_up += 1
                    picked_up_word_count += len(candidate.split())

                # if (picked_up_word_count >= LENGTH_LIMIT):
                #     break
                if (picked_up == PRED_LEN):
                    break

            _pred_final = sorted(_pred_final, key=itemgetter(1))


            if picked_up < PRED_LEN:
            # if picked_up_word_count < LENGTH_LIMIT:
            #     # it means that some sentences are just skept, sample from the rest of the list
                selected_ids_rest = selected_ids_unsorted[PRED_LEN:]

                for k in selected_ids_rest:
                    candidate = paper_srcs[p_id][k].strip()
                    if (self.args.block_trigram):
                        if (not _block_tri(candidate, [x[0] for x in _pred_final])):
                            _pred_final = _insert(_pred_final, candidate, k)
                            # if self.is_joint:
                            #     summary_sect_pred.append(sent_sects_whole[p_id][k])
                            # summary_sect_true.append(sent_sects_whole_true[p_id][k])
                            picked_up += 1
                            picked_up_word_count += len(candidate.split())

                    else:
                        _pred_final.append((candidate, k))
                        # if self.is_joint:
                        #     summary_sect_pred.append(sent_sects_whole[p_id][k])
                        # summary_sect_true.append(sent_sects_whole_true[p_id][k])
                        picked_up += 1
                        picked_up_word_count += len(candidate.split())

                    if (picked_up == PRED_LEN):
                    # if (picked_up_word_count >= LENGTH_LIMIT):
                        break

            _pred_final = sorted(_pred_final, key=itemgetter(1))

            _pred_final_str = '<q>'.join([x[0] for x in _pred_final])
            # if (self.args.recall_eval):
            #     _pred = ' '.join(_pred_final_str.split()[:len(paper_tgts[p_id].split())])

            preds[p_id] = _pred_final_str
            golds[p_id] = paper_tgts[p_id]
            preds_with_idx[p_id] = _pred_final
            # if self.is_joint:
            #     preds_sects[p_id] = summary_sect_pred
            # preds_sects_true[p_id] = summary_sect_true

        for id, pred in preds.items():
            save_pred.write(pred.strip().replace('<q>', ' ') + '\n')
            save_gold.write(golds[id].replace('<q>', ' ').strip() + '\n')

        p = _get_ir_eval_metrics(preds_with_idx, sent_labels_true, PRED_LEN)
        logger.info("Prection: {:4.4f}".format(p))

        print(f'Gold: {gold_path}')
        print(f'Prediction: {can_path}')


        r1, r2, rl = self._report_rouge(preds.values(), golds.values())
        stats.set_rl(r1, r2, rl)
        stats.set_ir_metrics(p)
        self.valid_rgls.append(r2)
        self._report_step(0, step,
                          self.model.uncertainty_loss._sigmas_sq[0] if self.is_joint else 0,
                          self.model.uncertainty_loss._sigmas_sq[1] if self.is_joint else 0,
                          valid_stats=stats)

        if len(self.valid_rgls) > 0:
            if self.min_rl < self.valid_rgls[-1]:
                self.min_rl = self.valid_rgls[-1]
                best_model_saved = True

        return stats, best_model_saved

    def validate_rouge_mmr(self, valid_iter_fct, step=0, valid_gl_stats=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """

        # Set model in validating mode.
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False

        def _insert(list, pred, pos):

            # Searching for the position
            for i in range(len(list)):
                if list[i][1] > pos:
                    break

            # Inserting pred in the list
            list = list[:i] + [(pred, pos)] + list[i:]
            return list

        preds = {}
        preds_with_idx = {}
        golds = {}
        can_path = '%s_step%d.source' % (self.args.result_path, step)
        gold_path = '%s_step%d.target' % (self.args.result_path, step)

        if step == self.best_val_step:
            can_path = '%s_step%d.source' % (self.args.result_path_test, step)
            gold_path = '%s_step%d.target' % (self.args.result_path_test, step)

        save_pred = open(can_path, 'w')
        save_gold = open(gold_path, 'w')
        sent_scores_whole = {}
        sent_sects_whole = {}
        sent_sects_whole_true = {}
        sent_labels_true = {}
        sent_sect_wise_rg_whole = {}
        paper_srcs = {}
        sent_sections_txt_whole = {}
        paper_tgts = {}
        preds_sects = {}
        preds_sects_true = {}
        source_sent_encodings = {}
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()
        best_model_saved = False
        # valid_iter = valid_iter_fct()
        # idx_num = self.args.idx_num
        # counter = 0
        # for _ in valid_iter:
        #     counter += 1

        valid_iter = valid_iter_fct()

        with torch.no_grad():
            for batch in tqdm(valid_iter):
                src = batch.src
                labels = batch.src_sent_labels
                if self.rg_predictor:
                    sent_true_rg = batch.src_sent_labels
                else:
                    sent_labels = batch.sent_labels
                segs = batch.segs
                clss = batch.clss
                # section_rg = batch.section_rg
                mask = batch.mask_src
                mask_cls = batch.mask_cls
                p_ids = batch.paper_id
                segment_src = batch.src_str
                paper_tgt = batch.tgt_str
                sent_sections_txt = batch.sent_sections_txt
                sent_sect_wise_rg = batch.sent_sect_wise_rg

                sent_sect_labels = batch.sent_sect_labels
                if self.is_joint:
                    sent_scores, sent_sect_scores, mask, loss, loss_sent, loss_sect = self.model(src, segs, clss, mask, mask_cls, sent_labels, sent_sect_labels)


                    acc, _ = self._get_mertrics(sent_sect_scores, sent_sect_labels, mask=mask,
                                                task='sent_sect')

                    batch_stats = Statistics(loss=float(loss.cpu().data.numpy().sum()),
                                             loss_sect=float(loss_sect.cpu().data.numpy().sum()),
                                             loss_sent=float(loss_sent.cpu().data.numpy().sum()),
                                             n_docs=len(labels),
                                             n_acc=batch.batch_size,
                                             RMSE=self._get_mertrics(sent_scores, labels, mask=mask, task='sent'),
                                             accuracy=acc)

                else:
                    sent_scores, mask, loss, _, _, sent_encodings = self.model(src, segs, clss, mask, mask_cls, sent_labels, sent_sect_labels=None, is_inference=True, return_encodings=True)
                    # sent_scores = (section_rg.unsqueeze(1).expand_as(sent_scores).to(device='cuda')*100) * sent_scores


                    batch_stats = Statistics(loss=float(loss.cpu().data.numpy().sum()),
                                             RMSE=self._get_mertrics(sent_scores, labels, mask=mask, task='sent'),
                                             n_acc=batch.batch_size,
                                             n_docs=len(labels))

                stats.update(batch_stats)

                sent_scores = sent_scores + mask.float()
                sent_scores = sent_scores.cpu().data.numpy()

                for idx, p_id in enumerate(p_ids):
                    p_id = p_id.split('___')[0]

                    if p_id not in sent_scores_whole.keys():
                        masked_scores = sent_scores[idx] * mask[idx].cpu().data.numpy()
                        masked_scores = masked_scores[masked_scores.nonzero()]

                        masked_paper_sent_encodings = (sent_encodings[idx]) * mask[idx].long().unsqueeze(1).repeat(1, 768)
                        masked_paper_sent_encodings = masked_paper_sent_encodings[masked_paper_sent_encodings.abs().sum(dim=1)>0]
                        source_sent_encodings[p_id] = masked_paper_sent_encodings.cpu()

                        masked_sent_labels_true = (sent_labels[idx] + 1) * mask[idx].long()
                        masked_sent_labels_true = masked_sent_labels_true[masked_sent_labels_true.nonzero()].flatten()
                        masked_sent_labels_true = (masked_sent_labels_true - 1)

                        sent_scores_whole[p_id] = masked_scores
                        # source_sent_encodings[p_id] = sent_encodings.cpu()
                        sent_labels_true[p_id] = masked_sent_labels_true.cpu()

                        masked_scores_true = (sent_sect_labels[idx] + 1) * mask[idx].long()
                        masked_scores_true = masked_scores_true[masked_scores_true.nonzero()].flatten()
                        masked_scores_true = (masked_scores_true - 1)
                        sent_sects_whole_true[p_id] = masked_scores_true.cpu()

                        if self.is_joint:
                            masked_scores_sects = sent_sect_scores[idx] * mask[idx].view(-1,1).expand_as(sent_sect_scores[idx]).float()
                            masked_scores_sects = masked_scores_sects[torch.abs(masked_scores_sects).sum(dim=1) != 0]



                            sent_sects_whole[p_id] = torch.max(self.softmax(masked_scores_sects), 1)[1].cpu()


                        paper_srcs[p_id] = segment_src[idx]
                        sent_sect_wise_rg_whole[p_id] = sent_sect_wise_rg[idx]
                        sent_sections_txt_whole[p_id] = sent_sections_txt[idx]
                        paper_tgts[p_id] = paper_tgt[idx]
                    else:
                        masked_paper_sent_encodings = (sent_encodings[idx]) * mask[idx].long().unsqueeze(1).repeat(1, 768)
                        masked_paper_sent_encodings = masked_paper_sent_encodings[masked_paper_sent_encodings.abs().sum(dim=1)>0]
                        source_sent_encodings[p_id] = np.concatenate((source_sent_encodings[p_id], masked_paper_sent_encodings.cpu()), 0)

                        masked_scores = sent_scores[idx] * mask[idx].cpu().data.numpy()
                        masked_scores = masked_scores[masked_scores.nonzero()]

                        masked_sent_labels_true = (sent_labels[idx] + 1) * mask[idx].long()
                        masked_sent_labels_true = masked_sent_labels_true[masked_sent_labels_true.nonzero()].flatten()
                        masked_sent_labels_true = (masked_sent_labels_true - 1)

                        masked_sent_labels_true = (sent_labels[idx] + 1) * mask[idx].long()
                        masked_sent_labels_true = masked_sent_labels_true[masked_sent_labels_true.nonzero()].flatten()
                        masked_sent_labels_true = (masked_sent_labels_true - 1)

                        sent_scores_whole[p_id] = np.concatenate((sent_scores_whole[p_id], masked_scores), 0)
                        sent_labels_true[p_id] = np.concatenate((sent_labels_true[p_id], masked_sent_labels_true.cpu()),
                                                                0)

                        masked_scores_true = (sent_sect_labels[idx] + 1) * mask[idx].long()
                        masked_scores_true = masked_scores_true[masked_scores_true.nonzero()].flatten()
                        masked_scores_true = (masked_scores_true - 1)
                        sent_sects_whole_true[p_id] = np.concatenate(
                            (sent_sects_whole_true[p_id], masked_scores_true.cpu()), 0)

                        if self.is_joint:
                            masked_scores_sects = sent_sect_scores[idx] * mask[idx].view(-1, 1).expand_as(
                                sent_sect_scores[idx]).float()
                            masked_scores_sects = masked_scores_sects[
                                torch.abs(masked_scores_sects).sum(dim=1) != 0]
                            sent_sects_whole[p_id] = np.concatenate((sent_sects_whole[p_id], torch.max(self.softmax(masked_scores_sects), 1)[1].cpu()), 0)

                        paper_srcs[p_id] = np.concatenate((paper_srcs[p_id], segment_src[idx]), 0)
                        sent_sect_wise_rg_whole[p_id] = np.concatenate((sent_sect_wise_rg_whole[p_id], sent_sect_wise_rg[idx]), 0)
                        sent_sections_txt_whole[p_id] = np.concatenate((sent_sections_txt_whole[p_id], sent_sections_txt[idx]), 0)


        ## mmr-baed

        saved_dict = {}

        for p_id, sent_scores in tqdm(sent_scores_whole.items(), total=len(sent_scores_whole)):
            saved_dict[p_id] = (p_id, sent_scores_whole[p_id], paper_srcs[p_id], paper_tgts[p_id], sent_sects_whole_true[p_id], source_sent_encodings[p_id], sent_sects_whole_true[p_id],sent_sections_txt_whole[p_id], sent_labels_true[p_id], sent_sect_wise_rg_whole[p_id])

        pickle.dump(saved_dict, open("save_list_lsum_" + self.args.exp_set + ".p", "wb"))

        # pool = Pool(9)
        # preds = {}
        # golds = {}
        #
        # for d in tqdm(pool.imap(self._multi_mmr, a_lst), total=len(a_lst)):
        #     p_id = d[1]
        #
        #     preds[d[1]] = d[0]
        #     golds[d[1]] = paper_tgts[p_id]
        #
        # pool.close()
        # pool.join()
        #
        # for id, pred in preds.items():
        #     save_pred.write(' '.join(pred).strip().replace('<q>', ' ') + '\n')
        #     save_gold.write(golds[id].replace('<q>', ' ').strip() + '\n')
        #
        # # p = _get_ir_eval_metrics(preds_with_idx, sent_labels_true)
        # # logger.info("Prection: {:4.4f}".format(p))
        #
        # print(f'Gold: {gold_path}')
        # print(f'Prediction: {can_path}')
        #
        # r1, r2, rl = self._report_rouge([' '.join(p) for p in preds.values()], golds.values())
        # stats.set_rl(r1, r2, rl)
        # # stats.set_ir_metrics(p)
        # self.valid_rgls.append(r2)
        # self._report_step(0, step,
        #                   self.model.uncertainty_loss._sigmas_sq[0] if self.is_joint else 0,
        #                   self.model.uncertainty_loss._sigmas_sq[1] if self.is_joint else 0,
        #                   valid_stats=stats)
        #
        # if len(self.valid_rgls) > 0:
        #     if self.min_rl < self.valid_rgls[-1]:
        #         self.min_rl = self.valid_rgls[-1]
        #         best_model_saved = True
        #
        # return stats, best_model_saved


    def validate_rouge_baseline(self, valid_iter_fct, step=0, valid_gl_stats=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """

        # Set model in validating mode.
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False

        def _insert(list, pred, pos):

            # Searching for the position
            for i in range(len(list)):
                if list[i][1] > pos:
                    break

            # Inserting pred in the list
            list = list[:i] + [(pred, pos)] + list[i:]
            return list

        preds = {}
        preds_with_idx = {}
        golds = {}
        can_path = '%s_step%d.source' % (self.args.result_path, step)
        gold_path = '%s_step%d.target' % (self.args.result_path, step)

        if step == self.best_val_step:
            can_path = '%s_step%d.source' % (self.args.result_path_test, step)
            gold_path = '%s_step%d.target' % (self.args.result_path_test, step)

        save_pred = open(can_path, 'w')
        save_gold = open(gold_path, 'w')
        sent_scores_whole = {}
        sent_sects_whole = {}
        sent_sects_whole_true = {}
        sent_labels_true = {}
        paper_srcs = {}
        paper_tgts = {}
        preds_sects = {}
        preds_sects_true = {}
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()
        best_model_saved = False
        # valid_iter = valid_iter_fct()
        # idx_num = self.args.idx_num
        # counter = 0
        # for _ in valid_iter:
        #     counter += 1

        valid_iter = valid_iter_fct()

        with torch.no_grad():
            for batch in tqdm(valid_iter):
                src = batch.src
                labels = batch.src_sent_labels
                sent_labels = batch.sent_labels

                if self.rg_predictor:
                    sent_true_rg = batch.src_sent_labels
                else:
                    sent_labels = batch.sent_labels
                segs = batch.segs
                clss = batch.clss
                # section_rg = batch.section_rg
                mask = batch.mask_src
                mask_cls = batch.mask_cls
                p_id = batch.paper_id
                segment_src = batch.src_str
                paper_tgt = batch.tgt_str

                sent_sect_labels = batch.sent_sect_labels
                if self.is_joint:
                    if not self.rg_predictor:
                        sent_scores, sent_sect_scores, mask, loss, loss_sent, loss_sect = self.model(src, segs, clss, mask, mask_cls, sent_labels, sent_sect_labels)
                    else:
                        sent_scores, sent_sect_scores, mask, loss, loss_sent, loss_sect = self.model(src, segs, clss, mask, mask_cls, sent_true_rg, sent_sect_labels)
                    acc, _ = self._get_mertrics(sent_sect_scores, sent_sect_labels, mask=mask,
                                                task='sent_sect')

                    batch_stats = Statistics(loss=float(loss.cpu().data.numpy().sum()),
                                             loss_sect=float(loss_sect.cpu().data.numpy().sum()),
                                             loss_sent=float(loss_sent.cpu().data.numpy().sum()),
                                             n_docs=len(labels),
                                             n_acc=batch.batch_size,
                                             RMSE=self._get_mertrics(sent_scores, labels, mask=mask, task='sent'),
                                             accuracy=acc)

                else:
                    if not self.rg_predictor:
                        sent_scores, mask, loss, _, _ = self.model(src, segs, clss, mask, mask_cls, sent_labels, sent_sect_labels=None, is_inference=True)
                    else:
                        sent_scores, mask, loss, _, _ = self.model(src, segs, clss, mask, mask_cls, sent_true_rg, sent_sect_labels=None, is_inference=True)

                    # sent_scores = (section_rg.unsqueeze(1).expand_as(sent_scores).to(device='cuda')*100) * sent_scores


                    batch_stats = Statistics(loss=float(loss.cpu().data.numpy().sum()),
                                             RMSE=self._get_mertrics(sent_scores, labels, mask=mask, task='sent'),
                                             n_acc=batch.batch_size,
                                             n_docs=len(labels))

                stats.update(batch_stats)

                sent_scores = sent_scores + mask.float()
                sent_scores = sent_scores.cpu().data.numpy()

                for idx, p_id in enumerate(p_id):
                    p_id = p_id.split('___')[0]

                    if p_id not in sent_scores_whole.keys():
                        masked_scores = sent_scores[idx] * mask[idx].cpu().data.numpy()
                        masked_scores = masked_scores[masked_scores.nonzero()]


                        masked_sent_labels_true = (sent_labels[idx] + 1) * mask[idx].long()
                        masked_sent_labels_true = masked_sent_labels_true[masked_sent_labels_true.nonzero()].flatten()
                        masked_sent_labels_true = (masked_sent_labels_true - 1)

                        sent_scores_whole[p_id] = masked_scores
                        sent_labels_true[p_id] = masked_sent_labels_true.cpu()

                        masked_scores_true = (sent_sect_labels[idx] + 1) * mask[idx].long()
                        masked_scores_true = masked_scores_true[masked_scores_true.nonzero()].flatten()
                        masked_scores_true = (masked_scores_true - 1)
                        sent_sects_whole_true[p_id] = masked_scores_true.cpu()

                        if self.is_joint:
                            masked_scores_sects = sent_sect_scores[idx] * mask[idx].view(-1,1).expand_as(sent_sect_scores[idx]).float()
                            masked_scores_sects = masked_scores_sects[torch.abs(masked_scores_sects).sum(dim=1) != 0]
                            sent_sects_whole[p_id] = torch.max(self.softmax(masked_scores_sects), 1)[1].cpu()


                        paper_srcs[p_id] = segment_src[idx]
                        paper_tgts[p_id] = paper_tgt[idx]
                    else:
                        masked_scores = sent_scores[idx] * mask[idx].cpu().data.numpy()
                        masked_scores = masked_scores[masked_scores.nonzero()]

                        masked_sent_labels_true = (sent_labels[idx] + 1) * mask[idx].long()
                        masked_sent_labels_true = masked_sent_labels_true[masked_sent_labels_true.nonzero()].flatten()
                        masked_sent_labels_true = (masked_sent_labels_true - 1)

                        sent_scores_whole[p_id] = np.concatenate((sent_scores_whole[p_id], masked_scores), 0)
                        sent_labels_true[p_id] = np.concatenate((sent_labels_true[p_id], masked_sent_labels_true.cpu()),
                                                                0)

                        masked_scores_true = (sent_sect_labels[idx] + 1) * mask[idx].long()
                        masked_scores_true = masked_scores_true[masked_scores_true.nonzero()].flatten()
                        masked_scores_true = (masked_scores_true - 1)
                        sent_sects_whole_true[p_id] = np.concatenate(
                            (sent_sects_whole_true[p_id], masked_scores_true.cpu()), 0)

                        if self.is_joint:
                            masked_scores_sects = sent_sect_scores[idx] * mask[idx].view(-1, 1).expand_as(
                                sent_sect_scores[idx]).float()
                            masked_scores_sects = masked_scores_sects[
                                torch.abs(masked_scores_sects).sum(dim=1) != 0]
                            sent_sects_whole[p_id] = np.concatenate((sent_sects_whole[p_id], torch.max(self.softmax(masked_scores_sects), 1)[1].cpu()), 0)

                        paper_srcs[p_id] = np.concatenate((paper_srcs[p_id], segment_src[idx]), 0)

        PRED_LEN = self.args.val_pred_len
        # LENGTH_LIMIT= 100
        for p_id, sent_scores in sent_scores_whole.items():
            selected_ids_unsorted = np.argsort(-sent_scores, 0)
            selected_ids_unsorted = [s for s in selected_ids_unsorted if
                                     len(paper_srcs[p_id][s].replace('.', '').replace(',', '').replace('(', '').replace(')', '').
                                         replace('-', '').replace(':','').replace(';','').replace('*','').split()) > 5 and
                                     len(paper_srcs[p_id][s].replace('.', '').replace(',', '').replace('(', '').replace(')', '').
                                         replace('-', '').replace(':','').replace(';','').replace('*','').split()) < 100]

            _pred = []
            for j in selected_ids_unsorted[1:len(paper_srcs)]:
                if (j >= len(paper_srcs)):
                    continue
                candidate = paper_srcs[p_id][j].strip()
                if True:
                    # if (not _block_tri(candidate, _pred)):
                    _pred.append((candidate, j))

                if (len(_pred) == PRED_LEN):
                    break

            _pred_final_str = '<q>'.join([x[0] for x in _pred])

            preds[p_id] = _pred_final_str
            golds[p_id] = paper_tgts[p_id]
            preds_with_idx[p_id] = _pred

        for id, pred in preds.items():
            save_pred.write(pred.strip().replace('<q>', ' ') + '\n')
            save_gold.write(golds[id].replace('<q>', ' ').strip() + '\n')

        p = _get_ir_eval_metrics(preds_with_idx, sent_labels_true, n=PRED_LEN)
        logger.info("Prection: {:4.4f}".format(p))

        print(f'Gold: {gold_path}')
        print(f'Prediction: {can_path}')


        r1, r2, rl = self._report_rouge(preds.values(), golds.values())
        stats.set_rl(r1, r2, rl)
        stats.set_ir_metrics(p)
        self.valid_rgls.append(r2)
        self._report_step(0, step,
                          self.model.uncertainty_loss._sigmas_sq[0] if self.is_joint else 0,
                          self.model.uncertainty_loss._sigmas_sq[1] if self.is_joint else 0,
                          valid_stats=stats)

        if len(self.valid_rgls) > 0:
            if self.min_rl < self.valid_rgls[-1]:
                self.min_rl = self.valid_rgls[-1]
                best_model_saved = True

        return stats, best_model_saved

    def _bertsum_baseline_validate(params):
        # Set model in validating mode.
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False

        p_id, sent_scores, paper_srcs, paper_tgt, sent_sects_whole_true, source_sent_encodings, \
        sent_sects_whole_true, section_textual, _, _, co1, co2, co3, cos = params
        # LENGTH_LIMIT= 100
        # print(sent_scores)
        sent_scores = np.asarray([s - 1.00 for s in sent_scores])

        selected_ids_unsorted = np.argsort(-sent_scores, 0)

        try:
            selected_ids_unsorted = [s for s in selected_ids_unsorted if
                                     len(paper_srcs[s].replace('.', '').replace(',', '').replace('(', '').replace(
                                         ')', '').
                                         replace('-', '').replace(':', '').replace(';', '').replace('*',
                                                                                                    '').split()) > 5 and
                                     len(paper_srcs[s].replace('.', '').replace(',', '').replace('(', '').replace(
                                         ')', '').
                                         replace('-', '').replace(':', '').replace(';', '').replace('*',
                                                                                                    '').split()) < 100]
            _pred = []
            for j in selected_ids_unsorted[1:len(paper_srcs)]:
                if (j >= len(paper_srcs)):
                    continue
                candidate = paper_srcs[p_id][j].strip()
                if True:
                    # if (not _block_tri(candidate, _pred)):
                    _pred.append(candidate)
                if (len(_pred) == 10):
                    break

            return _pred, paper_tgt, p_id
        except:

            return None


    def _multi_mmr(self, params):
        p_id, sent_scores, paper_srcs, sent_sects_whole_true, source_sent_encodings, \
        sent_sects_whole_true = params

        PRED_LEN = 10
        # LENGTH_LIMIT= 100
        top_score_ids = np.argsort(-sent_scores, 0)
        summary = []
        summary_idx = []
        summary_sects = []
        top_score_ids = [s for s in top_score_ids if len(
            paper_srcs[s].replace('.', '').replace(',', '').replace('(', '').replace(')', '').replace('-',
                                                                                                            '').replace(
                ':', '').replace(';', '').replace('*', '').split()) > 5 and len(
            paper_srcs[s].replace('.', '').replace(',', '').replace('(', '').replace(')', '').replace('-',
                                                                                                            '').replace(
                ':', '').replace(';', '').replace('*', '').split()) < 100]

        source_sents = paper_srcs

        # pick the first top-score sentence to start with...
        summary_sects += [sent_sects_whole_true[top_score_ids[0]]]
        summary += [paper_srcs[top_score_ids[0]]]
        # remove the first sentence from the search space
        source_sents = [s for idx, s in enumerate(source_sents) if idx in top_score_ids]
        sent_scores_model = [sc for idx, sc in enumerate(sent_scores) if idx in top_score_ids]
        source_sent_encodings = source_sent_encodings[top_score_ids]

        sent_sects_whole_true = [sc for idx, sc in enumerate(sent_sects_whole_true) if
                                       idx in top_score_ids]

        summary_idx = [source_sents.index(paper_srcs[top_score_ids[0]])]
        # augment the summary with MMR until the pred length reach.
        for summary_num in range(1, PRED_LEN):
            MMRs_score = self.cal_mmr(source_sents, summary, summary_idx, source_sent_encodings[p_id],
                                      summary_sects, sent_scores_model, sent_sects_whole_true[p_id])
            sent_scores = np.multiply(sent_scores_model, MMRs_score)
            # autment summary with the updated sent scores

            top_score_ids = np.argsort(-sent_scores, 0)
            summary_idx += [top_score_ids[0]]
            summary += [source_sents[top_score_ids[0]]]
            summary_sects += [sent_sects_whole_true[top_score_ids[0]]]

        summary = [s[1] for s in sorted(zip(summary_idx, summary), key=lambda x: x[0])]

        return summary, p_id


    def get_cosine_sim(self, sents_encodings, idx):
        sents_encodings = torch.from_numpy(sents_encodings)
        sentence_vector = sents_encodings[idx]

        document_vector = torch.mean(sents_encodings, dim=0)
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        sim = cos(sentence_vector, document_vector)
        return sim.item()

    def cal_mmr(self, source_sents, partial_summary, partial_summary_idx, sentence_encodings, partial_summary_sects, sent_scores, sent_sects_whole_true):
        current_mmrs = []
        # MMR formula:
        ## MMR = argmax (\alpha Sim(si, D) - \beta max SimSent(si, sj) - \theta max SimSect(sj, sj))
        for idx, sent in enumerate(source_sents):
            if idx in partial_summary_idx:
                current_mmrs.append(-10000)
                continue

            sent_txt = sent
            sent_section = sent_sects_whole_true[idx]


            ## calculate first term
            first_subterm1 = sent_scores[idx]
            first_subterm2 = self.get_cosine_sim(sentence_encodings, idx)
            first_term = (.5 * first_subterm1) + (.5 * first_subterm2)

            ## calculate second term
            max_rg_score = 0
            for sent in partial_summary:
                rg_score = evaluate_rouge([sent], [sent_txt], type='p')[2]
                if rg_score > max_rg_score:
                    max_rg_score = rg_score
            second_term = max_rg_score

            ## calculate third term
            partial_summary_sects_counter = {}
            for sect in partial_summary_sects:
                if sect not in partial_summary_sects_counter:
                    partial_summary_sects_counter[sect] = 1
                else:
                    partial_summary_sects_counter[sect] += 1

            for sect in partial_summary_sects_counter:
                if sect == sent_section:
                    partial_summary_sects_counter[sect] = (partial_summary_sects_counter[sect] + 1) / 10
                else:
                    partial_summary_sects_counter[sect] = partial_summary_sects_counter[sect] / 10


            third_term = max(partial_summary_sects_counter.values())

            # if len(partial_summary) == 9:
            #     import pdb;pdb.set_trace()

            mmr_sent = .98* first_term - .01 * second_term - .05 * third_term
            current_mmrs.append(mmr_sent)

        return current_mmrs

    def _gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats):

        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            sent_rg_scores = batch.src_sent_labels

            sent_sect_labels = batch.sent_sect_labels
            sent_bin_labels = batch.sent_labels
            # if self.rg_predictor:
            sent_rg_true_scores = batch.src_sent_labels
            segs = batch.segs
            clss = batch.clss
            mask = batch.mask_src
            mask_cls = batch.mask_cls
            src_str = batch.src_str
            paper_ids = batch.paper_id


            if self.is_joint:
                if not self.rg_predictor:
                    sent_scores, sent_sect_scores, mask, loss, loss_sent, loss_sect = self.model(src, segs, clss, mask, mask_cls, sent_bin_labels, sent_sect_labels)
                else:
                    sent_scores, sent_sect_scores, mask, loss, loss_sent, loss_sect = self.model(src, segs, clss, mask, mask_cls, sent_rg_scores, sent_sect_labels)
                # loss_sent = self.loss(sent_scores, sent_rg_scores.float())
                #
                # loss_sent = self.bin_loss(sent_scores, sent_bin_labels.float())
                # loss_sent = (loss_sent * mask.float()).sum(dim=1)
                # loss_sent = ((loss_sent)/(mask.sum(dim=1).float())).sum()

                # if self.rg_predictor:
                #     loss_sent = self.mse_loss(sent_scores, sent_rg_true_scores.float())
                #     loss_sent = (loss_sent * mask.float()).sum(dim=1)
                #     loss_sent = (loss_sent / mask.sum(dim=1).float()).sum()
                # else:
                #     loss_sent = self.bin_loss(sent_scores, sent_bin_labels.float())
                #     loss_sent = (loss_sent * mask.float()).sum()
                #     loss_sent = (loss_sent / loss_sent.numel())

                # loss_sect = self.loss_sect(sent_sect_scores.permute(0, 2, 1), sent_sect_labels)
                # loss_sect = (loss_sect * mask.float()).sum(dim=1)
                # loss_sect = ((loss_sect)/(mask.sum(dim=1).float())).sum()

                # loss_sect = self.loss_sect(sent_sect_scores.permute(0, 2, 1), sent_sect_labels)
                # loss_sect = (loss_sect * mask.float()).sum()
                # loss_sect = (loss_sect / loss_sect.numel())


                # loss_sent = self.alpha * loss_sent
                # loss_sect = (1 - self.alpha) * (loss_sect)
                # loss = loss_sent + loss_sect


                try:
                    acc, pred = self._get_mertrics(sent_sect_scores, sent_sect_labels, mask=mask, task='sent_sect')
                except:
                    import pdb;
                    pdb.set_trace()

                batch_stats = Statistics(loss=float(loss.cpu().data.numpy().sum()),
                                         loss_sect=float(loss_sect.cpu().data.numpy().sum()),
                                         loss_sent=float(loss_sent.cpu().data.numpy().sum()), n_docs=normalization,
                                         n_acc=batch.batch_size,
                                         RMSE=self._get_mertrics(sent_scores, sent_rg_scores, mask=mask, task='sent'),
                                         accuracy=acc,
                                         a1=self.model.uncertainty_loss._sigmas_sq[0].item(),
                                         a2=self.model.uncertainty_loss._sigmas_sq[1].item()
                                         )


            else:  # simple

                if not self.rg_predictor:
                    sent_scores, mask, loss, _, _ = self.model(src, segs, clss, mask, mask_cls, sent_bin_labels=sent_bin_labels, sent_sect_labels=None)
                else:
                    sent_scores, mask, loss, _, _ = self.model(src, segs, clss, mask, mask_cls, sent_bin_labels=sent_rg_scores, sent_sect_labels=None)


                # loss = self.loss(sent_scores, sent_rg_scores.float())


                batch_stats = Statistics(loss=float(loss.cpu().data.numpy().sum()),
                                         RMSE=self._get_mertrics(sent_scores, sent_rg_scores, mask=mask,
                                                                 task='sent'),
                                         n_acc=batch.batch_size,
                                         n_docs=normalization,
                                         a1=self.model.uncertainty_loss._sigmas_sq[0] if self.is_joint else 0,
                                         a2=self.model.uncertainty_loss._sigmas_sq[1] if self.is_joint else 0)

            loss.backward()
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))
                # self.optim.step(report_stats=report_stats)

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step(report_stats)

    def _save(self, step, best=False):
        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optim': self.optim,
        }
        if best:
            checkpoint_path = os.path.join(self.args.model_path, 'BEST_model.pt')
        else:
            checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)

        logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases
        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)
        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate, alpha_1, alpha2,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:

            return self.report_manager.report_training(
                step, num_steps, learning_rate, alpha_1, alpha2, report_stats, is_joint=self.is_joint,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate,  step, alpha1=0, alpha2=0, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, alpha1, alpha2, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)

    def _report_rouge(self, predictions, references):
        # r1, r2, rl, r1_cf, r2_cf, rl_cf = utils.rouge.get_rouge(predictions, references, use_cf=False)
        r1, r2, rl = evaluate_rouge_avg(predictions, references)
        # r1, r2, rl = get_rouge_pap(predictions, references)

        if len(self.args.log_folds) > 0:
            with open(self.args.log_folds, mode='a') as f:
                f.write("{:.4f}\t{:.4f}\t{:.4f}".format(r1 / 100, r2 / 100, rl / 100))
                f.write('\n')
        logger.info("Metric\tScore\t95% CI")
        logger.info("ROUGE-1\t{:.2f}\t({:.2f},{:.2f})".format(r1*100, 0, 0))
        logger.info("ROUGE-2\t{:.2f}\t({:.2f},{:.2f})".format(r2*100, 0, 0))
        logger.info("ROUGE-L\t{:.2f}\t({:.2f},{:.2f})".format(rl*100, 0, 0))
        return r1, r2, rl

    def _get_mertrics(self, sent_scores, labels, mask=None, task='sent_sect'):

        labels = labels.to('cuda')
        sent_scores = sent_scores.to('cuda')
        mask = mask.to('cuda')

        if task == 'sent_sect':

            sent_scores = self.softmax_acc_pred(sent_scores)
            pred = torch.max(sent_scores, 2)[1]
            acc = (((pred == labels) * mask.cuda()).sum(dim=1)).to(dtype=torch.float) / \
                  mask.sum(dim=1).to(dtype=torch.float)

            return acc.sum().item(), pred

        else:
            mseLoss = self.rmse_loss(sent_scores.float(), labels.float())
            mseLoss = (mseLoss.float() * mask.float()).sum(dim=1)

            # sent_scores = self.softmax_sent(sent_scores)
            # import pdb;pdb.set_trace()
            # pred = torch.max(sent_scores, 2)[1]
            # acc = (((pred == labels) * mask.cuda()).sum(dim=1)).to(dtype=torch.float) / mask.sum(dim=1).to(dtype=torch.float)
            return mseLoss.sum().item()

    def _get_preds(self, sent_scores, labels, mask=None, task='sent_sect'):

        sent_scores = sent_scores.to('cuda')
        sent_scores = self.softmax_acc_pred(sent_scores)
        pred = torch.max(sent_scores, 2)[1]
        return pred


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion.pdf')

def _get_ir_eval_metrics(preds_with_idx, sent_labels_true, n=10):
    avg_scores = {'f':[], 'p':[], 'r':[]}
    for p_id, pred_with_idx in preds_with_idx.items():
        retrieved_idx = [pred[1] for pred in pred_with_idx]
        retrieved_true_labels = [sent_labels_true[p_id][idx] for idx in retrieved_idx]
        avg_scores['p'].append(retrieved_true_labels.count(1) / n)
    return np.mean(avg_scores['p'])
