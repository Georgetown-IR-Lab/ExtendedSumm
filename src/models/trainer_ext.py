import collections
import glob
import json
import os
import pickle
# from multiprocessing.pool import Pool
import traceback
from operator import itemgetter

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score
from tensorboardX import SummaryWriter
from torch.multiprocessing import Pool
from tqdm import tqdm

import distributed
from models.reporter_ext import ReportMgr, Statistics
from others.logging import logger
from prepro.data_builder import LongformerData
# from utils.rouge_pap import get_rouge_pap
from utils.rouge_score import evaluate_rouge


def _multi_rg(params):
    return evaluate_rouge([params[0]], [params[1]])


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

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)

    trainer = Trainer(args, model, optim, grad_accum_count, n_gpu, gpu_rank, report_manager)

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
        self.overall_recalls = []
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
        self.overall_recall = -100000
        self.softmax = nn.Softmax(dim=1)
        self.softmax_acc_pred = nn.Softmax(dim=2)
        self.softmax_sent = nn.Softmax(dim=1)
        assert grad_accum_count > 0
        # Set model in training mode.
        if (model):
            self.model.train()

        self.bert = LongformerData(args)

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
                            val_stat, best_model_save, best_recall_model_save = self.validate_rouge_baseline(
                                valid_iter_fct, step,
                                valid_gl_stats=valid_global_stats)
                            self._save(step, best=best_model_save, recall_model=best_recall_model_save, valstat=val_stat)

                        if step == 5 or step % self.args.val_interval == 0:  # Validation
                        # if step % self.args.val_interval == 0:  # Validation
                            logger.info('----------------------------------------')
                            logger.info('Start evaluating on evaluation set... ')
                            self.args.pick_top = True

                            val_stat, best_model_save, best_recall_model_save = self.validate_rouge_baseline(valid_iter_fct, step,
                                                                                     valid_gl_stats=valid_global_stats)


                            if best_model_save:
                                self._save(step, best=True, valstat=val_stat)
                                logger.info(f'Best model saved sucessfully at step %d' % step)
                                self.best_val_step = step

                            if best_recall_model_save:
                                self._save(step, best=True, valstat=val_stat, recall_model=True)
                                logger.info(f'Best model saved sucessfully at step %d' % step)
                                self.best_val_step = step

                            self.save_validation_results(step, val_stat)

                            logger.info('----------------------------------------')
                            self.model.train()

                        step += 1
                        if step > train_steps:
                            break
            train_iter = train_iter_fct()

        return total_stats

    def validate_rouge_baseline(self, valid_iter_fct, step=0, valid_gl_stats=None, write_scores_to_pickle=False):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """

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
        sent_sects_whole_pred = {}
        sent_sects_whole_true = {}
        sent_labels_true = {}
        sent_numbers_whole = {}
        paper_srcs = {}
        paper_tgts = {}
        sent_sect_wise_rg_whole = {}
        sent_sections_txt_whole = {}
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()
        best_model_saved = False
        best_recall_model_saved = False

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
                mask = batch.mask_src
                mask_cls = batch.mask_cls
                p_id = batch.paper_id
                segment_src = batch.src_str
                paper_tgt = batch.tgt_str
                sent_sect_wise_rg = batch.sent_sect_wise_rg
                sent_sections_txt = batch.sent_sections_txt
                sent_numbers = batch.sent_numbers

                sent_sect_labels = batch.sent_sect_labels
                if self.is_joint:
                    if not self.rg_predictor:
                        sent_scores, sent_sect_scores, mask, loss, loss_sent, loss_sect = self.model(src, segs, clss,
                                                                                                     mask, mask_cls,
                                                                                                     sent_labels,
                                                                                                     sent_sect_labels)
                    else:
                        sent_scores, sent_sect_scores, mask, loss, loss_sent, loss_sect = self.model(src, segs, clss,
                                                                                                     mask, mask_cls,
                                                                                                     sent_true_rg,
                                                                                                     sent_sect_labels)
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
                        sent_scores, mask, loss, _, _ = self.model(src, segs, clss, mask, mask_cls, sent_labels,
                                                                   sent_sect_labels=None, is_inference=True)
                    else:
                        sent_scores, mask, loss, _, _ = self.model(src, segs, clss, mask, mask_cls, sent_true_rg,
                                                                   sent_sect_labels=None, is_inference=True)

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
                        masked_scores = masked_scores[np.nonzero(masked_scores)]

                        masked_sent_labels_true = (sent_labels[idx] + 1) * mask[idx].long()

                        masked_sent_labels_true = masked_sent_labels_true[np.nonzero(masked_sent_labels_true)].flatten()
                        masked_sent_labels_true = (masked_sent_labels_true - 1)

                        sent_scores_whole[p_id] = masked_scores
                        sent_labels_true[p_id] = masked_sent_labels_true.cpu()

                        masked_sents_sections_true = (sent_sect_labels[idx] + 1) * mask[idx].long()

                        masked_sents_sections_true = masked_sents_sections_true[
                            np.nonzero(masked_sents_sections_true)].flatten()
                        masked_sents_sections_true = (masked_sents_sections_true - 1)
                        sent_sects_whole_true[p_id] = masked_sents_sections_true.cpu()

                        if self.is_joint:
                            masked_scores_sects = sent_sect_scores[idx] * mask[idx].view(-1, 1).expand_as(
                                sent_sect_scores[idx]).float()
                            masked_scores_sects = masked_scores_sects[torch.abs(masked_scores_sects).sum(dim=1) != 0]
                            sent_sects_whole_pred[p_id] = torch.max(self.softmax(masked_scores_sects), 1)[1].cpu()

                        paper_srcs[p_id] = segment_src[idx]
                        if sent_numbers[0] is not None:
                            sent_numbers_whole[p_id] = sent_numbers[idx]
                            # sent_tokens_count_whole[p_id] = sent_tokens_count[idx]
                        paper_tgts[p_id] = paper_tgt[idx]
                        sent_sect_wise_rg_whole[p_id] = sent_sect_wise_rg[idx]
                        sent_sections_txt_whole[p_id] = sent_sections_txt[idx]


                    else:
                        masked_scores = sent_scores[idx] * mask[idx].cpu().data.numpy()
                        masked_scores = masked_scores[np.nonzero(masked_scores)]

                        masked_sent_labels_true = (sent_labels[idx] + 1) * mask[idx].long()
                        masked_sent_labels_true = masked_sent_labels_true[np.nonzero(masked_sent_labels_true)].flatten()
                        masked_sent_labels_true = (masked_sent_labels_true - 1)

                        sent_scores_whole[p_id] = np.concatenate((sent_scores_whole[p_id], masked_scores), 0)
                        sent_labels_true[p_id] = np.concatenate((sent_labels_true[p_id], masked_sent_labels_true.cpu()),
                                                                0)

                        masked_sents_sections_true = (sent_sect_labels[idx] + 1) * mask[idx].long()
                        masked_sents_sections_true = masked_sents_sections_true[
                            np.nonzero(masked_sents_sections_true)].flatten()
                        masked_sents_sections_true = (masked_sents_sections_true - 1)
                        sent_sects_whole_true[p_id] = np.concatenate(
                            (sent_sects_whole_true[p_id], masked_sents_sections_true.cpu()), 0)

                        if self.is_joint:
                            masked_scores_sects = sent_sect_scores[idx] * mask[idx].view(-1, 1).expand_as(
                                sent_sect_scores[idx]).float()
                            masked_scores_sects = masked_scores_sects[
                                torch.abs(masked_scores_sects).sum(dim=1) != 0]
                            sent_sects_whole_pred[p_id] = np.concatenate(
                                (sent_sects_whole_pred[p_id], torch.max(self.softmax(masked_scores_sects), 1)[1].cpu()),
                                0)

                        paper_srcs[p_id] = np.concatenate((paper_srcs[p_id], segment_src[idx]), 0)
                        if sent_numbers[0] is not None:
                            sent_numbers_whole[p_id] = np.concatenate((sent_numbers_whole[p_id], sent_numbers[idx]), 0)
                            # sent_tokens_count_whole[p_id] = np.concatenate(
                            #     (sent_tokens_count_whole[p_id], sent_tokens_count[idx]), 0)

                        sent_sect_wise_rg_whole[p_id] = np.concatenate(
                            (sent_sect_wise_rg_whole[p_id], sent_sect_wise_rg[idx]), 0)
                        sent_sections_txt_whole[p_id] = np.concatenate(
                            (sent_sections_txt_whole[p_id], sent_sections_txt[idx]), 0)


        PRED_LEN = self.args.val_pred_len
        acum_f_sent_labels = 0
        acum_p_sent_labels = 0
        acum_r_sent_labels = 0
        acc_total = 0
        for p_idx, (p_id, sent_scores) in enumerate(sent_scores_whole.items()):
            # sent_true_labels = pickle.load(open("sent_labels_files/pubmedL/val.labels.p", "rb"))
            # section_textual = np.array(section_textual)
            paper_sent_true_labels = np.array(sent_labels_true[p_id])
            if self.is_joint:
                sent_sects_true = np.array(sent_sects_whole_true[p_id])
                sent_sects_pred = np.array(sent_sects_whole_pred[p_id])

            sent_scores = np.array(sent_scores)
            p_src = np.array(paper_srcs[p_id])

            # selected_ids_unsorted = np.argsort(-sent_scores, 0)
            keep_ids = [idx for idx, s in enumerate(p_src) if
                        len(s.replace('.', '').replace(',', '').replace('(', '').replace(')', '').
                            replace('-', '').replace(':', '').replace(';', '').replace('*', '').split()) > 5 and
                        len(s.replace('.', '').replace(',', '').replace('(', '').replace(')', '').
                            replace('-', '').replace(':', '').replace(';', '').replace('*', '').split()) < 100
                        ]

            keep_ids = sorted(keep_ids)

            # top_sent_indexes = top_sent_indexes[top_sent_indexes]
            p_src = p_src[keep_ids]
            sent_scores = sent_scores[keep_ids]
            paper_sent_true_labels = paper_sent_true_labels[keep_ids]

            sent_scores = np.asarray([s - 1.00 for s in sent_scores])

            selected_ids_unsorted = np.argsort(-sent_scores, 0)

            _pred = []
            for j in selected_ids_unsorted:
                if (j >= len(p_src)):
                    continue
                candidate = p_src[j].strip()
                if True:
                    # if (not _block_tri(candidate, _pred)):
                    _pred.append((candidate, j))

                if (len(_pred) == PRED_LEN):
                    break
            _pred = sorted(_pred, key=lambda x: x[1])
            _pred_final_str = '<q>'.join([x[0] for x in _pred])

            preds[p_id] = _pred_final_str
            golds[p_id] = paper_tgts[p_id]
            preds_with_idx[p_id] = _pred
            if p_idx > 10:
                f, p, r = _get_precision_(paper_sent_true_labels, [p[1] for p in _pred])
                if self.is_joint:
                    acc_whole = _get_accuracy_sections(sent_sects_true, sent_sects_pred, [p[1] for p in _pred])
                    acc_total += acc_whole

            else:
                f, p, r = _get_precision_(paper_sent_true_labels, [p[1] for p in _pred], print_few=True, p_id=p_id)
                if self.is_joint:
                    acc_whole = _get_accuracy_sections(sent_sects_true, sent_sects_pred, [p[1] for p in _pred],
                                                       print_few=True, p_id=p_id)
                    acc_total += acc_whole

            acum_f_sent_labels += f
            acum_p_sent_labels += p
            acum_r_sent_labels += r

        for id, pred in preds.items():
            save_pred.write(pred.strip().replace('<q>', ' ') + '\n')
            save_gold.write(golds[id].replace('<q>', ' ').strip() + '\n')

        print(f'Gold: {gold_path}')
        print(f'Prediction: {can_path}')

        r1, r2, rl = self._report_rouge(preds.values(), golds.values())
        stats.set_rl(r1, r2, rl)
        logger.info("F-score: %4.4f, Prec: %4.4f, Recall: %4.4f" % (
        acum_f_sent_labels / len(sent_scores_whole), acum_p_sent_labels / len(sent_scores_whole),
        acum_r_sent_labels / len(sent_scores_whole)))
        if self.is_joint:
            logger.info("Section Accuracy: %4.4f" % (acc_total / len(sent_scores_whole)))


        stats.set_ir_metrics(acum_f_sent_labels / len(sent_scores_whole),
                             acum_p_sent_labels / len(sent_scores_whole),
                             acum_r_sent_labels / len(sent_scores_whole))
        self.valid_rgls.append((r2 + rl) / 2)
        self._report_step(0, step,
                          self.model.uncertainty_loss._sigmas_sq[0] if self.is_joint else 0,
                          self.model.uncertainty_loss._sigmas_sq[1] if self.is_joint else 0,
                          valid_stats=stats)

        if len(self.valid_rgls) > 0:
            if self.min_rl < self.valid_rgls[-1]:
                self.min_rl = self.valid_rgls[-1]
                best_model_saved = True

        return stats, best_model_saved, best_recall_model_saved

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
            segs = batch.segs
            clss = batch.clss
            mask = batch.mask_src
            mask_cls = batch.mask_cls

            if self.is_joint:
                if not self.rg_predictor:
                    sent_scores, sent_sect_scores, mask, loss, loss_sent, loss_sect = self.model(src, segs, clss, mask,
                                                                                                 mask_cls,
                                                                                                 sent_bin_labels,
                                                                                                 sent_sect_labels)
                else:
                    sent_scores, sent_sect_scores, mask, loss, loss_sent, loss_sect = self.model(src, segs, clss, mask,
                                                                                                 mask_cls,
                                                                                                 sent_rg_scores,
                                                                                                 sent_sect_labels)
                try:
                    acc, pred = self._get_mertrics(sent_sect_scores, sent_sect_labels, mask=mask, task='sent_sect')
                except:
                    logger.info("Accuracy cannot be computed due to some errors in loading approapriate files...")

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
                    sent_scores, mask, loss, _, _ = self.model(src, segs, clss, mask, mask_cls,
                                                               sent_bin_labels=sent_bin_labels, sent_sect_labels=None)
                else:
                    sent_scores, mask, loss, _, _ = self.model(src, segs, clss, mask, mask_cls,
                                                               sent_bin_labels=sent_rg_scores, sent_sect_labels=None)

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

    def _save(self, step, best=False, valstat=None, recall_model=False):

        BEST_MODEL_NAME = 'BEST_model_s%d_%4.4f_%4.4f_%4.4f.pt' % (step, valstat.r1, valstat.r2, valstat.rl)

        if recall_model:
            BEST_MODEL_NAME = 'Recall_BEST_model_s%d_%4.4f.pt' % (step, valstat.top_sents_recall)

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
            checkpoint_path_best = os.path.join(self.args.model_path, BEST_MODEL_NAME)

        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)

        if not recall_model:
            logger.info("Saving checkpoint %s" % checkpoint_path)
        else:
            logger.info("Saving checkpoint recall %s" % checkpoint_path)

        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)

        if best:
            if not recall_model:
                best_path = glob.glob(self.args.model_path + '/BEST_model*.pt')
            else:
                best_path = glob.glob(self.args.model_path + '/Recall_BEST_model*.pt')

            if len(best_path) > 0:
                for best in best_path:
                    os.remove(best)
                torch.save(checkpoint, checkpoint_path_best)
            else:
                torch.save(checkpoint, checkpoint_path_best)

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

    def _report_step(self, learning_rate, step, alpha1=0, alpha2=0, train_stats=None,
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

        a_lst = []
        predictions = list(predictions)
        references = list(references)
        for i, p in enumerate(predictions):
            a_lst.append((p, references[i]))

        pool = Pool(24)
        rouge_scores = {"r1": [], "r2": [], "rl": []}
        for d in tqdm(pool.imap(_multi_rg, a_lst), total=len(a_lst)):
            if d is not None:
                rouge_scores["r1"].append(d[0])
                rouge_scores["r2"].append(d[1])
                rouge_scores["rl"].append(d[2])
        pool.close()
        pool.join()

        r1 = np.mean(rouge_scores["r1"])
        r2 = np.mean(rouge_scores["r2"])
        rl = np.mean(rouge_scores["rl"])

        if len(self.args.log_folds) > 0:
            with open(self.args.log_folds, mode='a') as f:
                f.write("{:.4f}\t{:.4f}\t{:.4f}".format(r1 / 100, r2 / 100, rl / 100))
                f.write('\n')
        logger.info("Metric\tScore\t95% CI")
        logger.info("ROUGE-1\t{:.2f}\t({:.2f},{:.2f})".format(r1 * 100, 0, 0))
        logger.info("ROUGE-2\t{:.2f}\t({:.2f},{:.2f})".format(r2 * 100, 0, 0))
        logger.info("ROUGE-L\t{:.2f}\t({:.2f},{:.2f})".format(rl * 100, 0, 0))

        logger.info("Data path: %s" % self.args.bert_data_path)
        logger.info("Model path: %s" % self.args.model_path)

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

            return mseLoss.sum().item()

    def _get_preds(self, sent_scores, labels, mask=None, task='sent_sect'):

        sent_scores = sent_scores.to('cuda')
        sent_scores = self.softmax_acc_pred(sent_scores)
        pred = torch.max(sent_scores, 2)[1]
        return pred

    def save_validation_results(self, step, val_stat):
        def is_non_zero_file(fpath):
            return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

        def check_path_existence(dir):
            if os.path.exists(dir):
                return
            else:
                os.makedirs(dir)

        check_path_existence(os.path.join(self.args.model_path, "val_results"))
        if not is_non_zero_file(os.path.join(self.args.model_path, "val_results", "val.json")) or step < 100:
        # if not is_non_zero_file(os.path.join(self.args.model_path, "val_results", "val.json")):
            results = collections.defaultdict(dict)

            for metric, score in zip(["RG-1", "RG-2", "RG-L"], [val_stat.r1, val_stat.r2, val_stat.rl]):
                results[str(step)][metric] = score

            results[str(step)]["Recall-top"] = val_stat.top_sents_recall
            results[str(step)]["F1"] = val_stat.f
            with open(os.path.join(self.args.model_path, "val_results", "val.json"), mode='w') as F:
                json.dump(results, F, indent=4)
        else:
            results = json.load(open(os.path.join(self.args.model_path, "val_results", "val.json")))
            results_all = collections.defaultdict(dict)
            for key, val in results.items():
                for k, v in val.items():
                    results_all[key][k] = v

            # results_all[str(step)] = step
            # results_all[str(step)]["F1"] = val_stat.f
            for metric, score in zip(["RG-1", "RG-2", "RG-L"], [val_stat.r1, val_stat.r2, val_stat.rl]):
                results_all[str(step)][metric] = score

            results_all[str(step)]["Recall-top"] = val_stat.top_sents_recall
            results_all[str(step)]["F1"] = val_stat.f
            with open(os.path.join(self.args.model_path, "val_results", "val.json"), mode='w') as F:
                json.dump(results_all, F, indent=4)


def _get_ir_eval_metrics(preds_with_idx, sent_labels_true, n=10):
    avg_scores = {'f': [], 'p': [], 'r': []}
    for p_id, pred_with_idx in preds_with_idx.items():
        retrieved_idx = [pred[1] for pred in pred_with_idx]
        retrieved_true_labels = [sent_labels_true[p_id][idx] for idx in retrieved_idx]
        avg_scores['p'].append(retrieved_true_labels.count(1) / n)
    return np.mean(avg_scores['p'])


def _get_precision_(sent_true_labels, summary_idx, print_few=False, p_id=''):
    oracle_cout = sum(sent_true_labels)
    if oracle_cout == 0:
        return 0, 0, 0

    # oracle_cout = oracle_cout if oracle_cout > 0 else 1
    pos = 0
    neg = 0
    for idx in summary_idx:
        if sent_true_labels[idx] == 0:
            neg += 1
        else:
            pos += 1

    if print_few:
        logger.info("paper_id: {} ==> positive/negative cases: {}/{}".format(p_id, pos, neg))

    if pos == 0:
        return 0, 0, 0
    prec = pos / len(summary_idx)

    # recall --how many relevants are retrieved?
    recall = pos / int(oracle_cout)

    try:
        F = (2 * prec * recall) / (prec + recall)
        return F, prec, recall

    except Exception:
        traceback.print_exc()
        os._exit(2)


def _get_accuracy_sections(sent_sects_true, sent_sects_pred, summary_idx, print_few=False, p_id=''):
    acc = 0

    for idx in summary_idx:
        if sent_sects_true[idx] == sent_sects_pred[idx]:
            acc += 1

    if print_few:
        logger.info("paper_id: {} ==> acc: {}".format(p_id, acc / len(summary_idx)))

    return acc / len(summary_idx)
