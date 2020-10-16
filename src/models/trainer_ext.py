import os
import pickle
from operator import itemgetter

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm

import distributed
from models.reporter_ext import ReportMgr, Statistics
from models.uncertainty_loss import UncertaintyLoss
from others.logging import logger
from prepro.data_builder import check_path_existence
# from utils.rouge_pap import get_rouge_pap
from utils.rouge_score import evaluate_rouge
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
                            self.model.uncertainty_loss.alpha[0] if self.is_joint else 0,
                            self.model.uncertainty_loss.alpha[1] if self.is_joint else 0,
                            report_stats)

                        self._report_step(self.optim.learning_rate, step,
                                          self.model.uncertainty_loss.alpha[0] if self.is_joint else 0,
                                          self.model.uncertainty_loss.alpha[1] if self.is_joint else 0,
                                          train_stats=report_stats)

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            self._save(step)

                        # if step == 60 or step % self.args.val_interval == 0:  # Validation
                        if step % self.args.val_interval == 0:  # Validation
                            logger.info('----------------------------------------')
                            logger.info('Start evaluating on evaluation set... ')
                            val_stat, best_model_save = self.validate_rouge(valid_iter_fct, step,
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

    # def validate_cls(self, valid_iter_fct, step=0, valid_gl_stats=None):
    #     # Set model in validating mode.
    #     def _get_ngrams(n, text):
    #         ngram_set = set()
    #         text_length = len(text)
    #         max_index_ngram_start = text_length - n
    #         for i in range(max_index_ngram_start + 1):
    #             ngram_set.add(tuple(text[i:i + n]))
    #         return ngram_set
    #
    #     # Set model in validating mode.
    #     self.model.eval()
    #     stats = Statistics()
    #     best_model_saved = False
    #     valid_iter = valid_iter_fct()
    #     paper_sect_labels = {}
    #     paper_srcs = {}
    #     y_true = []
    #     y_pred = []
    #
    #     with torch.no_grad():
    #         for batch in valid_iter:
    #             src = batch.src
    #             labels = batch.src_sent_labels
    #             sent_labels = batch.sent_labels
    #             segs = batch.segs
    #             clss = batch.clss
    #             mask = batch.mask_src
    #             mask_cls = batch.mask_cls
    #             sent_sect_labels = batch.sent_sect_labels
    #             segment_src = batch.src_str
    #             p_ids = batch.paper_id
    #
    #             if self.is_joint:
    #                 sent_scores, sent_sect_scores, mask = self.model(src, segs, clss, mask, mask_cls)
    #
    #                 # loss_sent = self.loss(sent_scores, labels.float())
    #                 # loss_sent = self.bin_loss(sent_scores, sent_labels.float())
    #                 loss_sect = self.loss_sect(sent_sect_scores.permute(0, 2, 1), sent_sect_labels)
    #                 loss_sect = (loss_sect * mask.float()).sum()
    #                 # loss_sent = (loss_sent * mask.float()).sum()
    #
    #                 # loss_sent = self.alpha * loss_sent
    #                 # loss_sect = (1 - self.alpha) * loss_sect
    #
    #                 # loss_sent = loss_sent
    #                 # loss_sect = loss_sect
    #
    #                 # loss = loss_sent + loss_sect
    #                 loss = loss_sect
    #                 acc, pred = self._get_mertrics(sent_sect_scores, sent_sect_labels, mask=mask,
    #                                             task='sent_sect')
    #
    #
    #                 batch_stats = Statistics(loss=float(loss.cpu().data.numpy().sum()),
    #                                          loss_sect=float(loss_sect.cpu().data.numpy().sum()),
    #                                          loss_sent=float(loss_sect.cpu().data.numpy().sum()),
    #                                          n_docs=len(labels),
    #                                          n_acc=batch.batch_size,
    #                                          RMSE=self._get_mertrics(sent_scores, labels, mask=mask, task='sent'),
    #                                          accuracy=acc)
    #
    #             else:
    #                 sent_scores, mask, loss, _, _ = self.model(src, segs, clss, mask, mask_cls)
    #                 loss = self.bin_loss(sent_scores, sent_labels.float())
    #
    #                 loss = (loss * mask.float()).sum()
    #
    #                 batch_stats = Statistics(loss=float(loss.cpu().data.numpy().sum()),
    #                                          RMSE=self._get_mertrics(sent_scores, labels, mask=mask, task='sent'),
    #                                          n_acc=batch.batch_size,
    #                                          n_docs=len(labels))
    #
    #             stats.update(batch_stats)
    #
    #             for idx, p_id in enumerate(p_ids):
    #                 if p_id not in paper_sect_labels.keys():
    #                     # masked_scores = sent_scores[idx] * mask[idx].cpu().data.numpy()
    #                     # masked_scores = masked_scores[masked_scores.nonzero()]
    #                     # paper_sect_labels[p_id] = masked_scores
    #                     if self.is_joint:
    #                         masked_scores_sects = sent_sect_scores[idx] * mask[idx].view(-1,1).expand_as(sent_sect_scores[idx]).float()
    #                         # import pdb;pdb.set_trace()
    #                         masked_scores_sects = masked_scores_sects[torch.abs(masked_scores_sects).sum(dim=1) != 0]
    #
    #                         masked_true_labels = (sent_sect_labels[idx] + 1) * mask[idx].long()
    #                         masked_true_labels = masked_true_labels[masked_true_labels.nonzero()].flatten()
    #                         masked_true_labels = masked_true_labels -1
    #                         paper_sect_labels[p_id] = torch.max(self.softmax(masked_scores_sects), 1)[1].cpu()
    #
    #                         # if len(torch.max(self.softmax(masked_scores_sects), 1)[1].cpu().numpy()) != len(masked_true_labels.cpu().numpy()):
    #                         #     import pdb;pdb.set_trace()
    #                         y_pred.extend(torch.max(self.softmax(masked_scores_sects), 1)[1].cpu().numpy())
    #                         y_true.extend(masked_true_labels.cpu().numpy())
    #                         # paper_sect_labels[p_id] = masked_true_labels.cpu()
    #                     paper_srcs[p_id] = segment_src[idx]
    #
    #                 else:
    #                     if self.is_joint:
    #                         masked_scores_sects = sent_sect_scores[idx] * mask[idx].view(-1, 1).expand_as(
    #                             sent_sect_scores[idx]).float()
    #                         masked_scores_sects = masked_scores_sects[
    #                             torch.abs(masked_scores_sects).sum(dim=1) != 0]
    #                         masked_true_labels = sent_sect_labels[idx] * mask[idx].long()
    #                         masked_true_labels = masked_true_labels[masked_true_labels.nonzero()].flatten()
    #
    #                         paper_sect_labels[p_id] = np.concatenate((paper_sect_labels[p_id], torch.max(self.softmax(masked_scores_sects), 1)[1].cpu()), 0)
    #                         y_pred.extend(torch.max(self.softmax(masked_scores_sects), 1)[1].cpu())
    #                         y_true.extend(masked_true_labels)
    #
    #                     paper_srcs[p_id] = np.concatenate((paper_srcs[p_id], segment_src[idx]), 0)
    #
    #     # import pdb;pdb.set_trace()
    #     # self.plot_confusion(y_pred, y_true)
    #
    #     self._report_step(0, step, valid_stats=stats)
    #     self.valid_rgls.append(stats._get_acc_sect())
    #     if len(self.valid_rgls) > 0:
    #         if self.min_rl < self.valid_rgls[-1]:
    #             self.min_rl = self.valid_rgls[-1]
    #             best_model_saved = True
    #
    #     return stats, best_model_saved

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
                if self.rg_predictor:
                    sent_true_rg = batch.src_sent_labels
                else:
                    sent_labels = batch.sent_labels
                segs = batch.segs
                clss = batch.clss
                section_rg = batch.section_rg
                mask = batch.mask_src
                mask_cls = batch.mask_cls
                p_ids = batch.paper_id
                segment_src = batch.src_str
                paper_tgt = batch.tgt_str

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
                    sent_scores, mask, loss, _, _ = self.model(src, segs, clss, mask, mask_cls, sent_labels, sent_sect_labels=None, is_inference=True)
                    sent_scores = (section_rg.unsqueeze(1).expand_as(sent_scores).to(device='cuda')*100) * sent_scores


                    batch_stats = Statistics(loss=float(loss.cpu().data.numpy().sum()),
                                             RMSE=self._get_mertrics(sent_scores, labels, mask=mask, task='sent'),
                                             n_acc=batch.batch_size,
                                             n_docs=len(labels))

                stats.update(batch_stats)

                sent_scores = sent_scores + mask.float()
                sent_scores = sent_scores.cpu().data.numpy()

                for idx, p_id in enumerate(p_ids):
                    p_id = p_id.split('__')[0]

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

        PRED_LEN = 10
        # LENGTH_LIMIT= 100
        for p_ids, sent_scores in sent_scores_whole.items():
            selected_ids_unsorted = np.argsort(-sent_scores, 0)
            selected_ids_unsorted = [s for s in selected_ids_unsorted if
                                     len(paper_srcs[p_ids][s].replace('.', '').replace(',', '').replace('(', '').replace(')', '').
                                         replace('-', '').replace(':','').replace(';','').replace('*','').split()) > 5 and
                                     len(paper_srcs[p_ids][s].replace('.', '').replace(',', '').replace('(', '').replace(')', '').
                                         replace('-', '').replace(':','').replace(';','').replace('*','').split()) < 100]

            selected_ids_top = selected_ids_unsorted[:PRED_LEN]
            selected_ids = sorted(selected_ids_top, reverse=False)
            _pred_final = [(paper_srcs[p_ids][selected_ids_unsorted[0]].strip(),selected_ids_unsorted[0])]
            try:
                if self.is_joint:
                    summary_sect_pred = [sent_sects_whole[p_ids][selected_ids_unsorted[0]]]

                summary_sect_true = [sent_sects_whole_true[p_ids][selected_ids_unsorted[0]]]
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
            picked_up_word_count = len(paper_srcs[p_ids][selected_ids_unsorted[0]].strip().split())
            for j in selected_ids[:len(paper_srcs[p_ids])]:
                if (j >= len(paper_srcs[p_ids][0])):
                    continue
                candidate = paper_srcs[p_ids][j].strip()
                if (self.args.block_trigram):
                    if (not _block_tri(candidate, [x[0] for x in _pred_final])):
                        _pred_final = _insert(_pred_final, candidate, j)
                        if self.is_joint:
                            summary_sect_pred.append(sent_sects_whole[p_ids][j])

                        try:
                            summary_sect_true.append(sent_sects_whole_true[p_ids][j])
                        except:
                            import pdb;pdb.set_trace()
                        picked_up += 1
                        picked_up_word_count += len(candidate.split())
                else:
                    _pred_final.append((candidate, j))
                    if self.is_joint:
                        summary_sect_pred.append(sent_sects_whole[p_ids][j])
                        summary_sect_true.append(sent_sects_whole_true[p_ids][j])
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
                    candidate = paper_srcs[p_ids][k].strip()
                    if (self.args.block_trigram):
                        if (not _block_tri(candidate, [x[0] for x in _pred_final])):
                            _pred_final = _insert(_pred_final, candidate, k)
                            if self.is_joint:
                                summary_sect_pred.append(sent_sects_whole[p_ids][k])
                            summary_sect_true.append(sent_sects_whole_true[p_ids][k])
                            picked_up += 1
                            picked_up_word_count += len(candidate.split())

                    else:
                        _pred_final.append((candidate, k))
                        if self.is_joint:
                            summary_sect_pred.append(sent_sects_whole[p_ids][k])
                        summary_sect_true.append(sent_sects_whole_true[p_ids][k])
                        picked_up += 1
                        picked_up_word_count += len(candidate.split())

                    if (picked_up == PRED_LEN):
                    # if (picked_up_word_count >= LENGTH_LIMIT):
                        break

            _pred_final_str = '<q>'.join([x[0] for x in _pred_final])
            # if (self.args.recall_eval):
            #     _pred = ' '.join(_pred_final_str.split()[:len(paper_tgts[p_ids].split())])

            preds[p_ids] = _pred_final_str
            golds[p_ids] = paper_tgts[p_ids]
            preds_with_idx[p_ids] = _pred_final
            if self.is_joint:
                preds_sects[p_ids] = summary_sect_pred
            preds_sects_true[p_ids] = summary_sect_true

        for id, pred in preds.items():
            save_pred.write(pred.strip().replace('<q>', ' ') + '\n')
            save_gold.write(golds[id].replace('<q>', ' ').strip() + '\n')

        p = _get_ir_eval_metrics(preds_with_idx, sent_labels_true)
        logger.info("Prection: {:4.4f}".format(p))

        print(f'Gold: {gold_path}')
        print(f'Prediction: {can_path}')


        r1, r2, rl = self._report_rouge(preds.values(), golds.values())
        stats.set_rl(r1, r2, rl)
        stats.set_ir_metrics(p)
        self.valid_rgls.append(r2)
        self._report_step(0, step,
                          self.model.uncertainty_loss.alpha[0] if self.is_joint else 0,
                          self.model.uncertainty_loss.alpha[1] if self.is_joint else 0,
                          valid_stats=stats)

        if len(self.valid_rgls) > 0:
            if self.min_rl < self.valid_rgls[-1]:
                self.min_rl = self.valid_rgls[-1]
                best_model_saved = True

        return stats, best_model_saved

    def test_cls_csabs(self, test_iter, step, cal_lead=False, cal_oracle=False):
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        self.model.eval()
        stats = Statistics()
        self.er = {}

        with torch.no_grad():
            for batch in test_iter:
                src = batch.src
                labels = batch.src_sent_labels
                sent_labels = batch.sent_labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask_src
                mask_cls = batch.mask_cls
                sent_sect_labels = batch.sent_sect_labels
                segment_src = batch.src_str
                paper_id = batch.paper_id

                if self.is_joint:
                    sent_scores, sent_sect_scores, mask = self.model(src, segs, clss, mask, mask_cls)

                    # loss_sent = self.loss(sent_scores, labels.float())
                    loss_sent = self.bin_loss(sent_scores, sent_labels.float())
                    loss_sect = self.loss_sect(sent_sect_scores.permute(0, 2, 1), sent_sect_labels)
                    loss_sect = (loss_sect * mask.float()).sum()
                    loss_sent = (loss_sent * mask.float()).sum()

                    loss_sent = self.alpha * loss_sent
                    loss_sect = (1 - self.alpha) * loss_sect

                    loss = loss_sect
                    acc, pred = self._get_mertrics(sent_sect_scores, sent_sect_labels, mask=mask,
                                                   task='sent_sect')

                    # for i, (p, l) in enumerate(zip(pred[0], sent_sect_labels[0])):
                    #     key = str(p.data.item()) + '->' + str(l.data.item())
                    #     if key in self.er:
                    #         self.er[key] += 1
                    #         with open('er/' + key + '.txt', mode='a')as f:
                    #             f.write(segment_src[0][i])
                    #             f.write('\n')
                    #     else:
                    #         self.er[key] = 1
                    #         with open('er/' + key + '.txt', mode='a')as f:
                    #             f.write(segment_src[0][i])
                    #             f.write('\n')

                    batch_stats = Statistics(loss=float(loss.cpu().data.numpy().sum()),
                                             loss_sect=float(loss_sect.cpu().data.numpy().sum()),
                                             loss_sent=float(loss_sent.cpu().data.numpy().sum()),
                                             n_docs=len(labels),
                                             n_acc=batch.batch_size,
                                             RMSE=self._get_mertrics(sent_scores, labels, mask=mask, task='sent'),
                                             accuracy=acc)

                else:
                    sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)
                    loss = self.bin_loss(sent_scores, sent_labels.float())

                    loss = (loss * mask.float()).sum()

                    batch_stats = Statistics(loss=float(loss.cpu().data.numpy().sum()),
                                             RMSE=self._get_mertrics(sent_scores, labels, mask=mask, task='sent'),
                                             n_acc=batch.batch_size,
                                             n_docs=len(labels))

                stats.update(batch_stats)

        import operator
        sorted_x = sorted(self.er.items(), key=operator.itemgetter(1))
        print(sorted_x)

    def test_cls(self, test_iter_fct, step, cal_lead=False, cal_oracle=False):
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

        stats = Statistics()
        preds = {}
        golds = {}
        can_path = '%s_step%d.candidate' % (self.args.result_path, step)
        gold_path = '%s_step%d.gold' % (self.args.result_path, step)
        save_pred = open(can_path, 'w')
        save_gold = open(gold_path, 'w')
        sent_whole_labels = {}
        paper_srcs = {}
        paper_tgts = {}
        test_iter = test_iter_fct()
        labels_dist = set()
        counter = 0
        for _ in test_iter:
            counter += 1
        test_iter = test_iter_fct()
        # list = []
        with torch.no_grad():
            for batch in tqdm(test_iter, total=counter):
                src = batch.src
                segs = batch.segs
                clss = batch.clss
                sent_bin_labels = batch.sent_labels
                mask = batch.mask_src
                mask_cls = batch.mask_cls
                paper_ids = batch.paper_id
                segment_src = batch.src_str
                sent_sect_labels = batch.sent_sect_labels
                # list.append(paper_id)

                if self.is_joint:
                    labels, sent_sect_scores, mask = self.model(src, segs, clss, mask, mask_cls)

                    loss_sect = self.loss_sect(sent_sect_scores.permute(0, 2, 1), sent_sect_labels)
                    loss_sect = (loss_sect * mask.float()).sum()

                    loss = loss_sect
                    # acc, pred = self._get_mertrics(sent_sect_scores, sent_sect_labels, mask=mask,
                    #                                task='sent_sect')

                    pred = self._get_preds(sent_sect_scores, sent_sect_labels, mask=mask,
                                                   task='sent_sect')
                    # import pdb;pdb.set_trace()

                    # print(acc)
                    batch_stats = Statistics(loss=float(loss.cpu().data.numpy()), loss_sect=loss_sect,
                                             loss_sent=loss_sect,
                                             n_docs=len(labels),
                                             accuracy=0.001,
                                             n_acc=batch.batch_size
                                             )

                else:
                    labels, mask = self.model(src, segs, clss, mask, mask_cls)
                    # loss = self.loss(sent_scores, labels.float())
                    loss = self.bin_loss(labels, sent_bin_labels.float())
                    loss = (loss * mask.float()).sum()
                    batch_stats = Statistics(loss=float(loss.cpu().data.numpy()),
                                             n_docs=len(labels))

                stats.update(batch_stats)

                # sent_scores = sent_scores + mask.float()
                # sent_scores = sent_scores.cpu().data.numpy()


                for idx, paper_id in enumerate(paper_ids):
                    if paper_id not in sent_whole_labels.keys():
                        sent_whole_labels[paper_id] = {}
                        prd = pred[idx].cpu().data.numpy()
                        msk = mask[idx].cpu().data.numpy()
                        masked_pred = prd[msk.nonzero()].flatten()

                        for j, sent in enumerate(segment_src[idx]):
                            sent_whole_labels[paper_id][sent.replace(' ', '').strip()[:60]] = masked_pred[j]
                            labels_dist.add(masked_pred[j])

                        # sent_whole_labels[paper_id] = pred[0].cpu().data.numpy()
                        # paper_srcs[paper_id] = segment_src[0]
                        # paper_tgts[paper_id] = paper_tgt[0]
                        # import pdb;
                        # pdb.set_trace()
                    else:
                        # import pdb;
                        # pdb.set_trace()
                        to_append = {}
                        prd = pred[idx].cpu().data.numpy()
                        msk = mask[idx].cpu().data.numpy()
                        masked_pred = prd[msk.nonzero()].flatten()

                        for j, sent in enumerate(segment_src[idx]):
                            to_append[sent.replace(' ', '').strip()[:60]] = masked_pred[j]
                            labels_dist.add(masked_pred[j])

                        # sent_whole_labels[paper_id] = np.concatenate((sent_whole_labels[paper_id], pred[0].cpu().data.numpy()), 0)

                        sent_whole_labels[paper_id] = {**sent_whole_labels[paper_id], **to_append}
            print(labels_dist)
            pickle.dump(sent_whole_labels, open("sentence_labels/" + self.args.exp_set + "_labels_arxivLong_myIndex_seq.p", "wb"))
            # for paper_id, labels in sent_scores_whole.items():

        self._report_step(0, step, valid_stats=stats)
        return stats

    def test(self, test_iter_fct, step, cal_lead=False, cal_oracle=False):
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

            if len(list) == 0:
                list = [(pred, pos)]
                return list

            # Searching for the position
            for i in range(len(list)):
                if list[i][1] > pos:
                    break
            # Inserting pred in the list
            list = list[:i] + [(pred, pos)] + list[i:]
            return list
            # return list

        if (not cal_lead and not cal_oracle):
            self.model.eval()
        stats = Statistics()
        preds = {}
        golds = {}
        check_path_existence(self.args.result_path)
        can_path = '%s/%s.source' % (self.args.result_path, self.args.exp_set)
        gold_path = '%s/%s.target' % (self.args.result_path, self.args.exp_set)

        official_pred = '%s-official.json' % (self.args.result_path)

        save_pred = open(can_path, 'w')
        save_gold = open(gold_path, 'w')
        sent_scores_whole = {}
        paper_srcs = {}
        src_strs = {}
        paper_tgts = {}
        test_iter = test_iter_fct()
        counter = 0
        for b in test_iter:
            counter += 1
        test_iter = test_iter_fct()
        with torch.no_grad():
            # for batch in tqdm(test_iter_fct, total=counter):
            for batch in tqdm(test_iter, total=counter):
                src = batch.src
                labels = batch.src_sent_labels
                segs = batch.segs
                clss = batch.clss
                sent_bin_labels = batch.sent_labels
                mask = batch.mask_src
                mask_cls = batch.mask_cls
                paper_id = batch.paper_id[0]
                isAbs = False
                try:
                    int(paper_id[0].replace('.json', '').replace('.pdf', ''))
                    isAbs = True
                except:
                    isAbs = False
                segment_src = batch.src_str
                paper_tgt = batch.tgt_str[0]
                sent_sect_labels = batch.sent_sect_labels

                if self.is_joint:
                    sent_scores, sent_sect_scores, mask = self.model(src, segs, clss, mask, mask_cls)
                    # loss_sent = self.loss(sent_scores, labels.float())
                    # loss_sent = self.loss(sent_scores, labels.float())
                    loss_sent = self.bin_loss(sent_scores, sent_bin_labels.float())
                    loss_sent = (loss_sent * mask.float()).sum()
                    loss_sect = self.loss_sect(sent_sect_scores.permute(0, 2, 1), sent_sect_labels)
                    loss_sect = (loss_sect * mask.float()).sum()
                    loss_sent = (loss_sent * mask.float()).sum()
                    loss_sent = self.alpha * loss_sent
                    loss_sect = (1 - self.alpha) * loss_sect
                    loss = loss_sent + loss_sect
                    acc, pred = self._get_mertrics(sent_sect_scores, sent_sect_labels, mask=mask,
                                                   task='sent_sect')
                    # print(acc)
                    batch_stats = Statistics(loss=float(loss.cpu().data.numpy()), loss_sect=loss_sect,
                                             loss_sent=loss_sent,
                                             n_docs=len(labels),
                                             accuracy=acc,
                                             n_acc=batch.batch_size
                                             )

                else:
                    sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)
                    # loss = self.loss(sent_scores, labels.float())
                    loss = self.bin_loss(sent_scores, sent_bin_labels.float())
                    loss = (loss * mask.float()).sum()
                    batch_stats = Statistics(loss=float(loss.cpu().data.numpy()),
                                             n_docs=len(labels))

                stats.update(batch_stats)
                sent_scores = sent_scores + mask.float()
                sent_scores = sent_scores.cpu().data.numpy()
                # for idx, p_id in enumerate(paper_id):
                #     if p_id not in sent_scores_whole.keys():
                #         s_scores = sent_scores[idx] * mask[idx].cpu().data.numpy()
                #         sent_scores_whole[p_id] = (s_scores)[s_scores > 0]
                #         paper_srcs[p_id] = segment_src[idx]
                #
                #         paper_tgts[p_id] = paper_tgt[idx]
                #     else:
                #         s_score = sent_scores[idx] * mask[idx].cpu().data.numpy()
                #         sent_scores_whole[p_id] = np.concatenate((sent_scores_whole[p_id], s_score[s_score > 0]), 0)
                #         paper_srcs[p_id] = np.concatenate((paper_srcs[p_id], segment_src[idx]), 0)
                if paper_id not in sent_scores_whole.keys():
                    sent_scores_whole[paper_id] = sent_scores
                    # if self.is_joint:
                    #     sent_sects_whole[paper_id] = torch.max(self.softmax(sent_sect_scores), 2)[1].cpu()
                    #     sent_sects_whole_true[paper_id] = sent_sect_labels.cpu()
                    paper_srcs[paper_id] = segment_src
                    paper_tgts[paper_id] = paper_tgt
                else:
                    sent_scores_whole[paper_id] = np.concatenate((sent_scores_whole[paper_id], sent_scores), 1)
                    # if self.is_joint:
                    #     sent_sects_whole[paper_id] = np.concatenate(
                    #         (sent_sects_whole[paper_id], torch.max(self.softmax(sent_sect_scores), 2)[1].cpu()), 1)
                    #     sent_sects_whole_true[paper_id] = np.concatenate(
                    #         (sent_sects_whole_true[paper_id], sent_sect_labels.cpu()), 1)

                    paper_srcs[paper_id] = np.concatenate((paper_srcs[paper_id], segment_src), 1)


        PRED_LEN = 5
        for paper_id, sent_scores in sent_scores_whole.items():

            selected_ids_unsorted = np.argsort(-sent_scores, 1)[0]
            # if dd==0:
            #     print(sent_scores)
            #     print(selected_ids)
            selected_ids_unsorted = [s for s in selected_ids_unsorted if
                                     len(paper_srcs[paper_id][0][s].split()) > 5 and len(
                                         paper_srcs[paper_id][0][s].split()) < 91]

            selected_ids_top = selected_ids_unsorted[:PRED_LEN]
            selected_ids = sorted(selected_ids_top, reverse=False)
            _pred_final = [(paper_srcs[paper_id][0][selected_ids_unsorted[0]].strip(), selected_ids_unsorted[0])]

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

            picked_up = 0
            for j in selected_ids[:len(paper_srcs[paper_id][0])]:
                if (j >= len(paper_srcs[paper_id][0])):
                    continue
                candidate = paper_srcs[paper_id][0][j].strip()
                if (self.args.block_trigram):
                    if (not _block_tri(candidate, [x[0] for x in _pred_final])):
                        _pred_final = _insert(_pred_final, candidate, j)
                        picked_up += 1
                else:
                    _pred_final.append((candidate, j))
                    picked_up += 1

            _pred_final = sorted(_pred_final, key=itemgetter(1))

            if picked_up < PRED_LEN:
                # it means that some sentences are just skept, sample from the rest of the list
                selected_ids_rest = selected_ids_unsorted[PRED_LEN:]

                for k in selected_ids_rest:
                    candidate = paper_srcs[paper_id][0][k].strip()
                    if (self.args.block_trigram):
                        if (not _block_tri(candidate, [x[0] for x in _pred_final])):
                            _pred_final = _insert(_pred_final, candidate, k)
                            picked_up += 1
                    else:
                        _pred_final.append((candidate, k))
                        picked_up += 1

                    if (picked_up == PRED_LEN):
                        break

            # NUM_PICKED_SENTS = 70
            # for paper_id, sent_scores in sent_scores_whole.items():
            #     selected_ids_unsorted_init = np.argsort(-sent_scores, 0)
            #     selected_ids_unsorted_init = [s for s in selected_ids_unsorted_init if
            #                                   len(paper_srcs[paper_id][s].split()) > 6 and
            #                                   ('contributed equally' not in paper_srcs[paper_id][s] and
            #                                    'equally contributed' not in paper_srcs[paper_id][s] and
            #                                    'equal contribution' not in paper_srcs[paper_id][s])]
            #     selected_ids_unsorted = selected_ids_unsorted_init
            #
            #     selected_ids = selected_ids_unsorted[:NUM_PICKED_SENTS]
            #     selected_ids = sorted(selected_ids, reverse=False)
            #
            #     # _pred_final = [(paper_srcs[paper_id][selected_ids_unsorted[0]].strip(), selected_ids_unsorted[0])]
            #     _pred_final = []
            #     picked_up = 1
            #     for j in selected_ids[:len(paper_srcs[paper_id])]:
            #         if j in [x[1] for x in _pred_final]:
            #             continue
            #         if (j >= len(paper_srcs[paper_id])):
            #             continue
            #         candidate = paper_srcs[paper_id][j].strip()
            #         if (self.args.block_trigram):
            #             try:
            #                 if (not _block_tri(candidate, [x[0] for x in _pred_final])):
            #                     _pred_final = _insert(_pred_final, candidate, j)
            #                     picked_up += 1
            #             except:
            #                 import pdb;pdb.set_trace()
            #         else:
            #             _pred_final.append((candidate, j))
            #             picked_up += 1
            #         # if picked_up==30:
            #         #     break
            #
            #     _pred_final = sorted(_pred_final, key=itemgetter(1))
            #
            #
            #     # if picked_up < NUM_PICKED_SENTS < len(sent_scores):
            #     #     # it means that some sentences are just skept, sample from the rest of the list
            #     #     selected_ids_rest = selected_ids_unsorted[NUM_PICKED_SENTS:]
            #     #
            #     #     for k in selected_ids_rest:
            #     #         candidate = paper_srcs[paper_id][k].strip()
            #     #         if (self.args.block_trigram):
            #     #             if (not _block_tri(candidate, [x[0] for x in _pred_final])):
            #     #                 _pred_final = _insert(_pred_final, candidate, k)
            #     #                 picked_up += 1
            #     #         else:
            #     #             _pred_final.append((candidate, k))
            #     #             picked_up += 1
            #     #
            #     #         if (picked_up == NUM_PICKED_SENTS):
            #     #             break
            #
            #     # if picked_up < NUM_PICKED_SENTS:
            #     #
            #     #     selected_idxs = [x[1] for x in _pred_final]
            #     #     for j in selected_ids_unsorted_init:
            #     #         if j in selected_idxs:
            #     #             continue
            #     #         if (j >= len(paper_srcs[paper_id])):
            #     #             continue
            #     #         candidate = paper_srcs[paper_id][j].strip()
            #     #         _pred_final = _insert(_pred_final, candidate, j)
            #     #         picked_up += 1
            #     #         if picked_up==NUM_PICKED_SENTS:
            #     #             break
            #
            #     # _pred_final = sorted(_pred_final, key=itemgetter(1))

                _pred_final_str = '<q>'.join([x[0] for x in _pred_final])
                if (self.args.recall_eval):
                    _pred = ' '.join(_pred_final_str.split()[:len(paper_tgts[paper_id].split())])

                preds[paper_id] = _pred_final_str
                golds[paper_id] = paper_tgts[paper_id]
                # preds_sorted = {}
            # for key in sorted(preds, reverse=False):
            #     preds_sorted[key] = preds[key]

            # l = list(preds.items())
            # random.seed(888)
            # random.shuffle(l)
            # preds = dict(l)
            #
            # dictionary_items = preds.items()
            # preds = sorted(dictionary_items)
            # preds = dict(preds)


            # for normal saving
            for id, pred in preds.items():
                ID = ''
                try:
                    int(id.replace('.json', '').replace('.pdf', ''))
                    ID = '[ABS]'
                except:
                    ID = '[EXT]'

                # save_pred.write(ID + ' ' + pred.replace('<q>', ' ').strip() + '\n')
                save_pred.write(pred.replace('<q>', ' ').strip() + '\n')
                save_gold.write(golds[id].replace('<q>', ' ').strip() + '\n')

            # # For official submission
            # import json
            # for id, pred in preds.items():
            #     preds[id] = pred.replace('<q>', ' ').strip()\
            #         .replace(' \u2019', "'").replace('\u2013', '–').replace(' \u201c', '“').replace(' \u201d', '”').replace('\u223c', '∼').replace('\u00d7','×').\
            #         strip()
            #
            # with open(official_pred, mode='w', encoding='utf-8') as fW:
            #     json.dump(preds, fW, indent=4, ensure_ascii=False)
        #

        print(f'Gold: {gold_path}')
        print(f'Prediction: {can_path}')

        # for paper_id, sent_scores in sent_scores_whole.items():

        # for i in range(len(gold)):
        #     save_gold.write(gold[i].strip() + '\n')
        # for i in range(len(pred)):
        #     save_pred.write(pred[i].strip() + '\n')
        # if (step != -1 and self.args.report_rouge):
        #     rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
        #     logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
        import pdb;pdb.set_trace()
        self._report_rouge(preds.values(), golds.values())
        self._report_step(0, step, valid_stats=stats)
        # with open(self.args.fold_base_dir + '/papers_id_' + self.args.exp_set +'.txt', mode='w') as F:
        #     for p in papers_list:
        #         F.write(p[0] + '\n')
        return stats

    def test_section_based(self, test_iter_fct, step, cal_lead=False, cal_oracle=False):
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

        if (not cal_lead and not cal_oracle):
            self.model.eval()
        stats = Statistics()
        preds = {}
        golds = {}
        can_path = '%s.candidate' % (self.args.result_path)
        gold_path = '%s.gold' % (self.args.result_path)

        if len(self.args.bart_dir_out.strip()) > 0:
            check_path_existence(self.args.bart_dir_out)
            can_path = ('%s/%s.candidate') % (self.args.bart_dir_out, self.args.exp_set)
            gold_path = ('%s/%s.gold') % (self.args.bart_dir_out, self.args.exp_set)
        save_pred = open(can_path, 'w')
        save_gold = open(gold_path, 'w')
        sent_scores_whole = {}
        paper_srcs = {}
        src_strs = {}
        paper_tgts = {}
        test_iter = test_iter_fct()
        counter = 0
        for b in test_iter:
            counter += 1
        test_iter = test_iter_fct()
        papers_list = []
        with torch.no_grad():
            # for batch in tqdm(test_iter_fct, total=counter):
            for batch in tqdm(test_iter, total=counter):
                src = batch.src
                labels = batch.src_sent_labels
                segs = batch.segs
                clss = batch.clss
                sent_bin_labels = batch.sent_labels
                mask = batch.mask_src
                mask_cls = batch.mask_cls
                paper_id = batch.paper_id[0]

                segment_src = batch.src_str[0]
                paper_tgt = batch.tgt_str[0]
                sent_sect_labels = batch.sent_sect_labels[0]

                if self.is_joint:
                    section_sent_scores, sent_sect_scores, mask = self.model(src, segs, clss, mask, mask_cls)
                    # loss_sent = self.loss(sent_scores, labels.float())
                    # loss_sent = self.loss(sent_scores, labels.float())
                    loss_sent = self.bin_loss(section_sent_scores, sent_bin_labels.float())
                    loss_sent = (loss_sent * mask.float()).sum()

                    loss_sect = self.loss_sect(sent_sect_scores.permute(0, 2, 1), sent_sect_labels)
                    loss_sect = (loss_sect * mask.float()).sum()
                    loss_sent = (loss_sent * mask.float()).sum()
                    loss_sent = self.alpha * loss_sent
                    loss_sect = (1 - self.alpha) * loss_sect
                    loss = loss_sent + loss_sect
                    acc, sect_pred = self._get_mertrics(sent_sect_scores, sent_sect_labels, mask=mask,
                                                        task='sent_sect')
                    # print(acc)
                    batch_stats = Statistics(loss=float(loss.cpu().data.numpy()), loss_sect=loss_sect,
                                             loss_sent=loss_sent,
                                             n_docs=len(labels),
                                             accuracy=acc,
                                             n_acc=batch.batch_size
                                             )

                else:
                    section_sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)
                    # loss = self.loss(sent_scores, labels.float())
                    loss = self.bin_loss(section_sent_scores, sent_bin_labels.float())
                    loss = (loss * mask.float()).sum()
                    batch_stats = Statistics(loss=float(loss.cpu().data.numpy()),
                                             n_docs=len(labels))

                stats.update(batch_stats)
                section_sent_scores = section_sent_scores + mask.float()
                section_sent_scores = section_sent_scores.cpu().data.numpy()[0]

                if paper_id not in sent_scores_whole.keys():
                    paper_sent_scores = section_sent_scores * mask.cpu().data.numpy()
                    sent_scores_whole[paper_id] = {}
                    sent_scores_whole[paper_id][sent_sect_labels[0]] = (paper_sent_scores)[paper_sent_scores > 0]

                    paper_srcs.setdefault(paper_id, {})[sent_sect_labels[0]] = segment_src

                    paper_tgts[paper_id] = paper_tgt

                else:

                    paper_sent_scores = section_sent_scores * mask.cpu().data.numpy()

                    if sent_sect_labels[0] not in sent_scores_whole[paper_id]:
                        sent_scores_whole[paper_id][sent_sect_labels[0]] = (paper_sent_scores)[paper_sent_scores > 0]
                        paper_srcs.setdefault(paper_id, {})[sent_sect_labels[0]] = segment_src

                    else:
                        sent_scores_whole.setdefault(paper_id, {})[sent_sect_labels[0]] = np.concatenate((
                            sent_scores_whole[
                                paper_id][
                                sent_sect_labels[
                                    0]],
                            paper_sent_scores[
                                paper_sent_scores > 0]),
                            0)
                        paper_srcs.setdefault(paper_id, {})[sent_sect_labels[0]] = np.concatenate(
                            (paper_srcs[paper_id][sent_sect_labels[0]], segment_src), 0)

            NUM_PICKED_UP = 10
            for paper_id, sent_scores_sectionized in sent_scores_whole.items():

                for section, section_sent_scores in sent_scores_sectionized.items():

                    selected_ids_unsorted_init = np.argsort(-section_sent_scores, 0)

                    selected_ids_unsorted_init = [s for s in selected_ids_unsorted_init if
                                                  len(paper_srcs[paper_id][section][s].split()) > 6 and
                                                  ('contributed equally' not in paper_srcs[paper_id][section][s] and
                                                   'equally contributed' not in paper_srcs[paper_id][section][s] and
                                                   'equal contribution' not in paper_srcs[paper_id][section][s])]

                    selected_ids_unsorted = selected_ids_unsorted_init

                    selected_ids = selected_ids_unsorted[:NUM_PICKED_UP]
                    selected_ids = sorted(selected_ids, reverse=False)

                    _pred_final_sect = []

                    picked_up = 0
                    for j in selected_ids[:len(paper_srcs[paper_id][section])]:
                        if (j >= len(paper_srcs[paper_id][section])):
                            continue
                        candidate = paper_srcs[paper_id][section][j].strip()
                        if (self.args.block_trigram):
                            if (not _block_tri(candidate, [x[0] for x in _pred_final_sect])):
                                _pred_final_sect.append((candidate, j))
                                picked_up += 1
                        else:
                            _pred_final_sect.append((candidate, j))

                    _pred_final_sect = sorted(_pred_final_sect, key=itemgetter(1))

                    if picked_up < NUM_PICKED_UP:
                        # it means that some sentences are just skept, sample from the rest of the list
                        selected_ids_rest = selected_ids_unsorted[NUM_PICKED_UP:]

                        for k in selected_ids_rest:
                            candidate = paper_srcs[paper_id][section][k].strip()
                            if (self.args.block_trigram):
                                if (not _block_tri(candidate, [x[0] for x in _pred_final_sect])):
                                    _insert(_pred_final_sect, candidate, k)
                                    picked_up += 1
                            else:
                                _pred_final_sect.append((candidate, k))

                            if (picked_up == NUM_PICKED_UP):
                                break

                    _pred_final_sect = sorted(_pred_final_sect, key=itemgetter(1))

                    _pred_final_str = '<q>'.join([x[0] for x in _pred_final_sect])
                    if (self.args.recall_eval):
                        _pred = ' '.join(_pred_final_str.split()[:len(paper_tgts[paper_id].split())])

                    preds.setdefault(paper_id, {})[section] = _pred_final_str
                    golds[paper_id] = paper_tgts[paper_id]

            for id, sect_preds in preds.items():
                for sect, pred in sect_preds.items():
                    save_pred.write(pred.replace('<q>', ' ').strip() + ' [SECT] ')
                save_pred.write('\n')
                save_gold.write(golds[id].replace('<q>', ' ').strip() + '\n')

        print(f'Gold: {gold_path}')
        print(f'Prediction: {can_path}')

        # for paper_id, sent_scores in sent_scores_whole.items():

        # for i in range(len(gold)):
        #     save_gold.write(gold[i].strip() + '\n')
        # for i in range(len(pred)):
        #     save_pred.write(pred[i].strip() + '\n')
        # if (step != -1 and self.args.report_rouge):
        #     rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
        #     logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))

        for p_id, section_preds in preds.items():
            paper_pred = ''
            for section, pred in section_preds.items():
                paper_pred += pred
            preds[p_id] = paper_pred

        self._report_rouge(preds.values(), golds.values())
        self._report_step(0, step, valid_stats=stats)
        # with open(self.args.fold_base_dir + '/papers_id_' + self.args.exp_set +'.txt', mode='w') as F:
        #     for p in papers_list:
        #         F.write(p[0] + '\n')
        return stats

    def test_ids(self, test_iter_fct, step, cal_lead=False, cal_oracle=False):
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

        if (not cal_lead and not cal_oracle):
            self.model.eval()
        stats = Statistics()
        preds = {}
        golds = {}
        selected_sent_idS = {}
        can_path = '%s_step%d.candidate' % (self.args.result_path, step)
        gold_path = '%s_step%d.gold' % (self.args.result_path, step)
        save_pred = open(can_path, 'w')
        save_gold = open(gold_path, 'w')
        sent_scores_whole = {}
        paper_srcs = {}
        paper_tgts = {}
        test_iter = test_iter_fct()
        counter = 0
        for b in test_iter:
            counter += 1

        test_iter = test_iter_fct()

        with torch.no_grad():
            for batch in tqdm(test_iter, total=counter):
                src = batch.src
                labels = batch.src_sent_labels
                segs = batch.segs
                clss = batch.clss
                sent_bin_labels = batch.sent_labels
                mask = batch.mask_src
                mask_cls = batch.mask_cls
                paper_id = batch.paper_id[0]
                segment_src = batch.src_str
                paper_tgt = batch.tgt_str[0]
                sent_sect_labels = batch.sent_sect_labels

                if self.is_joint:
                    sent_scores, sent_sect_scores, mask = self.model(src, segs, clss, mask, mask_cls)
                    # loss_sent = self.loss(sent_scores, labels.float())
                    # loss_sent = self.loss(sent_scores, labels.float())
                    loss_sent = self.bin_loss(sent_scores, sent_bin_labels.float())
                    loss_sent = (loss_sent * mask.float()).sum()

                    loss_sect = self.loss_sect(sent_sect_scores.permute(0, 2, 1), sent_sect_labels)
                    loss_sect = (loss_sect * mask.float()).sum()
                    loss_sent = (loss_sent * mask.float()).sum()
                    loss_sent = self.alpha * loss_sent
                    loss_sect = (1 - self.alpha) * loss_sect
                    loss = loss_sent + loss_sect
                    acc, pred = self._get_mertrics(sent_sect_scores, sent_sect_labels, mask=mask,
                                                   task='sent_sect')
                    # print(acc)
                    batch_stats = Statistics(loss=float(loss.cpu().data.numpy()), loss_sect=loss_sect,
                                             loss_sent=loss_sent,
                                             n_docs=len(labels),
                                             accuracy=acc,
                                             n_acc=batch.batch_size
                                             )

                else:
                    sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)
                    # loss = self.loss(sent_scores, labels.float())
                    loss = self.bin_loss(sent_scores, sent_bin_labels.float())
                    loss = (loss * mask.float()).sum()
                    batch_stats = Statistics(loss=float(loss.cpu().data.numpy()),
                                             n_docs=len(labels))

                stats.update(batch_stats)
                sent_scores = sent_scores + mask.float()
                sent_scores = sent_scores.cpu().data.numpy()

                if paper_id not in sent_scores_whole.keys():
                    sent_scores_whole[paper_id] = sent_scores
                    paper_srcs[paper_id] = segment_src
                    paper_tgts[paper_id] = paper_tgt
                else:
                    try:
                        sent_scores_whole[paper_id] = np.concatenate((sent_scores_whole[paper_id], sent_scores), 1)
                    except:
                        import pdb;
                        pdb.set_trace()
                    paper_srcs[paper_id] = np.concatenate((paper_srcs[paper_id], segment_src), 1)

            for paper_id, sent_scores in sent_scores_whole.items():
                selected_ids = np.argsort(-sent_scores, 1)[0]
                _pred = []
                selected_ids = selected_ids[:60]
                selected_ids = sorted(selected_ids, reverse=False)
                # for j in selected_ids:
                #     candidate = paper_srcs[paper_id][0][j].strip()
                #     if (self.args.block_trigram):
                #         if (not _block_tri(candidate, _pred)):
                #             _pred.append(candidate)
                #     else:
                #         _pred.append(candidate)

                # if (len(_pred) == 50):
                #     break

                # _pred = '<q>'.join(_pred)
                #
                # if (self.args.recall_eval):
                #     _pred = ' '.join(_pred.split()[:len(paper_tgts[paper_id].split())])
                #
                # preds[paper_id] = _pred
                # golds[paper_id] = paper_tgts[paper_id]
                selected_sent_idS[paper_id] = selected_ids

            # for id, pred in preds.items():
            #     save_pred.write(pred.strip() + '\n')
            #     save_gold.write(golds[id].strip() + '\n')

            with open('selected_ids_test.pkl', 'wb') as output:
                pickle.dump(selected_sent_idS, output)

        # print(f'Gold: {gold_path}')
        # print(f'Prediction: {can_path}')

        # for paper_id, sent_scores in sent_scores_whole.items():

        # for i in range(len(gold)):
        #     save_gold.write(gold[i].strip() + '\n')
        # for i in range(len(pred)):
        #     save_pred.write(pred[i].strip() + '\n')
        # if (step != -1 and self.args.report_rouge):
        #     rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
        #     logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
        # self._report_rouge(preds.values(), golds.values())
        self._report_step(0, step, valid_stats=stats)

        return stats

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
                sent_scores, sent_sect_scores, mask, loss, loss_sent, loss_sect = self.model(src, segs, clss, mask, mask_cls, sent_bin_labels, sent_sect_labels)
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
                                         accuracy=acc)


            else:  # simple

                sent_scores, mask, loss, _, _ = self.model(src, segs, clss, mask, mask_cls, sent_bin_labels=sent_bin_labels, sent_sect_labels=None)
                # loss = self.loss(sent_scores, sent_rg_scores.float())


                batch_stats = Statistics(loss=float(loss.cpu().data.numpy().sum()),
                                         RMSE=self._get_mertrics(sent_scores, sent_rg_scores, mask=mask,
                                                                 task='sent'),
                                         n_acc=batch.batch_size,
                                         n_docs=normalization)

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
        r1, r2, rl = evaluate_rouge(predictions, references)
        # r1, r2, rl = get_rouge_pap(predictions, references)

        if len(self.args.log_folds) > 0:
            with open(self.args.log_folds, mode='a') as f:
                f.write("{:.4f}\t{:.4f}\t{:.4f}".format(r1 / 100, r2 / 100, rl / 100))
                f.write('\n')
        logger.info("Metric\tScore\t95% CI")
        logger.info("ROUGE-1\t{:.2f}\t({:.2f},{:.2f})".format(r1, 0, 0))
        logger.info("ROUGE-2\t{:.2f}\t({:.2f},{:.2f})".format(r2, 0, 0))
        logger.info("ROUGE-L\t{:.2f}\t({:.2f},{:.2f})".format(rl, 0, 0))
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


    def _get_sect_accurracy(self, pred, true):

        acc_out = [0, 0, 0, 0, 0, 0]

        for p, t in zip(pred[0], true[0]):

            if p == t:
                acc_out[0] += 1
            if p == 0 and t == 0:
                acc_out[1] += 1
            if p == 1 and t == 1:
                acc_out[2] += 1
            if p == 2 and t == 2:
                acc_out[3] += 1
            if p == 3 and t == 3:
                acc_out[4] += 1
            if p == 4 and t == 4:
                acc_out[5] += 1

        acc_outy = []
        count = []
        try:
            acc_outy.append(acc_out[0] / len(true[0]))
        except:
            acc_outy.append(0)

        try:
            acc_outy.append(acc_out[1] / np.count_nonzero(true[0] == 0))
            count.append(1)

        except:
            acc_outy.append(0)
            count.append(0)

        try:
            acc_outy.append(acc_out[2] / np.count_nonzero(true[0] == 1))
            count.append(1)

        except:
            acc_outy.append(0)
            count.append(0)

        try:
            acc_outy.append(acc_out[3] / np.count_nonzero(true[0] == 2))
            count.append(1)

        except:
            acc_outy.append(0)
            count.append(0)

        # try:
        #     acc_outy.append(acc_out[4] / np.count_nonzero(true[0] == 3))
        #     count.append(1)
        #
        # except:
        #     acc_outy.append(0)
        #     count.append(0)
        #
        # try:
        #     acc_outy.append(acc_out[5] / np.count_nonzero(true[0] == 4))
        #     count.append(1)
        #
        # except:
        #     acc_outy.append(0)
        #     count.append(0)

        return acc_outy, count

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

def _get_ir_eval_metrics(preds_with_idx, sent_labels_true):
    avg_scores = {'f':[], 'p':[], 'r':[]}
    for p_id, pred_with_idx in preds_with_idx.items():
        retrieved_idx = [pred[1] for pred in pred_with_idx]
        retrieved_true_labels = [sent_labels_true[p_id][idx] for idx in retrieved_idx]
        avg_scores['p'].append(retrieved_true_labels.count(1) / 10)
    return np.mean(avg_scores['p'])
