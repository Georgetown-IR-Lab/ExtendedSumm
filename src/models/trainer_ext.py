import os

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

import distributed
import utils.rouge
from models.reporter_ext import ReportMgr, Statistics
from others.logging import logger


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
        # self.loss = torch.nn.BCELoss(reduction='none')
        self.loss = torch.nn.MSELoss(reduction='none')
        self.loss_sect = torch.nn.CrossEntropyLoss(reduction='none')
        self.min_val_loss = 100000
        self.min_rl = -100000
        self.softmax = nn.Softmax(dim=1)
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

                        # if step == 1:  # Validation
                        #     logger.info('----------------------------------------')
                        #     logger.info('Start evaluating on evaluation set... ')
                        #     val_stat, best_model_save = self.validate_rouge(valid_iter_fct, step,
                        #                                                     valid_gl_stats=valid_global_stats)
                        #     if best_model_save:
                        #         self._save(step, best=True)
                        #         logger.info(f'Best model saved sucessfully at step %d' % step)
                        #     logger.info('----------------------------------------')

                        self._gradient_accumulation(
                            true_batchs, normalization, total_stats,
                            report_stats)

                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optim.learning_rate,
                            report_stats)

                        self._report_step(self.optim.learning_rate, step, train_stats=report_stats)

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        sent_num_normalization = 0
                        if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            self._save(step)

                        if step % 2500 == 0:  # Validation
                            logger.info('----------------------------------------')
                            logger.info('Start evaluating on evaluation set... ')
                            val_stat, best_model_save = self.validate_rouge(valid_iter_fct, step,
                                                                            valid_gl_stats=valid_global_stats)
                            if best_model_save:
                                self._save(step, best=True)
                                logger.info(f'Best model saved sucessfully at step %d' % step)
                            logger.info('----------------------------------------')
                            self.model.train()

                        # if step % 6000 == 0:
                        #     self.alpha_decacy(step)
                        #     logger.info((f'Alpha degraded to %4.2f at step %d') % (self.alpha, step))

                        step += 1
                        if step > train_steps:
                            break
            train_iter = train_iter_fct()

        return total_stats

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

        preds = {}
        golds = {}
        can_path = '%s_step%d.candidate' % (self.args.result_path, step)
        gold_path = '%s_step%d.gold' % (self.args.result_path, step)
        save_pred = open(can_path, 'w')
        save_gold = open(gold_path, 'w')
        sent_scores_whole = {}
        paper_srcs = {}
        paper_tgts = {}

        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()
        best_model_saved = False
        valid_iter = valid_iter_fct()

        with torch.no_grad():
            for batch in valid_iter:
                src = batch.src
                labels = batch.src_sent_labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask_src
                mask_cls = batch.mask_cls
                paper_id = batch.paper_id[0]
                segment_src = batch.src_str
                paper_tgt = batch.tgt_str[0]
                sent_sect_labels = batch.sent_sect_labels

                if self.is_joint:
                    sent_scores, sent_sect_scores, mask = self.model(src, segs, clss, mask, mask_cls)

                    loss_sent = self.loss(sent_scores, labels.float())
                    loss_sect = self.loss_sect(sent_sect_scores.permute(0, 2, 1), sent_sect_labels)

                    # loss_sent = torch.sum((loss_sent * mask.float()), dim=1)
                    loss_sent = (loss_sent * mask.float()).sum()
                    # loss_sent = (loss_sent / torch.sum(mask, dim=1).float())
                    # loss_sent = loss_sent.float().mean()

                    # loss_sect = (loss_sect * mask.float()).sum(dim=1)
                    loss_sect = (loss_sect * mask.float()).sum()
                    # loss_sect = (loss_sect / torch.sum(mask, dim=1).float()).mean()

                    # loss_sent = self.alpha * loss_sent
                    # loss_sect = (1 - self.alpha) * loss_sect

                    loss_sent = loss_sent
                    loss_sect = loss_sect

                    # loss = loss_sent + loss_sect
                    loss = loss_sect

                    batch_stats = Statistics(loss=float(loss.cpu().data.numpy().sum()),
                                             loss_sect=float(loss_sect.cpu().data.numpy().sum()),
                                             loss_sent=float(loss_sent.cpu().data.numpy().sum()),
                                             n_docs=len(labels),
                                             n_acc=batch.batch_size,
                                             RMSE=self._get_mertrics(sent_scores, labels, mask=mask, task='sent'),
                                             accuracy=self._get_mertrics(sent_sect_scores, sent_sect_labels, mask=mask,
                                                                       task='sent_sect'))

                else:
                    sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

                    loss = self.loss(sent_scores, labels.float())

                    loss = (loss * mask.float()).sum()

                    batch_stats = Statistics(loss=float(loss.cpu().data.numpy().sum()),
                                             RMSE=self._get_mertrics(sent_scores, labels, mask=mask, task='sent'),
                                             n_acc=batch.batch_size,
                                             n_docs=len(labels))

                stats.update(batch_stats)

                sent_scores = sent_scores + mask.float()
                sent_scores = sent_scores.cpu().data.numpy()

                if paper_id not in sent_scores_whole.keys():
                    sent_scores_whole[paper_id] = sent_scores
                    paper_srcs[paper_id] = segment_src
                    paper_tgts[paper_id] = paper_tgt
                else:
                    sent_scores_whole[paper_id] = np.concatenate((sent_scores_whole[paper_id], sent_scores), 1)
                    paper_srcs[paper_id] = np.concatenate((paper_srcs[paper_id], segment_src), 1)

            for paper_id, sent_scores in sent_scores_whole.items():
                selected_ids = np.argsort(-sent_scores, 1)[0]
                _pred = []

                for j in selected_ids:
                    candidate = paper_srcs[paper_id][0][j].strip()
                    if (self.args.block_trigram):
                        if (not _block_tri(candidate, _pred)):
                            _pred.append(candidate)
                    else:
                        _pred.append(candidate)

                    if (len(_pred) == 4):
                        break

                _pred = '<q>'.join(_pred)
                if (self.args.recall_eval):
                    _pred = ' '.join(_pred.split()[:len(paper_tgts[paper_id].split())])

                preds[paper_id] = _pred
                golds[paper_id] = paper_tgts[paper_id]

        for id, pred in preds.items():
            save_pred.write(pred.strip() + '\n')
            save_gold.write(golds[id].strip() + '\n')

        r1, r2, rl = self._report_rouge(preds.values(), golds.values())
        stats.set_rl(r1, r2, rl)
        valid_gl_stats._write_file(step, stats, r1, r2, rl)
        self.valid_rgls.append(rl)
        self._report_step(0, step, valid_stats=stats)

        if len(self.valid_rgls) > 0:
            if self.min_rl < self.valid_rgls[-1]:
                self.min_rl = self.valid_rgls[-1]
                best_model_saved = True

        return stats, best_model_saved

    # def validate(self, valid_iter_fct, step=0):
    #     """ Validate model.
    #         valid_iter: validate data iterator
    #     Returns:
    #         :obj:`nmt.Statistics`: validation loss statistics
    #     """
    #     # Set model in validating mode.
    #     self.model.eval()
    #     stats = Statistics()
    #     best_model_saved = False
    #     valid_iter = valid_iter_fct()
    #     with torch.no_grad():
    #         for batch in valid_iter:
    #             src = batch.src
    #             labels = batch.src_sent_labels
    #             segs = batch.segs
    #             clss = batch.clss
    #             mask = batch.mask_src
    #             mask_cls = batch.mask_cls
    #             sent_sect_labels = batch.sent_sect_labels
    #
    #             if self.is_joint:
    #                 sent_scores, sent_sect_scores, mask = self.model(src, segs, clss, mask, mask_cls)
    #                 loss_sent = self.loss(sent_scores, labels.float())
    #                 loss_sect = self.loss_sect(sent_sect_scores.permute(0,2,1), sent_sect_labels)
    #                 # for deep_slice in range(sent_sect_scores.size(1)):
    #                 #     loss_sect += self.loss_sect(sent_sect_scores[:, deep_slice, :], sent_sect_labels[:, deep_slice])
    #                 loss_sect = (loss_sect * mask.float()).sum()
    #                 loss_sent = (loss_sent * mask.float()).sum()
    #                 loss_sent = self.alpha * loss_sent
    #                 loss_sect = (1 - self.alpha) * loss_sect
    #                 loss = loss_sent + loss_sect
    #
    #                 # print(float(loss.cpu().data.numpy()))
    #                 batch_stats = Statistics(loss=float(loss.cpu().data.numpy()),
    #                                          loss_sect=float(loss_sect.cpu().data.numpy()),
    #                                          loss_sent=float(loss_sent.cpu().data.numpy()), n_docs=len(labels),
    #                                          print_traj=True)
    #
    #
    #             else:  # not joint
    #                 sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)
    #                 loss = self.loss(sent_scores, labels.float())
    #                 loss = (loss * mask.float()).sum()
    #                 batch_stats = Statistics(float(loss.cpu().data.numpy()), n_docs=len(labels))
    #             stats.update(batch_stats)
    #
    #         self.valid_trajectories.append(stats.xent())
    #         if self.valid_trajectories[-1] < self.min_val_loss:
    #             self.min_val_loss = self.valid_trajectories[-1]
    #             best_model_saved = True
    #         self._report_step(self.optim.learning_rate, step, valid_stats=stats)
    #         return stats, best_model_saved

    def test(self, test_iter, step, cal_lead=False, cal_oracle=False):
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
        can_path = '%s_step%d.candidate' % (self.args.result_path, step)
        gold_path = '%s_step%d.gold' % (self.args.result_path, step)
        save_pred = open(can_path, 'w')
        save_gold = open(gold_path, 'w')
        sent_scores_whole = {}
        paper_srcs = {}
        paper_tgts = {}

        with torch.no_grad():
            for batch in test_iter:
                src = batch.src
                labels = batch.src_sent_labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask_src
                mask_cls = batch.mask_cls
                paper_id = batch.paper_id[0]
                segment_src = batch.src_str
                paper_tgt = batch.tgt_str[0]
                sent_sect_labels = batch.sent_sect_labels

                if self.is_joint:
                    sent_scores, sent_sect_scores, mask = self.model(src, segs, clss, mask, mask_cls)
                    loss_sent = self.loss(sent_scores, labels.float())
                    loss_sect = self.loss_sect(sent_sect_scores.permute(0, 2, 1), sent_sect_labels)
                    loss_sect = (loss_sect * mask.float()).sum()
                    loss_sent = (loss_sent * mask.float()).sum()
                    loss_sent = self.alpha * loss_sent
                    loss_sect = (1 - self.alpha) * loss_sect
                    loss = loss_sent + loss_sect
                    batch_stats = Statistics(loss=float(loss.cpu().data.numpy()), loss_sect=loss_sect,
                                             loss_sent=loss_sent,
                                             n_docs=len(labels))

                else:
                    sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)
                    loss = self.loss(sent_scores, labels.float())
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
                    sent_scores_whole[paper_id] = np.concatenate((sent_scores_whole[paper_id], sent_scores), 1)
                    paper_srcs[paper_id] = np.concatenate((paper_srcs[paper_id], segment_src), 1)

            for paper_id, sent_scores in sent_scores_whole.items():
                selected_ids = np.argsort(-sent_scores, 1)[0]
                # selected_ids = np.sort(selected_ids,1)
                # for i, idx in enumerate(selected_ids):
                _pred = []

                for j in selected_ids:
                    candidate = paper_srcs[paper_id][0][j].strip()
                    if (self.args.block_trigram):
                        if (not _block_tri(candidate, _pred)):
                            _pred.append(candidate)
                    else:
                        _pred.append(candidate)

                    if (len(_pred) == 4):
                        break

                _pred = '<q>'.join(_pred)
                if (self.args.recall_eval):
                    _pred = ' '.join(_pred.split()[:len(paper_tgts[paper_id].split())])

                # pred.append(_pred)
                # if paper_id in preds.keys():
                preds[paper_id] = _pred
                # else:
                #     preds[paper_id] = _pred + ' '
                golds[paper_id] = paper_tgts[paper_id]

                # gold.append(batch.tgt_str[i])
        for id, pred in preds.items():
            save_pred.write(pred.strip() + '\n')
            save_gold.write(golds[id].strip() + '\n')
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
        self._report_rouge(preds.values(), golds.values())
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
            labels = batch.src_sent_labels
            sent_sect_labels = batch.sent_sect_labels
            segs = batch.segs
            clss = batch.clss
            mask = batch.mask_src
            mask_cls = batch.mask_cls
            if self.is_joint:
                sent_scores, sent_sect_scores, mask = self.model(src, segs, clss, mask, mask_cls)
                loss_sent = self.loss(sent_scores, labels.float())
                loss_sent = (loss_sent * mask.float()).sum(dim=1)
                loss_sent = (loss_sent / loss_sent.numel()).sum()

                # loss_sent = torch.sum((loss_sent * mask.float()), dim=1)
                # loss_sent = (loss_sent / torch.sum(mask, dim=1).float())
                # loss_sent = loss_sent.float().mean()

                loss_sect = self.loss_sect(sent_sect_scores.permute(0, 2, 1), sent_sect_labels)
                # loss_sect = (loss_sect * mask.float()).sum()
                # loss_sect = (loss_sect * mask.float()).sum(dim=1)
                loss_sect = (loss_sect * mask.float()).sum()
                # loss_sent = (loss_sent / torch.sum(mask, dim=1).float()).sum()

                # loss_sect = (loss_sect / loss_sect.numel()).sum()

                # loss_sect = (loss_sect / torch.sum(mask, dim=1).float()).mean()

                # L2 reg
                # reg = 0
                # for param in self.model.section_predictor.parameters():
                #     reg += 0.5 * (param ** 2).sum()

                loss_sent = self.alpha * loss_sent
                loss_sect = (1 - self.alpha) * (loss_sect)
                loss = loss_sect
                # loss = loss_sent + loss_sect
                acc = self._get_mertrics(sent_sect_scores, sent_sect_labels, mask=mask,
                                                                   task='sent_sect')

                batch_stats = Statistics(loss=float(loss.cpu().data.numpy().sum()),
                                         loss_sect=float(loss_sect.cpu().data.numpy().sum()),
                                         loss_sent=float(loss_sent.cpu().data.numpy().sum()), n_docs=normalization,
                                         n_acc=batch.batch_size,
                                         RMSE=self._get_mertrics(sent_scores, labels, mask=mask, task='sent'),
                                         accuracy=acc)


            else:  # simple
                sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)
                loss = self.loss(sent_scores, labels.float())

                loss = torch.sum((loss * mask.float()), dim=1)  # loss for each batch
                loss = (loss / torch.sum(mask, dim=1).float())
                loss = loss.float().sum()

                batch_stats = Statistics(loss=float(loss.cpu().data.numpy().sum()),
                                         RMSE=self._get_mertrics(sent_scores, labels, mask=mask, task='sent'),
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
                self.optim.step()

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

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats, is_joint=self.is_joint,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)

    def _report_rouge(self, predictions, references):
        r1, r2, rl, r1_cf, r2_cf, rl_cf = utils.rouge.get_rouge(predictions, references, use_cf=True)
        # print("{} set results:\n".format(args.filename))
        print("Metric\tScore\t95% CI")
        print("ROUGE-1\t{:.2f}\t({:.2f},{:.2f})".format(r1, r1_cf[0] - r1, r1_cf[1] - r1))
        print("ROUGE-2\t{:.2f}\t({:.2f},{:.2f})".format(r2, r2_cf[0] - r2, r2_cf[1] - r2))
        print("ROUGE-L\t{:.2f}\t({:.2f},{:.2f})".format(rl, rl_cf[0] - rl, rl_cf[1] - rl))
        return r1, r2, rl

    def _get_mertrics(self, sent_scores, labels, mask=None, task='sent_sect'):

        labels = labels.to('cuda')
        sent_scores = sent_scores.to('cuda')
        mask = mask.to('cuda')

        def _compute_f1(labels, pred):
            tp = ((labels * pred).long() * mask.long()).sum(dim=1).to(dtype=torch.float)
            fp = (((1 - labels) * (pred)).long() * mask.long()).sum(dim=1).to(dtype=torch.float)
            fn = (((labels) * (1 - pred)).long() * mask.long()).sum(dim=1).to(dtype=torch.float)
            tn = (((1-labels) * (1 - pred)).long() * mask.long()).sum(dim=1).to(dtype=torch.float)
            # epsilon = 1e-7
            # precision = tp / (tp + fp + epsilon)
            # recall = tp / (tp + fn + epsilon)
            # f1 = 2 * (precision * recall) / (precision + recall + epsilon)
            return tp, fp, fn, tn

        if task == 'sent_sect':
            sent_scores = self.softmax(sent_scores)
            pred = torch.max(sent_scores, 2)[1]
            acc = (((pred == labels) * mask.cuda()).sum(dim=1)).to(dtype=torch.float) / \
                  mask.sum(dim=1).to(dtype=torch.float)
            return acc.sum().item()

            # pos = torch.ones(labels.shape[0], labels.shape[1]).to('cuda')
            # neg = torch.zeros(labels.shape[0], labels.shape[1]).to('cuda')
            # # 0 as positive: wherever 0-->1, others 0; pass in labels and preds
            # f10, f11, f12, f13, f14 = -1, -1, -1, -1, -1
            # if 0 in labels:
            #     f10 = _compute_f1(torch.where(labels == 0, pos, neg), torch.where(pred == 0, pos, neg))
            # if 1 in labels:
            #     f11 = _compute_f1(torch.where(labels == 1, pos, neg), torch.where(pred == 1, pos, neg))
            # if 2 in labels:
            #     f12 = _compute_f1(torch.where(labels == 2, pos, neg), torch.where(pred == 2, pos, neg))
            # if 3 in labels:
            #     f13 = _compute_f1(torch.where(labels == 3, pos, neg), torch.where(pred == 3, pos, neg))
            # if 4 in labels:
            #     f14 = _compute_f1(torch.where(labels == 4, pos, neg), torch.where(pred == 4, pos, neg))
            #
            # # return {'f0': (f10.sum().item(), sent_scores.size(0)) if f10 != -1 else -1,
            # #         'f1': (f11.sum().item(), sent_scores.size(0)) if f11 != -1 else -1,
            # #         'f2': (f12.sum().item(), sent_scores.size(0)) if f12 != -1 else -1,
            # #         'f3': (f13.sum().item(), sent_scores.size(0)) if f13 != -1 else -1,
            # #         'f4': (f14.sum().item(), sent_scores.size(0)) if f14 != -1 else -1}
            # return {'f0': (1,1),
            #         'f1': (1,1),
            #         'f2': (1,1),
            #         'f3': (1,1),
            #         'f4': (1,1)}



        else:
            mseLoss = self.loss(sent_scores.float(), labels.float())
            mseLoss = (mseLoss.float() * mask.float()).sum(dim=1)

            return mseLoss.sum().item()

    def alpha_decacy(self, step):
        alpha0 = .95
        self.alpha = alpha0 - (alpha0 * (.75 ** (152000 / step)))
