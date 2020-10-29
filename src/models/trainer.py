import os
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

import distributed
from models.predictor import build_predictor
from models.reporter import ReportMgr, Statistics
from others.logging import logger
# from others.tokenization import BertTokenizer
from transformers import BertTokenizer
from others.utils import test_rouge, rouge_results_to_str
# import utils.rouge
from utils.rouge_score import evaluate_rouge


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, device_id, model, optims, loss):
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
    device = "cpu" if args.visible_gpus == '-1' else "cuda"


    grad_accum_count = args.accum_count
    n_gpu = args.world_size

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    print('gpu_rank %d' % gpu_rank)

    tensorboard_log_dir = args.model_path

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)


    trainer = Trainer(args, model, optims, loss, grad_accum_count, n_gpu, gpu_rank, report_manager)

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

    def __init__(self,  args, model,  optims, loss,
                  grad_accum_count=1, n_gpu=1, gpu_rank=1,
                  report_manager=None):
        # Basic attributes.
        self.valid_rgls = []
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optims = optims
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager
        self.valid_accuracies = []
        self.max_rl = 0
        self.loss = loss
        self.last_best_step = []
        # self.tokenizer = BertTokenizer.from_pretrained('/disk1/sajad/pretrained-bert/scibert_scivocab_uncased', do_lower_case=True)
        self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

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
        step =  self.optims[0]._step + 1

        true_batchs = []
        accum = 0
        normalization = 0
        train_iter = train_iter_fct()

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:
            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):
                    true_batchs.append(batch)
                    num_tokens = batch.tgt[:, 1:].ne(self.loss.padding_idx).sum()
                    normalization += num_tokens.item()
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
                            self.optims[0].learning_rate,
                            report_stats)
                        true_batchs = []
                        accum = 0
                        normalization = 0
                        if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            self._save(step)

                        if (step % self.args.val_interval == 0):
                            _, is_best = self.validate_rouge(valid_iter_fct, step)
                            if is_best:
                                self._save(step)
                                logger.info('Best model saved at step %d' % step)
                                if len(self.last_best_step) > 1:
                                    self._delete()
                            self.model.train()
                        step += 1
                        if step > train_steps:
                            break
            train_iter = train_iter_fct()

        return total_stats

    def _report_rouge(self, predictions, references):
        r1, r2, rl = evaluate_rouge(predictions, references)
        # print("{} set results:\n".format(args.filename))
        logger.info("Metric\tScore\t95% CI")
        logger.info("ROUGE-1\t{:.2f}\t({:.2f},{:.2f})".format(r1, 0, 0))
        logger.info("ROUGE-2\t{:.2f}\t({:.2f},{:.2f})".format(r2, 0, 0))
        logger.info("ROUGE-L\t{:.2f}\t({:.2f},{:.2f})".format(rl, 0, 0))
        return r1, r2, rl

    def validate(self, valid_iter_fct, step=0):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()
        valid_iter = valid_iter_fct()
        cn = 0
        for _ in valid_iter:
            cn+=1
        best_model_saved = False
        valid_iter = valid_iter_fct()
        with torch.no_grad():
            for batch in tqdm(valid_iter, total=cn):
                src = batch.src
                tgt = batch.tgt
                segs = batch.segs
                clss = batch.clss
                mask_src = batch.mask_src
                mask_tgt = batch.mask_tgt
                mask_cls = batch.mask_cls

                outputs, _ = self.model(src, tgt, segs, clss, mask_src, mask_tgt, mask_cls)

                batch_stats = self.loss.monolithic_compute_loss(batch, outputs)
                stats.update(batch_stats)

            self.valid_rgls.append(stats.accuracy())
            if len(self.valid_rgls) > 0:
                if self.max_rl < self.valid_rgls[-1]:
                    self.max_rl = self.valid_rgls[-1]
                    best_model_saved = True

            self._report_step(0, step, valid_stats=stats)
            return stats, best_model_saved

    def validate_rouge(self, valid_iter, step=0):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()

        preds, golds = self.val_abs(self.args, valid_iter, step)
        best_model_saved = False
        preds_sorted = [s[1] for s in sorted(preds.items())]
        golds_sorted = [g[1] for g in sorted(golds.items())]
        logger.info('Some samples...')
        logger.info('[' + preds_sorted[random.randint(0, len(preds_sorted)-1)] + ']')
        logger.info('[' + preds_sorted[random.randint(0, len(preds_sorted)-1)] + ']')
        logger.info('[' + preds_sorted[random.randint(0, len(preds_sorted)-1)] + ']')
        r1, r2, rl = self._report_rouge(preds_sorted, golds_sorted)

        stats.set_rl(r1, r2, rl)
        self.valid_rgls.append(rl)
        # self._report_step(0, step, valid_stats=stats)

        if len(self.valid_rgls) > 0:
            if self.max_rl < self.valid_rgls[-1]:
                self.max_rl = self.valid_rgls[-1]
                best_model_saved = True

        # with torch.no_grad():
        #     for batch in valid_iter:
        #         src = batch.src
        #         tgt = batch.tgt
        #         segs = batch.segs
        #         clss = batch.clss
        #         mask_src = batch.mask_src
        #         mask_tgt = batch.mask_tgt
        #         mask_cls = batch.mask_cls
        #
        #         outputs, _ = self.model(src, tgt, segs, clss, mask_src, mask_tgt, mask_cls)
        #
        #         batch_stats = self.loss.monolithic_compute_loss(batch, outputs)
        #         stats.update(batch_stats)
        #     self._report_step(0, step, valid_stats=stats)
        #
        #     # if len(self.valid_accuracies) > 0:
        #     if self.best_acc < stats.accuracy():
        #         is_best = True
        #         self.best_acc = stats.accuracy()
        #         self.last_best_step.append(step)
            # self.valid_accuracies.append(stats.accuracy())
            return stats, best_model_saved

    def val_abs(self, args, iter_fct, step):
        self.model.eval()

        symbols = {'BOS': self.tokenizer.vocab['[unused0]'], 'EOS': self.tokenizer.vocab['[unused1]'],
                   'PAD': self.tokenizer.vocab['[PAD]'], 'EOQ': self.tokenizer.vocab['[unused2]']}
        predictor = build_predictor(args, self.tokenizer, symbols, self.model, logger)
        return predictor.translate(iter_fct, step, return_entities=True)

    def _gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()
            src = batch.src
            tgt = batch.tgt
            segs = batch.segs
            clss = batch.clss
            mask_src = batch.mask_src
            mask_tgt = batch.mask_tgt
            mask_cls = batch.mask_cls
            outputs, attn, scores = self.model(src, tgt, segs, clss, mask_src, mask_tgt, mask_cls)
            import pdb;pdb.set_trace()
            batch_stats = self.loss.sharded_compute_loss(batch, attn, self.args.generator_shard_size, normalization, optim=self.optims[0])

            batch_stats.n_docs = int(src.size(0))

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

                for optim in self.optims:
                    optim.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            for optim in self.optims:
                optim.step()


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
                if len(tri_c.intersection(tri_s))>0:
                    return True
            return False

        if (not cal_lead and not cal_oracle):
            self.model.eval()
        stats = Statistics()

        can_path = '%s_step%d.candidate'%(self.args.result_path,step)
        gold_path = '%s_step%d.gold' % (self.args.result_path, step)
        with open(can_path, 'w') as save_pred:
            with open(gold_path, 'w') as save_gold:
                with torch.no_grad():
                    for batch in test_iter:
                        gold = []
                        pred = []
                        if (cal_lead):
                            selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
                        for i, idx in enumerate(selected_ids):
                            _pred = []
                            if(len(batch.src_str[i])==0):
                                continue
                            for j in selected_ids[i][:len(batch.src_str[i])]:
                                if(j>=len( batch.src_str[i])):
                                    continue
                                candidate = batch.src_str[i][j].strip()
                                _pred.append(candidate)

                                if ((not cal_oracle) and (not self.args.recall_eval) and len(_pred) == 3):
                                    break

                            _pred = '<q>'.join(_pred)
                            if(self.args.recall_eval):
                                _pred = ' '.join(_pred.split()[:len(batch.tgt_str[i].split())])

                            pred.append(_pred)
                            gold.append(batch.tgt_str[i])

                        for i in range(len(gold)):
                            save_gold.write(gold[i].strip()+'\n')
                        for i in range(len(pred)):
                            save_pred.write(pred[i].strip()+'\n')
        if(step!=-1 and self.args.report_rouge):
            rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
            logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
        self._report_step(0, step, valid_stats=stats)

        return stats

    def _save(self, step):
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
            'optims': self.optims,
        }
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
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1, is_train=True)

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

    def _delete(self):
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % self.last_best_step[-2])
        os.remove(checkpoint_path)

