""" Report manager utility """
from __future__ import print_function

import math
import os
import sys
import time
from datetime import datetime

from others.logging import logger


def build_report_manager(opt):
    if opt.tensorboard:
        from tensorboardX import SummaryWriter
        tensorboard_log_dir = opt.tensorboard_log_dir

        if not opt.train_from:
            tensorboard_log_dir += datetime.now().strftime("/%b-%d_%H-%M-%S")

        writer = SummaryWriter(tensorboard_log_dir,
                               comment="Unmt")
    else:
        writer = None

    report_mgr = ReportMgr(opt.report_every, start_time=-1,
                           tensorboard_writer=writer)
    return report_mgr


class ReportMgrBase(object):
    """
    Report Manager Base class
    Inherited classes should override:
        * `_report_training`
        * `_report_step`
    """

    def __init__(self, report_every, start_time=-1.):
        """
        Args:
            report_every(int): Report status every this many sentences
            start_time(float): manually set report start time. Negative values
                means that you will need to set it later or use `start()`
        """
        self.report_every = report_every
        self.progress_step = 0
        self.start_time = start_time

    def start(self):
        self.start_time = time.time()

    def log(self, *args, **kwargs):
        logger.info(*args, **kwargs)

    def report_training(self, step, num_steps, learning_rate,
                        report_stats, is_joint=False, multigpu=False):
        """
        This is the user-defined batch-level traing progress
        report function.

        Args:
            step(int): current step count.
            num_steps(int): total number of batches.
            learning_rate(float): current learning rate.
            report_stats(Statistics): old Statistics instance.
        Returns:
            report_stats(Statistics): updated Statistics instance.
        """
        if self.start_time < 0:
            raise ValueError("""ReportMgr needs to be started
                                (set 'start_time' or use 'start()'""")

        if step % self.report_every == 0:
            if multigpu:
                report_stats = \
                    Statistics.all_gather_stats(report_stats)
            self._report_training(
                step, num_steps, learning_rate, report_stats)
            self.progress_step += 1
            return Statistics(print_traj=is_joint)
        else:
            return report_stats

    def _report_training(self, *args, **kwargs):
        """ To be overridden """
        raise NotImplementedError()

    def report_step(self, lr, step, train_stats=None, valid_stats=None):
        """
        Report stats of a step

        Args:
            train_stats(Statistics): training stats
            valid_stats(Statistics): validation stats
            lr(float): current learning rate
        """
        self._report_step(
            lr, step, train_stats=train_stats, valid_stats=valid_stats)

    def _report_step(self, *args, **kwargs):
        raise NotImplementedError()


class ReportMgr(ReportMgrBase):
    def __init__(self, report_every, start_time=-1., tensorboard_writer=None):
        """
        A report manager that writes statistics on standard output as well as
        (optionally) TensorBoard

        Args:
            report_every(int): Report status every this many sentences
            tensorboard_writer(:obj:`tensorboard.SummaryWriter`):
                The TensorBoard Summary writer to use or None
        """
        super(ReportMgr, self).__init__(report_every, start_time)
        self.tensorboard_writer = tensorboard_writer

    def maybe_log_tensorboard(self, stats, prefix, learning_rate, step, report_rl=False):
        if self.tensorboard_writer is not None:
            stats.log_tensorboard(
                prefix, self.tensorboard_writer, learning_rate, step, report_rl)

    def _report_training(self, step, num_steps, learning_rate,
                         report_stats):
        """
        See base class method `ReportMgrBase.report_training`.
        """
        report_stats.output(step, num_steps,
                            learning_rate, self.start_time)

        # Log the progress using the number of batches on the x-axis.
        self.maybe_log_tensorboard(report_stats,
                                   "progress",
                                   learning_rate,
                                   self.progress_step)
        report_stats = Statistics()

        return report_stats

    def _report_opt(self, step, report_stats):
        self.maybe_log_tensorboard(report_stats,
                                   "Opt/Steps",
                                   learning_rate=1,
                                   step=step)

    def _report_step(self, lr, step, train_stats=None, valid_stats=None):
        """
        See base class method `ReportMgrBase.report_step`.
        """
        # if train_stats is not None:
        # self.log('Train xent: %g' % train_stats.xent())

        # self.maybe_log_tensorboard(train_stats,
        #                            "train",
        #                            lr,
        #                            step)

        if valid_stats is not None:
            self.log('Validation xent: %g (mse_sent: %g, xent_sect: %g), (ACC: %4.4f) at step %d' % (valid_stats.total_loss(),
                                                                                       valid_stats.mse_sent(),
                                                                                       valid_stats.xent_sect(), valid_stats._get_acc_sect(), step))
            # self.log('Validation: sent_xent: %g, sect_xent: %g, xent: %g at step %d' %
            #          (valid_stats.xent_sent(),valid_stats.xent_sect(),valid_stats.xent(), step))

            self.maybe_log_tensorboard(valid_stats,
                                       "valid",
                                       lr,
                                       step, report_rl=True)


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, loss_sent=0, loss_sect=0, n_docs=0, n_acc=0, RMSE=0, accuracy=0, F1sect={},
                 n_correct=0, r1=0, r2=0, rl=0, stat_file_dir=None, print_traj=False):
        self.loss = loss
        self.loss_sect = loss_sect
        self.loss_sent = loss_sent
        self.print_traj = print_traj
        self.n_docs = n_docs
        self.n_acc = n_acc
        self.r1 = r1
        self.r2 = r2
        self.rl = rl
        self.RMSE = RMSE
        self.accuracy = accuracy
        # self.F1sect = F1sect.copy()
        # self.n_f1_sep = F1sect.copy()
        #
        # for key, val in F1sect.items():
        #     if val != -1:
        #         self.n_f1_sep[key] = val[1]
        #     if val == -1:
        #         self.n_f1_sep[key] = 0
        #
        # for key, val in F1sect.items():
        #     if val != -1:
        #         self.F1sect[key] = val[0]
        #     if val == -1:
        #         self.F1sect[key] = 0

        self.start_time = time.time()
        if stat_file_dir is not None:
            if stat_file_dir[-1] == '/':
                stat_file_dir = stat_file_dir[:-1]
            stat_file_dir = stat_file_dir.rsplit('/', 1)[1]
            self.validation_file = open('../logs/valstat.' + stat_file_dir + '.txt', mode='a')

    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """
        Gather a `Statistics` object accross multiple process/nodes

        Args:
            stat(:obj:Statistics): the statistics object to gather
                accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            `Statistics`, the update stats object
        """
        stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
        return stats[0]

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        """
        Gather a `Statistics` list accross all processes/nodes

        Args:
            stat_list(list([`Statistics`])): list of statistics objects to
                gather accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            our_stats(list([`Statistics`])): list of updated stats
        """
        from torch.distributed import get_rank
        from distributed import all_gather_list

        # Get a list of world_size lists with len(stat_list) Statistics objects
        all_stats = all_gather_list(stat_list, max_size=max_size)

        our_rank = get_rank()
        our_stats = all_stats[our_rank]
        for other_rank, stats in enumerate(all_stats):
            if other_rank == our_rank:
                continue
            for i, stat in enumerate(stats):
                our_stats[i].update(stat, update_n_src_words=True)
        return our_stats

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not

        """
        self.loss += stat.loss
        self.loss_sent += stat.loss_sent
        self.loss_sect += stat.loss_sect

        self.RMSE += stat.RMSE
        self.accuracy += stat.accuracy
        self.n_docs += stat.n_docs
        self.n_acc += stat.n_acc

        # section-wise
        # self.F1sect += stat.F1sect
        # self.F1sect = np.add(self.F1sect, stat.F1sect)

        # for key, val in stat.F1sect.items():
        #     if key in self.F1sect.keys():
        #         self.F1sect[key] += val
        #     else:
        #         self.F1sect[key] = val
        #
        # for key, val in stat.n_f1_sep.items():
        #     if key in self.n_f1_sep.keys():
        #         self.n_f1_sep[key] += val
        #     else:
        #         self.n_f1_sep[key] = val

        # self.F1sect = np.add(self.F1sect, stat.F1sect)

        # self.n_acc += stat.n_acc

        # import pdb;
        # pdb.set_trace()

    def set_rl(self, r1, r2, rl):
        self.r1 = r1
        self.r2 = r2
        self.rl = rl

    def total_loss(self):
        """ compute cross entropy """
        if (self.n_docs == 0):
            return 0
        return self.loss / self.n_docs

    def _get_rmse_sent(self):
        if self.n_acc == 0:
            return 0
        return math.sqrt(self.RMSE / self.n_acc)

    def _get_f1_sect(self):
        if self.n_acc == 0:
            return 0
        f0 = self.F1sect['f0'] / self.n_f1_sep['f0'] if self.n_f1_sep['f0'] != 0 else 0
        f1 = self.F1sect['f1'] / self.n_f1_sep['f1'] if self.n_f1_sep['f1'] != 0 else 0
        f2 = self.F1sect['f2'] / self.n_f1_sep['f2'] if self.n_f1_sep['f2'] != 0 else 0
        f3 = self.F1sect['f3'] / self.n_f1_sep['f3'] if self.n_f1_sep['f3'] != 0 else 0
        f4 = self.F1sect['f4'] / self.n_f1_sep['f4'] if self.n_f1_sep['f4'] != 0 else 0

        return (f0 + f1 + f2 + f3 + f4) / 5, f0, f1, f2, f3, f4

    def mse_sent(self):
        """ compute cross entropy """
        if (self.n_docs == 0):
            return 0
        # if float(('%4.6f')%(self.loss_sent / self.n_docs)) == 0:
        #     import pdb;pdb.set_trace()
        return self.loss_sent / self.n_docs

    def xent_sect(self):
        """ compute cross entropy """
        if (self.n_docs == 0):
            return 0
        return self.loss_sect / self.n_docs

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate, start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        step_fmt = "%2d" % step
        if num_steps > 0:
            step_fmt = "%s/%5d" % (step_fmt, num_steps)

        if self.print_traj:
            # foveral, f11, f12, f13, f14, f15 = self.self_get_f1_sect()
            logger.info(
                (
                        "Step %s; mse_sent: %4.2f (%4.2f/%d) + xent_sect: %4.2f = mlt: %4.2f (RMSE-sent: %4.4f, ACC: %4.4f) "
                        # "F1-sect: %4.2f ([0] %4.2f, "
                        # "[1] %4.2f, [2] %4.2f, [3] %4.2f, [4] %4.2f)); " +
                        "lr: %7.7f; %3.0f docs/s; %6.0f sec")
                % (step_fmt,
                   self.mse_sent(),
                   self.loss_sent,
                   self.n_docs,
                   self.xent_sect(),
                   self.total_loss(),
                   self._get_rmse_sent(),
                   self._get_acc_sect(),
                   # foveral,
                   # f11,
                   # f12,
                   # f13,
                   # f14,
                   # f15,
                   learning_rate,
                   self.n_docs / (t + 1e-5),
                   time.time() - start))
        else:
            logger.info(
                ("Step %s; mse_sent: %4.2f (%4.2f/%d), (RMSE-sent:%4.2f); " +
                 "lr: %7.7f; %3.0f docs/s; %6.0f sec")
                % (step_fmt,
                   self.total_loss(),
                   self.loss,
                   self.n_docs,
                   self._get_rmse_sent(),
                   learning_rate,
                   self.n_docs / (t + 1e-5),
                   time.time() - start))
        sys.stdout.flush()

    def log_tensorboard(self, prefix, writer, learning_rate, step, report_rl=False):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/total_loss", self.total_loss(), step)
        # writer.add_scalar(prefix + "/RMSE", self._get_rmse_sent(), step)
        # writer.add_scalar(prefix + "/F1_sect", self._get_f1_sect()[0], step)
        # writer.add_scalar(prefix + "/mse_sent", self.mse_sent(), step)
        writer.add_scalar(prefix + "/xent_sect", self.xent_sect(), step)
        writer.add_scalar(prefix + "/ACC", self._get_acc_sect(), step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)
        if report_rl:
            writer.add_scalar(prefix + "/Rouge-1", self.r1, step)
            writer.add_scalar(prefix + "/Rouge-2", self.r2, step)
            writer.add_scalar(prefix + "/Rouge-l", self.rl, step)

    def write_stat_header(self, is_joint):
        if not is_joint:
            self.validation_file.write(
                f'\n#Step\t\tLoss\tR-1\tR-2\tR-L\n'
            )
        else:
            self.validation_file.write(
                f'\n#Step\t\t(sent, sect, loss)\t\tR-1\tR-2\tR-L\n'
            )

        self.validation_file.flush()
        os.fsync(self.validation_file)

    def _write_file(self, step, stat, r1, r2, rl):
        self.validation_file.write(
            '%d\t\t(%4.2f, %4.2f, %4.2f)\t\t%4.2f\t%4.2f\t%4.2f\n' % (step, stat.mse_sent(),
                                                                      stat.xent_sect(), stat.total_loss()
                                                                      , r1, r2, rl)
        )
        self.validation_file.flush()
        os.fsync(self.validation_file)

    def _get_acc_sect(self):
        if self.n_docs == 0:
            return 0
        return self.accuracy / self.n_acc