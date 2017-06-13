import rllab.misc.logger as logger
from rllab import config
import os.path as osp


class RLLabRunner(object):

    def __init__(self, algo, args, exp_name):
        self.args = args
        self.algo = algo

        env = algo.env
        baseline = algo.baseline

        # Logger
        default_log_dir = config.LOG_DIR
        if args.log_dir is None:
            log_dir = osp.join(default_log_dir, exp_name)
        else:
            log_dir = args.log_dir

        tabular_log_file = osp.join(log_dir, args.tabular_log_file)
        text_log_file = osp.join(log_dir, args.text_log_file)
        params_log_file = osp.join(log_dir, args.params_log_file)

        logger.log_parameters_lite(params_log_file, args)
        logger.add_text_output(text_log_file)
        logger.add_tabular_output(tabular_log_file)
        logger.set_snapshot_dir(log_dir)
        logger.set_snapshot_mode(args.snapshot_mode)
        logger.set_log_tabular_only(args.log_tabular_only)
        logger.push_prefix("[%s] " % exp_name)

        prev_snapshot_dir = logger.get_snapshot_dir()
        prev_mode = logger.get_snapshot_mode()

    def train(self):
        self.algo.train()
