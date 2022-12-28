import glob
import os
import time

import numpy as np

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.postprocessors import get_postprocessor
from openood.utils import setup_logger


class TestOODPipelineMultiruns:
    def __init__(self, config) -> None:
        self.config = config
        self.num_runs = len(
            glob.glob(os.path.join(self.config.network.ckpt_dir, 's*')))
        self.ckpt_paths = [None for _ in range(self.num_runs)]

        for i, folder in enumerate(
                glob.glob(os.path.join(self.config.network.ckpt_dir, 's*'))):
            temp = glob.glob(os.path.join(folder, 'best_epoch*.ckpt'))
            assert len(temp) == 1, temp  # sanity check
            self.ckpt_paths[i] = temp[0]

    def run(self):
        # generate output directory and save the full config file
        # manually modify the output dir
        self.config.output_dir = self.config.network.ckpt_dir
        setup_logger(self.config)

        # get dataloader
        id_loader_dict = get_dataloader(self.config)
        ood_loader_dict = get_ood_dataloader(self.config)

        all_ood_metrics = []
        for r in range(self.num_runs):
            # init network
            self.config.network.pretrained = True
            self.config.network.checkpoint = self.ckpt_paths[r]
            net = get_network(self.config.network)

            # manually modify the output dir
            self.config.output_dir = '/'.join(
                self.config.network.checkpoint.split('/')[:-1])

            # init ood evaluator
            evaluator = get_evaluator(self.config)

            # init ood postprocessor
            postprocessor = get_postprocessor(self.config)
            # setup for distance-based methods
            postprocessor.setup(net, id_loader_dict, ood_loader_dict)
            print('\n', flush=True)
            print(u'\u2500' * 70, flush=True)

            # start calculating accuracy
            print(f'\nStart evaluation for [{r+1}/{self.num_runs}] run...',
                  flush=True)
            acc_metrics = evaluator.eval_acc(net, id_loader_dict['test'],
                                             postprocessor)
            print('\nAccuracy {:.2f}%'.format(100 * acc_metrics['acc']),
                  flush=True)
            print(u'\u2500' * 70, flush=True)

            # start evaluating ood detection methods
            timer = time.time()
            ood_metrics = evaluator.eval_ood(net, id_loader_dict,
                                             ood_loader_dict, postprocessor)
            print('Time used for eval_ood: {:.0f}s'.format(time.time() -
                                                           timer))
            all_ood_metrics.append(ood_metrics)

        self.config.output_dir = self.config.network.ckpt_dir
        for dataset_name in all_ood_metrics[0].keys():
            metrics_over_runs = [all_ood_metrics[0][dataset_name]]
            for r in range(1, self.num_runs):
                metrics_over_runs.append(all_ood_metrics[1][dataset_name])
            metrics_over_runs = np.array(metrics_over_runs)
            metrics_mean = np.mean(metrics_over_runs, axis=0)
            metrics_std = np.std(metrics_over_runs, axis=0)
            evaluator._save_csv_with_std(metrics_mean, metrics_std,
                                         dataset_name, self.num_runs)

        print('Completed!', flush=True)
