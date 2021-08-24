import os

import pytorch_lightning as pl

from metric import metric_file


class MetricCallback(pl.Callback):

    def __init__(self, save_dir, valid_data_path=None, test_data_path=None):
        self.save_dir = save_dir
        self.valid_data_path = valid_data_path
        self.test_data_path = test_data_path

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.valid_data_path is None:
            return

        valid_cnt = int(trainer.callback_metrics['valid_cnt'])
        pred_path = os.path.join(self.save_dir, f'pred-valid_cnt={valid_cnt}.json')
        scores = metric_file(
            pred_path=pred_path,
            truth_path=self.valid_data_path,
        )

        for k, v in scores.items():
            self.log(f'Valid/{k}', v, on_step=False, on_epoch=True)

        self.log('joint_acc', scores['joint_acc'], on_step=False, on_epoch=True)

        if trainer.is_global_zero:
            print('\n\n', end='')
            for k, v in scores.items():
                print(f'{k}: {v:.2f}')
            print()

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_data_path is None:
            return

        pred_path = os.path.join(self.save_dir, 'test.json')
        scores = metric_file(
            pred_path=pred_path,
            truth_path=self.test_data_path,
        )

        self.log('hp_metric', scores['joint_acc'], on_step=False, on_epoch=True)

        if trainer.is_global_zero:
            print('\n\n', end='')
            for k, v in scores.items():
                print(f'{k}: {v:.2f}')
            print()
