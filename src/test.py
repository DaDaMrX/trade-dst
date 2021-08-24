import argparse
import os
import warnings

import pytorch_lightning as pl

from callbacks.metric_callback import MetricCallback
from callbacks.results_writer import ResultsWriter
from pl_data import TradeDataModule
from pl_model import LightningModel
from train import set_gpu

warnings.simplefilter("ignore", UserWarning)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_path', required=True)

    parser.add_argument('--save_dir', default='outputs')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--test_data_path', default='data/clean/test.json')
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()
    return vars(args)


def build_trainer(args):
    results_writer = ResultsWriter(
        save_dir=os.path.join(args['save_dir'], 'test'),
    )
    metric_callback = MetricCallback(
        save_dir=results_writer.save_dir,
        test_data_path=args['test_data_path'],
    )
    callbacks = [results_writer, metric_callback]

    trainer = pl.Trainer(
        gpus=1,
        logger=False,
        checkpoint_callback=False,
        callbacks=callbacks,
    )
    return trainer


if __name__ == '__main__':
    args = parse_args()
    set_gpu(1)
    data = TradeDataModule(
        data_dir=args['data_dir'],
        batch_size=args['batch_size'],
        train=False,
        valid=False,
        test=True,
    )
    model = LightningModel.load_from_checkpoint(args['ckpt_path'])
    trainer = build_trainer(args)
    trainer.test(
        model=model,
        datamodule=data,
    )
