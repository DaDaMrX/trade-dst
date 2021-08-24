import argparse
import os
import warnings

import pynvml
import pytorch_lightning as pl
import torch

from callbacks.metric_callback import MetricCallback
from callbacks.results_writer import ResultsWriter
from pl_data import TradeDataModule
from pl_model import LightningModel

warnings.simplefilter("ignore", UserWarning)


def set_gpu(n):
    pynvml.nvmlInit()
    total_gpus = pynvml.nvmlDeviceGetCount()
    infos = []
    for index in range(total_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        infos.append({
            'id': index,
            'free': memory.free,
        })
    infos.sort(key=lambda x: x['free'], reverse=True)
    ids = [str(info['id']) for info in infos[:n]]
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(ids)
    return index


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tag', '-t', required=True)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--save_dir', default='outputs')
    parser.add_argument('--gpus', type=int, default=1)

    # Data
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--valid_data_path', default='data/clean/valid.json')
    parser.add_argument('--test_data_path', default='data/clean/test.json')
    parser.add_argument('--batch_size', type=int, default=32)

    # Model
    parser.add_argument('--load_embedding', default='yes', choices=['yes', 'no'])
    parser.add_argument('--hidden_size', type=int, default=400)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--grad_clip', type=int, default=10)
    parser.add_argument('--teacher_forcing_prob', type=float, default=0.5)

    # Training
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--reduce_lr_factor', type=float, default=0.5)
    parser.add_argument('--reduce_lr_patience', type=int, default=1)
    parser.add_argument('--reduce_lr_min', type=float, default=1e-4)
    parser.add_argument('--early_stop_patience', type=int, default=6)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--n_valid_every_epoch', type=int, default=1)
    parser.add_argument('--log_every_n_steps', type=int, default=20)
    parser.add_argument('--precision', type=int, default=32, choices=[32, 16])

    # Saving
    parser.add_argument('--save_filename', default='{valid_cnt:.0f}-{epoch}-{step}-{joint_acc:.4f}')
    parser.add_argument('--save_top_k', type=int, default=1)
    parser.add_argument('--save_monitor', default='joint_acc')
    parser.add_argument('--save_monitor_mode', default='max')

    args = parser.parse_args()
    return vars(args)


def build_trainer(args):
    # 1. Save log
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=args['save_dir'],
        name=args['tag'],
    )
    lr_monitor = pl.callbacks.LearningRateMonitor()
    # 2. Save ckpt
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(tb_logger.log_dir, 'ckpts'),
        filename=args['save_filename'],
        save_weights_only=True,
        save_top_k=args['save_top_k'],
        monitor=args['save_monitor'],
        mode=args['save_monitor_mode'],
    )
    # 3. Save results
    results_writer = ResultsWriter(
        save_dir=os.path.join(tb_logger.log_dir, 'results'),
    )
    metric_callback = MetricCallback(
        save_dir=results_writer.save_dir,
        valid_data_path=args['valid_data_path'],
        test_data_path=args['test_data_path'],
    )
    # 4. Early Stop
    early_stopping = pl.callbacks.EarlyStopping(
        monitor='joint_acc',
        patience=args['early_stop_patience'],
        verbose=True,
        mode='max',
        check_on_train_epoch_end=False,
    )
    callbacks = [
        lr_monitor, ckpt_callback, results_writer,
        metric_callback, early_stopping,
    ]

    if args['gpus'] > 1:
        accelerator = 'ddp'
        plugins = pl.plugins.DDPPlugin(find_unused_parameters=False)
    else:
        accelerator, plugins = None, None

    trainer = pl.Trainer(
        gpus=args['gpus'],
        accelerator=accelerator,
        plugins=plugins,
        deterministic=True,
        gradient_clip_val=args['grad_clip'],
        gradient_clip_algorithm='norm',
        logger=tb_logger,
        log_every_n_steps=args['log_every_n_steps'],
        callbacks=callbacks,
        max_epochs=args['max_epochs'],
        val_check_interval=1 / args['n_valid_every_epoch'],
        num_sanity_val_steps=0,
        precision=args['precision'],
    )
    return trainer


if __name__ == '__main__':
    args = parse_args()
    pl.seed_everything(args['seed'], workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    set_gpu(args['gpus'])

    data = TradeDataModule(
        data_dir=args['data_dir'],
        batch_size=args['batch_size'],
    )

    model = LightningModel(args)

    trainer = build_trainer(args)

    trainer.fit(
        model=model,
        datamodule=data,
    )

    for callback in trainer.callbacks:
        if isinstance(callback, pl.callbacks.ModelCheckpoint):
            print('best_model_path:', callback.best_model_path)
            print('best_model_score:', callback.best_model_score)

    trainer.test(datamodule=data)
