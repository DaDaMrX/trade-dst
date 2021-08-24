import pytorch_lightning as pl
import torch

from models.trade import Trade


class LightningModel(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(self.args)
        self.model = Trade(
            hidden_size=self.args['hidden_size'],
            dropout_p=self.args['dropout'],
            teacher_forcing_prob=self.args['teacher_forcing_prob'],
            padding_idx=self.args['padding_idx'],
        )
        if self.args['load_embedding'] == 'yes':
            self.model.load_embedding()
        self.valid_cnt = 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args['lr'],
        )
        optimizers = [optimizer]

        # warm_steps = len(self.trainer.datamodule.train_dataloader())
        # warm_steps = (warm_steps + self.trainer.gpus - 1) // self.trainer.gpus
        # scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer=optimizer,
        #     lr_lambda=lambda step: step / warm_steps if step < warm_steps else 1.0,
        # )
        # scheduler_warmup_dict = {
        #     'scheduler': scheduler_warmup,
        #     'interval': 'step',
        #     'frequency': 1,
        #     'name': 'lr/Warmup'
        # }

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='max',
            factor=self.args['reduce_lr_factor'],
            patience=self.args['reduce_lr_patience'],
            min_lr=self.args['reduce_lr_min'],
            verbose=True,
        )
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'joint_acc',
            'name': 'lr/ReduceLROnPlateau'
        }
        schedulers = [scheduler_dict]
        # schedulers = [scheduler_warmup_dict, scheduler_dict]

        return optimizers, schedulers

    def on_train_epoch_start(self):
        # Set tqdm train description: [tag] Epoch x
        pbar_desc = f'[{self.args["tag"]}] Epoch {self.trainer.current_epoch}'
        for callback in self.trainer.callbacks:
            if isinstance(callback, pl.callbacks.ProgressBar):
                callback.main_progress_bar.set_description(pbar_desc)

    def training_step(self, batch, batch_idx):
        output = self.model(batch)

        self.log('Loss/train_step_loss_gate', output['loss_gate'], on_step=True, on_epoch=False)
        self.log('Loss/train_step_loss_ptr', output['loss_ptr'], on_step=True, on_epoch=False)
        self.log('Loss/train_step_loss', output['loss'], on_step=True, on_epoch=False)

        result = {
            'loss_gate': output['loss_gate'].item(),
            'loss_ptr': output['loss_ptr'].item(),
            'loss': output['loss'],
        }
        return result

    def training_epoch_end(self, results):
        loss_gate = [d['loss_gate'] for d in results]
        epoch_loss_gate = sum(loss_gate) / len(loss_gate)
        loss_ptr = [d['loss_ptr'] for d in results]
        epoch_loss_ptr = sum(loss_ptr) / len(loss_ptr)
        epoch_loss = torch.mean(torch.stack([d['loss'] for d in results]))
        self.log('Loss/train_epoch_loss_gate', epoch_loss_gate, on_step=False, on_epoch=True)
        self.log('Loss/train_epoch_loss_ptr', epoch_loss_ptr, on_step=False, on_epoch=True)
        self.log('Loss/train_epoch_loss', epoch_loss, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        output = self.model(batch)

        result = {
            'dialogue_idx': batch['dialogue_idx'],
            'turn_idx': batch['turn_idx'],
            'belief_state': batch['belief_state'],
            'preds_idx': output['preds_idx'].cpu(),
            'gates_idx': output['gates_idx'].cpu(),
        }
        return result

    def validation_epoch_end(self, results):
        self.valid_cnt += 1
        self.log('valid_cnt', self.valid_cnt, on_step=False, on_epoch=True)
        self.log('step', self.valid_cnt, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
