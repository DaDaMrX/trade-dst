import os
import json

import torch
import pytorch_lightning as pl

from common.gate import load_inv_gate_dict
from common.slots import load_slots
from common.vocab import build_vocab


class ResultsWriter(pl.Callback):

    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        self.results = []

        self.vocab = build_vocab()
        self.slots = load_slots()
        self.inv_gating_dict = load_inv_gate_dict()

    def on_validation_epoch_start(self, trainer, pl_module):
        self.results = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.results.append(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        data = self.process_results()

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        valid_cnt = int(trainer.callback_metrics['valid_cnt'])
        path = os.path.join(self.save_dir, f'pred-valid_cnt={valid_cnt}-rank={trainer.local_rank}.json')
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

        if trainer.gpus > 1:
            torch.distributed.barrier()
        self.merge_valid_results(valid_cnt, trainer.gpus)
        if trainer.gpus > 1:
            torch.distributed.barrier()

    @pl.utilities.rank_zero_only
    def merge_valid_results(self, valid_cnt, gpus):
        data = []
        for rank in range(gpus):
            path = os.path.join(self.save_dir, f'pred-valid_cnt={valid_cnt}-rank={rank}.json')
            with open(path) as f:
                data += json.load(f)
            os.remove(path)

        data = {(turn['dialogue_idx'], turn['turn_idx']): turn for turn in data}
        data = list(data.values())

        path = os.path.join(self.save_dir, f'pred-valid_cnt={valid_cnt}.json')
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    def on_test_epoch_start(self, trainer, pl_module):
        self.results = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.results.append(outputs)

    def on_test_epoch_end(self, trainer, pl_module):
        data = self.process_results()

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        path = os.path.join(self.save_dir, f'test-rank={trainer.local_rank}.json')
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

        if trainer.gpus > 1:
            torch.distributed.barrier()
        self.merge_test_results(trainer.gpus)
        if trainer.gpus > 1:
            torch.distributed.barrier()

    @pl.utilities.rank_zero_only
    def merge_test_results(self, gpus):
        data = []
        for rank in range(gpus):
            path = os.path.join(self.save_dir, f'test-rank={rank}.json')
            with open(path) as f:
                data += json.load(f)
            os.remove(path)

        data = {(turn['dialogue_idx'], turn['turn_idx']): turn for turn in data}
        data = list(data.values())

        path = os.path.join(self.save_dir, f'test.json')
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    def process_results(self):
        data = []
        for result in self.results:
            preds_idx = result['preds_idx'].tolist()  # B * S * K
            gates_idx = result['gates_idx'].tolist()  # B * S
            for i in range(len(result['dialogue_idx'])):
                turn = {}
                data.append(turn)
                turn['dialogue_idx'] = result['dialogue_idx'][i]
                turn['turn_idx'] = result['turn_idx'][i]
                turn['belief_state'] = result['belief_state'][i]
                turn['pred_belief_state'] = []
                for slot, gate, value in zip(self.slots, gates_idx[i], preds_idx[i]):
                    gate = self.inv_gating_dict[gate]
                    if gate == 'none':
                        continue
                    elif gate == 'dontcare':
                        value = 'dontcare'
                    else:
                        value = self.vocab.idxs_to_sent(value, rm_special_tokens=True)
                    turn['pred_belief_state'].append(f'{slot}-{value}')

        data.sort(key=lambda turn: (turn['dialogue_idx'], turn['turn_idx']))
        return data
