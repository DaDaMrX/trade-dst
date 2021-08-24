import json
import os
import pickle

import pytorch_lightning as pl
import torch
from tqdm import tqdm

from common.gate import load_gate_dict
from common.slots import load_slots
from common.vocab import build_vocab


class TradeDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size, *,
        train=True, valid=True, test=True, enforce_refresh=False):
        super().__init__()
        self.data_dir = data_dir
        self.clean_dir = os.path.join(self.data_dir, 'clean')
        self.pickle_dir = os.path.join(self.data_dir, 'pickle')

        self.batch_size = batch_size
        self.train = train
        self.valid = valid
        self.test = test
        self.enforce_refresh = enforce_refresh

        self.slots = load_slots()
        self.gate_dict = load_gate_dict()
        self.vocab = build_vocab()

        self.exp_domains = ['hotel', 'train', 'restaurant', 'attraction', 'taxi']

    def prepare_data(self):
        if self.train:
            self.read_data(
                data_path=os.path.join(self.clean_dir, 'train.json'),
                pickle_path=os.path.join(self.pickle_dir, 'train.pickle'),
            )

        if self.valid:
            self.read_data(
                data_path=os.path.join(self.clean_dir, 'valid.json'),
                pickle_path=os.path.join(self.pickle_dir, 'valid.pickle'),
            )

        if self.test:
            self.read_data(
                data_path=os.path.join(self.clean_dir, 'test.json'),
                pickle_path=os.path.join(self.pickle_dir, 'test.pickle'),
            )

    def read_data(self, data_path, pickle_path):
        if os.path.exists(pickle_path) and not self.enforce_refresh:
            return

        with open(data_path) as f:
            data = json.load(f)

        dataset = []
        for dialog in data:
            context_txt = ''
            for turn in dialog['dialogue']:
                item = {
                    'dialogue_idx': None,
                    'turn_idx': None,
                    'context_txt': None,
                    'belief_state': None,
                    'gates_txt': None,
                    'values_txt': None,
                }
                dataset.append(item)

                item['dialogue_idx'] = dialog['dialogue_idx']
                item['turn_idx'] = turn['turn_idx']

                context_txt += turn['system_transcript'] + ' ; ' + turn['transcript'] + ' ; '
                item['context_txt'] = context_txt.strip()

                item['belief_state'] = turn['belief_state']
                belief_state_dict = {}
                for state in turn['belief_state']:
                    slot, value = state.rsplit('-', 1)
                    belief_state_dict[slot] = value

                gates_txt, values_txt = [], []
                for slot in self.slots:
                    if slot not in belief_state_dict.keys():
                        gates_txt.append('none')
                        values_txt.append('none')
                        continue

                    value = belief_state_dict[slot]
                    if value == 'none':
                        gates_txt.append('none')
                        values_txt.append('none')
                    elif value == 'dontcare':
                        gates_txt.append('dontcare')
                        values_txt.append('dontcare')
                    else:
                        gates_txt.append('ptr')
                        values_txt.append(value)
                item['gates_txt'] = gates_txt
                item['values_txt'] = values_txt

                assert None not in item.values()

        # Make data: txt to idx
        for item in tqdm(dataset, desc='Load Data'):
            item['context_idx'] = None
            item['gates_idx'] = None
            item['values_idx'] = None

            item['context_idx'] = self.vocab.sent_to_idxs(item['context_txt'])
            item['gates_idx'] = [self.gate_dict[g] for g in item['gates_txt']]

            values_idx = []
            for value in item['values_txt']:
                idx = self.vocab.sent_to_idxs(value, with_eos=True)
                values_idx.append(idx)
            item['values_idx'] = values_idx

            assert None not in item.values()

        pickle_dir = os.path.dirname(pickle_path)
        if not os.path.exists(pickle_dir):
            os.mkdir(pickle_dir)
        with open(pickle_path, 'wb') as f:
            pickle.dump(dataset, f)

    @staticmethod
    def collate_fn(items):
        items.sort(key=lambda x: len(x['context_idx']), reverse=True) 

        batch = {}
        for k in items[0]:
            batch[k] = [item[k] for item in items]

        def pad(seqs):
            lens = [len(seq) for seq in seqs]
            seqs = torch.nn.utils.rnn.pad_sequence(
                sequences=list(map(torch.tensor, seqs)),
                batch_first=True,
                padding_value=0,
            )
            lengths = torch.tensor(lens).unsqueeze(1).expand_as(seqs)
            arange = torch.arange(seqs.shape[1]).expand_as(seqs)
            mask = (arange < lengths).type(torch.int64)
            return seqs, lens, mask

        batch['context_idx'], batch['context_lens'], batch['context_mask'] = pad(batch['context_idx'])

        batch['gates_idx'] = torch.tensor(batch['gates_idx'])

        values_idx = sum(batch['values_idx'], [])
        batch['values_idx'], batch['values_lens'], batch['values_mask'] = pad(values_idx)

        batch_size, slots_size = len(items), 30  # len(self.slots)
        batch['values_idx'] = batch['values_idx'].reshape(batch_size, slots_size, -1)
        batch['values_lens'] = torch.tensor(batch['values_lens']).reshape(batch_size, slots_size)
        batch['values_mask'] = batch['values_mask'].reshape(batch_size, slots_size, -1)

        # context_idx: B * L
        # context_lens: List[int], B
        # context_mask: B * L
        # gates_idx: B
        # values_idx: B * S * K
        # values_lens: B * S
        # values_mask: B * S * K
        return batch

    def train_dataloader(self):
        if not self.train:
            return None
        pickle_path = os.path.join(self.pickle_dir, 'train.pickle')
        with open(pickle_path, 'rb') as f:
            dataset = pickle.load(f)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=TradeDataModule.collate_fn,
            shuffle=True,
            num_workers=1,
        )
        return loader

    def val_dataloader(self):
        if not self.valid:
            return None
        pickle_path = os.path.join(self.pickle_dir, 'valid.pickle')
        with open(pickle_path, 'rb') as f:
            dataset = pickle.load(f)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=TradeDataModule.collate_fn,
            shuffle=False,
            num_workers=1,
        )
        return loader

    def test_dataloader(self):
        if not self.test:
            return None
        pickle_path = os.path.join(self.pickle_dir, 'test.pickle')
        with open(pickle_path, 'rb') as f:
            dataset = pickle.load(f)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=TradeDataModule.collate_fn,
            shuffle=False,
            num_workers=1,
        )
        return loader


if __name__ == '__main__':
    pl.seed_everything(12345)

    data = TradeDataModule(
        data_dir='data',
        batch_size=32,
        train=True,
        valid=True,
        test=True,
        enforce_refresh=True,
    )
    data.prepare_data()

    train = data.train_dataloader()
    valid = data.val_dataloader()
    test = data.test_dataloader()
    print(len(train), len(valid), len(test))
