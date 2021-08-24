import json
import os

from common.slots import load_slots


class Vocab:

    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}

        self.pad_token, self.pad_idx = 'PAD', 0
        self.unk_token, self.unk_idx = 'UNK', 1
        self.sos_token, self.sos_idx = 'SOS', 2
        self.eos_token, self.eos_idx = 'EOS', 3
        self.special_tokens = [
            self.pad_token, self.unk_token,
            self.sos_token, self.eos_token,
        ]
        for token in self.special_tokens:
            self.add_token(token)

    def __len__(self):
        return len(self.token2idx)

    def add_token(self, token):
        if token in self.token2idx:
            return
        idx = len(self.token2idx)
        self.token2idx[token] = idx
        self.idx2token[idx] = token

    def add_sent(self, sent):
        for token in sent.strip().split():
            self.add_token(token)

    def token_to_idx(self, token):
        return self.token2idx.get(token, self.unk_idx)

    def idx_to_token(self, idx):
        idx = int(idx)
        return self.idx2token[idx] if idx < len(self.idx2token) else self.unk_token

    def sent_to_idxs(self, sent, with_sos=False, with_eos=False):
        s = [self.token_to_idx(t) for t in sent.strip().split()]
        if with_sos:
            s.insert(0, self.sos_idx)
        if with_eos:
            s.append(self.eos_idx)
        return s

    def idxs_to_sent(self, idxs, rm_special_tokens=False):
        s = [self.idx_to_token(idx) for idx in idxs]
        if rm_special_tokens:
            while s and s[0] in self.special_tokens:
                s.pop(0)
            for i in range(len(s)):
                if s[i] == self.eos_token:
                    s = s[:i]
                    break
        return ' '.join(s)

    def dump(self, path):
        data = list(self.token2idx.items())
        data.sort(key=lambda t: t[1])
        data = [f'{token}\t{idx}' for token, idx in data]
        with open(path, 'w') as f:
            f.write('\n'.join(data) + '\n')

    def load(self, path):
        with open(path) as f:
            data = [s.strip().split() for s in f.read().splitlines()]
        self.token2idx = {token: int(idx) for token, idx in data}
        self.idx2token = {int(idx): token for token, idx in data}


def build_vocab(data_dir='data', enforce_refresh=False):
    vocab = Vocab()

    vocab_path = os.path.join(data_dir, 'cache', 'vocab.tsv')
    if not enforce_refresh and os.path.exists(vocab_path):
        vocab.load(vocab_path)
        return vocab

    slots = load_slots()
    for domain_slot in slots:
        domain, slot = domain_slot.strip().split('-')
        vocab.add_token(domain)
        vocab.add_sent(slot)

    path = os.path.join(data_dir, 'clean', 'train.json')
    with open(path) as f:
        data = json.load(f)
    for dialog in data:
        for turn in dialog['dialogue']:
            vocab.add_sent(turn['system_transcript'])
            vocab.add_sent(turn['transcript'])
            for state in turn['belief_state']:
                vocab.add_sent(state.split('-')[-1])
    vocab.add_token(';')

    dump_dir = os.path.join(data_dir, 'cache')
    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)
    dump_path = os.path.join(dump_dir, 'vocab.tsv')
    vocab.dump(dump_path)

    return vocab


if __name__ == '__main__':
    vocab = build_vocab(
        enforce_refresh=True,
    )
