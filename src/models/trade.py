import random

import torch

from common.gate import load_gate_dict
from common.slots import load_slots
from common.vocab import build_vocab
from common.embedding import build_embeddings


class Trade(torch.nn.Module):

    def __init__(self, hidden_size, dropout_p, teacher_forcing_prob):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.teacher_forcing_prob = teacher_forcing_prob
        # self.padding_idx = padding_idx

        self.vocab = build_vocab()
        self.vocab_size = len(self.vocab)
        self.slots = load_slots()
        gate_dict = load_gate_dict()
        self.gate_size = len(gate_dict)

        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
            # padding_idx=self.padding_idx,
        )
        self.encoder = Encoder(
            embedding=self.embedding,
            hidden_size=self.hidden_size,
            dropout_p=self.dropout_p,
        )
        self.decoder = Decoder(
            embedding=self.embedding,
            hidden_size=self.hidden_size,
            dropout_p=self.dropout_p,
            vocab_size=self.vocab_size,
            slots=self.slots,
            gate_size=self.gate_size,
            teacher_forcing_prob=self.teacher_forcing_prob,
        )
        
    def load_embedding(self):
        embeddings = build_embeddings()
        new = self.embedding.weight.data.new  # NOTE
        self.embedding.weight.data.copy_(new(embeddings))

    def forward(self, batch):
        preds_probs, gates_logits = self.encode_and_decode(batch)
        # preds_probs: B * S * K * V
        # gates_logits: B * S * 3
        
        if self.training:
            loss_gate, loss_ptr = self.loss_fn(
                gates_logits=gates_logits,
                gates_target=batch['gates_idx'],
                preds_probs=preds_probs,
                values_target=batch['values_idx'],
                values_mask=batch['values_mask'],
            )
            loss = loss_ptr + loss_gate
            result = {
                'loss': loss,
                'loss_gate': loss_gate,
                'loss_ptr': loss_ptr,
            }
            return result
        else:
            preds_idx = preds_probs.argmax(dim=-1)  # B * S * K * V -> B * S * K
            gates_idx = gates_logits.argmax(dim=-1)  # B * S * 3 -> B * S
            result = {
                'preds_idx': preds_idx,
                'gates_idx': gates_idx,
            }
            return result

    def encode_and_decode(self, batch):
        # In training, random mask some idx to unk
        context_idx = batch['context_idx']
        if self.training:
            probs = torch.full_like(context_idx, self.dropout_p, dtype=torch.float32)
            mask = torch.bernoulli(probs).bool()
            context_idx = context_idx.masked_fill(mask, self.vocab.unk_idx)

        outputs, hidden = self.encoder(
            context_idx=context_idx,
            lengths=batch['context_lens'],
        )

        preds_probs, gates_logits = self.decoder(
            encoder_hidden=hidden,
            encoder_outputs=outputs,
            context_idx=context_idx,
            context_mask=batch['context_mask'],
            target_values=batch['values_idx'] if self.training else None,
        )
        return preds_probs, gates_logits

    def loss_fn(self, gates_logits, gates_target, preds_probs, values_target, values_mask):
        loss_gate = torch.nn.functional.cross_entropy(
            input=gates_logits.reshape(-1, gates_logits.shape[-1]),
            target=gates_target.reshape(-1),
            reduction='mean',
        )
        loss_ptr = torch.nn.functional.nll_loss(
            input=preds_probs.log().reshape(-1, preds_probs.shape[-1]),
            target=values_target.reshape(-1),
            reduction='none',
        )
        values_mask = values_mask.flatten()
        loss_ptr = (loss_ptr * values_mask).sum() / values_mask.sum()
        return loss_gate, loss_ptr


class Encoder(torch.nn.Module):

    def __init__(self, embedding, hidden_size, dropout_p):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self.embedding = embedding
        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.gru = torch.nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=False,
            bidirectional=True,
        )

    def forward(self, context_idx, lengths):
        # context_idx: B * L
        # lengths: List[int], B
        context_idx = context_idx.transpose(0, 1)  # B * L -> L * B
        embeddings = self.embedding(context_idx)  # L * B -> L * B * H
        embeddings = self.dropout(embeddings)  # L * B * H
        embeddings = torch.nn.utils.rnn.pack_padded_sequence(
            input=embeddings,
            lengths=lengths,
            batch_first=False,
            enforce_sorted=True,
        )
        outputs, hidden = self.gru(embeddings)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=outputs,
            batch_first=False,
            padding_value=0.0, # NOTE
        )
        # output: L * B * DH (L * B * 2H)
        # hidden: DN * B * H (2 * B * H)

        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]  # L * B * H
        hidden = hidden[0] + hidden[1]  # 2 * B * H -> B * H
        outputs = outputs.transpose(0, 1)  # L * B * H -> B * L * H
        return outputs, hidden


class Decoder(torch.nn.Module):

    def __init__(self, embedding, hidden_size, dropout_p, vocab_size, slots, gate_size, teacher_forcing_prob):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout_p
        self.vocab_size = vocab_size
        self.slots = slots
        self.slots_size = len(self.slots)
        self.gate_size = gate_size
        self.teacher_forcing_prob = teacher_forcing_prob

        self.embedding = embedding
        self.dropout = torch.nn.Dropout(self.dropout)
        self.gru_cell = torch.nn.GRUCell(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
        )
        self.w_ratio = torch.nn.Linear(3 * self.hidden_size, 1)
        self.w_gate = torch.nn.Linear(self.hidden_size, self.gate_size)
        self.output_layer = torch.nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.output_layer.weight = self.embedding.weight  # V * H

        # Slot embeddings
        self.slot_token2idx = {}
        for s in self.slots:
            domain, slot = s.split('-')
            if domain not in self.slot_token2idx:
                self.slot_token2idx[domain] = len(self.slot_token2idx)
            if slot not in self.slot_token2idx.keys():
                self.slot_token2idx[slot] = len(self.slot_token2idx)
        self.slot_embedding = torch.nn.Embedding(len(self.slot_token2idx), self.hidden_size)
        self.slot_embedding.weight.data.normal_(0, 0.1)

    def forward(self, encoder_hidden, encoder_outputs, context_idx, context_mask, target_values):
        # encoder_hidden: B * H
        # encoder_outputs: B * L * H
        # context_idx: B * L
        # context_lens: List[int], B
        # target_values: B * S * K
        batch_size = context_idx.shape[0]
        device = context_idx.device
        
        # Slot embeddings
        domains = [self.slot_token2idx[s.split('-')[0]] for s in self.slots]
        domains = torch.tensor(domains, device=device)
        domains = self.slot_embedding(domains)  # S * H
        slots = [self.slot_token2idx[s.split('-')[1]] for s in self.slots]
        slots = torch.tensor(slots, device=device)
        slots = self.slot_embedding(slots)  # S * H
        slot_embeddings = domains + slots  # S * H

        slot_embeddings = slot_embeddings.to(device)  # S * H
        slot_embeddings = slot_embeddings.unsqueeze(1) \
            .expand(-1, batch_size, -1) \
            .reshape(-1, slot_embeddings.shape[-1])  # S * H -> S * 1 * H -> S * B * H -> SB * H
        embeddings = slot_embeddings  # SB * H

        # Expand
        hidden = encoder_hidden.expand(self.slots_size, -1, -1) \
            .reshape(-1, encoder_hidden.shape[-1])  # B * H -> SB * H
        encoder_outputs = encoder_outputs.expand(self.slots_size, -1, -1, -1) \
            .reshape(-1, *encoder_outputs.shape[-2:])  # B * L * H -> SB * L * H
        context_idx = context_idx.expand(self.slots_size, -1, -1) \
            .reshape(-1, context_idx.shape[-1])  # B * L -> SB * L
        context_mask = context_mask.expand(self.slots_size, -1, -1) \
            .reshape(-1, context_mask.shape[-1])  # B * L -> SB * L

        if self.training:
            target_values = target_values.transpose(0, 1) \
                .reshape(-1, target_values.shape[-1])  # B * S * K -> SB * K

        preds_probs = []
        use_teacher_forcing = self.training and random.random() < self.teacher_forcing_prob
        max_value_len = target_values.shape[-1] if self.training else 10
        for i in range(max_value_len):
            embeddings = self.dropout(embeddings)
            hidden = self.gru_cell(embeddings, hidden)  # BS * H

            # p_vocab
            p_vocab = self.output_layer(hidden).softmax(dim=-1)  # BS * V

            # p_context
            context_probs = torch.bmm(encoder_outputs, hidden.unsqueeze(2)).squeeze(2)  # SB * L * H x SB * H * 1 -> SB * L
            context_probs = context_probs.masked_fill((1 - context_mask).bool(), float('-inf'))  # SB * L
            context_probs = context_probs.softmax(dim=-1)  # SB * L
            p_context = torch.zeros_like(p_vocab)  # SB * V
            p_context.scatter_add_(dim=1, index=context_idx, src=context_probs)  # SB * V

            # p_gen
            context_vec = (encoder_outputs * context_probs.unsqueeze(2)).sum(dim=1)
            p_gen_vec = torch.cat([hidden, context_vec, embeddings], dim=-1)  # SB * 3H
            p_gen = self.w_ratio(p_gen_vec).sigmoid()  # SB * 1

            # gate
            if i == 0:
                gates_logits = self.w_gate(context_vec)  # SB * H -> SB * 3
                gates_logits = gates_logits.reshape(self.slots_size, batch_size, -1)  # SB * 3 -> S * B * 3

            # p_final
            p_final = (1 - p_gen) * p_context + p_gen * p_vocab  # SB * V
            pred_idx = p_final.argmax(dim=-1)  # SB
            p_final = p_final.reshape(self.slots_size, batch_size, self.vocab_size)  # SB * V -> S * B * V
            preds_probs.append(p_final)  # List[S * B * V], K

            # next
            if use_teacher_forcing:
                embeddings = self.embedding(target_values[:, i])  # SB * K -> SB -> SB * H
            else:
                embeddings = self.embedding(pred_idx)  # SB -> SB * H
            
        preds_probs = torch.stack(preds_probs, dim=2)  # List[S * B * V], K -> S * B * K * V
        preds_probs = preds_probs.transpose(0, 1)  # S * B * K * V -> B * S * K * V
        gates_logits = gates_logits.transpose(0, 1)  # S * B * 3 -> B * S * 3

        # preds_probs: B * S * K * V
        # gates_logits: B * S * 3
        return preds_probs, gates_logits
