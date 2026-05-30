import random
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, num_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(
            emb_dim,
            hid_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, num_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(
            emb_dim,
            hid_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tokens, hidden):
        input_tokens = input_tokens.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_tokens))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, sos_idx, eos_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        output_dim = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, output_dim, device=self.device)
        hidden = self.encoder(src, src_lengths)
        input_tokens = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input_tokens, hidden)
            outputs[:, t, :] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(dim=1)
            input_tokens = trg[:, t] if teacher_force else top1

        return outputs

    @torch.no_grad()
    def greedy_decode(self, src, src_lengths, max_len=50):
        batch_size = src.size(0)
        hidden = self.encoder(src, src_lengths)

        input_tokens = torch.full(
            (batch_size,), self.sos_idx, dtype=torch.long, device=self.device
        )
        generated = [input_tokens.unsqueeze(1)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(max_len):
            output, hidden = self.decoder(input_tokens, hidden)
            top1 = output.argmax(dim=1)
            generated.append(top1.unsqueeze(1))
            finished |= (top1 == self.eos_idx)
            input_tokens = top1
            if finished.all():
                break

        return torch.cat(generated, dim=1)


def build_model(cfg, meta, device):
    enc = Encoder(
        input_dim=len(meta["src_vocab"]),
        emb_dim=cfg.emb_dim,
        hid_dim=cfg.hid_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout_rnn,
        pad_idx=meta["src_pad_idx"],
    )
    dec = Decoder(
        output_dim=len(meta["trg_vocab"]),
        emb_dim=cfg.emb_dim,
        hid_dim=cfg.hid_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout_rnn,
        pad_idx=meta["trg_pad_idx"],
    )
    return Seq2Seq(enc, dec, device, meta["trg_sos_idx"], meta["trg_eos_idx"]).to(device)