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
        packed_outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=src.size(1)
        )
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask == 0, -1e4)
        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, attention, num_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(
            emb_dim + hid_dim,
            hid_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hid_dim * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tokens, hidden, encoder_outputs, mask):
        input_tokens = input_tokens.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_tokens))
        top_hidden = hidden[-1]
        attn_weights = self.attention(top_hidden, encoder_outputs, mask).unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc_out(
            torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1)
        )
        return prediction, hidden, attn_weights.squeeze(1)


class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device, sos_idx, eos_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def create_mask(self, src):
        return src != self.src_pad_idx

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        output_dim = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, output_dim, device=self.device)

        encoder_outputs, hidden = self.encoder(src, src_lengths)
        mask = self.create_mask(src)
        input_tokens = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input_tokens, hidden, encoder_outputs, mask)
            outputs[:, t, :] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(dim=1)
            input_tokens = trg[:, t] if teacher_force else top1

        return outputs

    @torch.no_grad()
    def greedy_decode(self, src, src_lengths, max_len=50):
        batch_size = src.size(0)
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        mask = self.create_mask(src)

        input_tokens = torch.full(
            (batch_size,), self.sos_idx, dtype=torch.long, device=self.device
        )
        generated = [input_tokens.unsqueeze(1)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(max_len):
            output, hidden, _ = self.decoder(input_tokens, hidden, encoder_outputs, mask)
            top1 = output.argmax(dim=1)
            generated.append(top1.unsqueeze(1))
            finished |= (top1 == self.eos_idx)
            input_tokens = top1
            if finished.all():
                break

        return torch.cat(generated, dim=1)


def build_model(cfg, meta, device):
    attn = Attention(cfg.hid_dim)
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
        attention=attn,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout_rnn,
        pad_idx=meta["trg_pad_idx"],
    )
    return Seq2SeqAttention(
        enc, dec, meta["src_pad_idx"], device, meta["trg_sos_idx"], meta["trg_eos_idx"]
    ).to(device)