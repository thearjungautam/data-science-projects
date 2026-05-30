import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerNMT(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        emb_dim=512,
        n_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        ff_dim=2048,
        dropout=0.1,
        src_pad_idx=0,
        trg_pad_idx=0,
        sos_idx=2,
        eos_idx=3,
        max_len=512,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

        self.src_embedding = nn.Embedding(src_vocab_size, emb_dim, padding_idx=src_pad_idx)
        self.trg_embedding = nn.Embedding(trg_vocab_size, emb_dim, padding_idx=trg_pad_idx)

        self.pos_encoder = PositionalEncoding(emb_dim, dropout=dropout, max_len=max_len)
        self.pos_decoder = PositionalEncoding(emb_dim, dropout=dropout, max_len=max_len)

        self.transformer = nn.Transformer(
            d_model=emb_dim,
            nhead=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )

        self.fc_out = nn.Linear(emb_dim, trg_vocab_size)

    def make_src_key_padding_mask(self, src):
        return src == self.src_pad_idx

    def make_trg_key_padding_mask(self, trg):
        return trg == self.trg_pad_idx

    def make_tgt_mask(self, tgt_len, device):
        return torch.triu(torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        trg_input = trg[:, :-1]
        src_key_padding_mask = self.make_src_key_padding_mask(src)
        trg_key_padding_mask = self.make_trg_key_padding_mask(trg_input)
        tgt_mask = self.make_tgt_mask(trg_input.size(1), src.device)

        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.emb_dim))
        trg_emb = self.pos_decoder(self.trg_embedding(trg_input) * math.sqrt(self.emb_dim))

        output = self.transformer(
            src=src_emb,
            tgt=trg_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=trg_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.fc_out(output)

    @torch.no_grad()
    def greedy_decode(self, src, src_lengths, max_len=50):
        batch_size = src.size(0)
        src_key_padding_mask = self.make_src_key_padding_mask(src)

        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.emb_dim))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        ys = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=src.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            tgt_mask = self.make_tgt_mask(ys.size(1), src.device)
            tgt_key_padding_mask = self.make_trg_key_padding_mask(ys)
            tgt_emb = self.pos_decoder(self.trg_embedding(ys) * math.sqrt(self.emb_dim))

            out = self.transformer.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            out = self.fc_out(out[:, -1:, :])
            next_token = out.argmax(dim=-1)
            ys = torch.cat([ys, next_token], dim=1)
            finished |= (next_token.squeeze(1) == self.eos_idx)
            if finished.all():
                break

        return ys


def build_model(cfg, meta, device):
    return TransformerNMT(
        src_vocab_size=len(meta["src_vocab"]),
        trg_vocab_size=len(meta["trg_vocab"]),
        emb_dim=cfg.tf_emb_dim,
        n_heads=cfg.tf_heads,
        num_encoder_layers=cfg.tf_enc_layers,
        num_decoder_layers=cfg.tf_dec_layers,
        ff_dim=cfg.tf_ff_dim,
        dropout=cfg.tf_dropout,
        src_pad_idx=meta["src_pad_idx"],
        trg_pad_idx=meta["trg_pad_idx"],
        sos_idx=meta["trg_sos_idx"],
        eos_idx=meta["trg_eos_idx"],
        max_len=cfg.max_len + 5,
    ).to(device)