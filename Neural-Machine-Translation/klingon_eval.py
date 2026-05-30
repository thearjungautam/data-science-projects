import os
import re
import math
import json
import time
import random
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, DatasetDict
import sacrebleu
from tqdm import tqdm




def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate a Seq2Seq+Attention model on English-Klingon")
    parser.add_argument("--dataset", type=str, default="MihaiPopa-1/custom-klingon-33k",
                        help="HF dataset id. Best first try: MihaiPopa-1/custom-klingon-33k")
    parser.add_argument("--fallback_dataset", type=str, default="ymoslem/Tatoeba-Translations",
                        help="Fallback dataset id if primary load fails")
    parser.add_argument("--lang1", type=str, default="eng")
    parser.add_argument("--lang2", type=str, default="tlh")
    parser.add_argument("--max_train", type=int, default=30000)
    parser.add_argument("--max_valid", type=int, default=2000)
    parser.add_argument("--max_test", type=int, default=2000)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--hid_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="./klingon_results")
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--lowercase", action="store_true")
    return parser.parse_args()



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str, lowercase: bool = False) -> list[str]:
    text = clean_text(text)
    if lowercase:
        text = text.lower()
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text.strip()


SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>"]


def build_vocab(token_lists, min_freq=1):
    counter = Counter()
    for toks in token_lists:
        counter.update(toks)
    vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    for tok, freq in counter.items():
        if freq >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def numericalize(tokens, vocab):
    unk = vocab["<unk>"]
    sos = vocab["<sos>"]
    eos = vocab["<eos>"]
    return [sos] + [vocab.get(tok, unk) for tok in tokens] + [eos]


def ids_to_tokens(ids, itos, eos_idx, pad_idx, sos_idx):
    out = []
    for idx in ids:
        if idx == eos_idx:
            break
        if idx in (pad_idx, sos_idx):
            continue
        out.append(itos.get(idx, "<unk>"))
    return out



def infer_parallel_df(split):
    """
    Convert a HF dataset split into a 2-column dataframe: src, trg.
    Handles:
    - normal named columns
    - translation dict format
    - weird parquet where first row became headers
    """
    import pandas as pd

    df = split.to_pandas()

    
    if "translation" in df.columns:
        trans_df = pd.DataFrame(df["translation"].tolist())
        
        if "en" in trans_df.columns and "tlh" in trans_df.columns:
            return trans_df[["en", "tlh"]].rename(columns={"en": "src", "tlh": "trg"})
        
        if trans_df.shape[1] >= 2:
            trans_df = trans_df.iloc[:, :2].copy()
            trans_df.columns = ["src", "trg"]
            return trans_df

    
    lower_cols = [str(c).lower() for c in df.columns]

    possible_src = ["en", "eng", "english", "source", "src"]
    possible_trg = ["tlh", "klingon", "target", "trg"]

    src_col = None
    trg_col = None

    for c, lc in zip(df.columns, lower_cols):
        if lc in possible_src:
            src_col = c
        if lc in possible_trg:
            trg_col = c

    if src_col is not None and trg_col is not None:
        out = df[[src_col, trg_col]].copy()
        out.columns = ["src", "trg"]
        return out

    
    if df.shape[1] >= 4:
        out = df.iloc[:, [1, 3]].copy()
        out.columns = ["src", "trg"]
        return out

    
    if df.shape[1] >= 2:
        text_cols = []
        for col in df.columns:
            sample = df[col].dropna().astype(str).head(5)
            avg_len = sample.map(len).mean() if len(sample) else 0
            if avg_len > 2:
                text_cols.append(col)

        if len(text_cols) >= 2:
            out = df[[text_cols[0], text_cols[1]]].copy()
            out.columns = ["src", "trg"]
            return out

        out = df.iloc[:, :2].copy()
        out.columns = ["src", "trg"]
        return out

    raise ValueError(f"Could not infer source/target columns from dataset columns: {list(df.columns)}")


def load_klingon_dataset(args) -> DatasetDict:
    try:
        ds = load_dataset(args.dataset)
        print(f"Loaded primary dataset: {args.dataset}")
    except Exception as e:
        print(f"Primary dataset load failed: {e}")
        print(f"Trying fallback dataset: {args.fallback_dataset}")
        ds = load_dataset(args.fallback_dataset, lang1=args.lang1, lang2=args.lang2)
        print(f"Loaded fallback dataset: {args.fallback_dataset}")

    
    if isinstance(ds, DatasetDict):
        if "train" in ds and "validation" in ds and "test" in ds:
            return ds
        if "train" in ds and "test" not in ds:
            split_1 = ds["train"].train_test_split(test_size=0.1, seed=args.seed)
            split_2 = split_1["train"].train_test_split(test_size=0.1, seed=args.seed)
            return DatasetDict({
                "train": split_2["train"],
                "validation": split_2["test"],
                "test": split_1["test"],
            })
        if "train" in ds and "test" in ds and "validation" not in ds:
            split = ds["train"].train_test_split(test_size=0.1, seed=args.seed)
            return DatasetDict({
                "train": split["train"],
                "validation": split["test"],
                "test": ds["test"],
            })

    
    if hasattr(ds, "train_test_split"):
        split_1 = ds.train_test_split(test_size=0.1, seed=args.seed)
        split_2 = split_1["train"].train_test_split(test_size=0.1, seed=args.seed)
        return DatasetDict({
            "train": split_2["train"],
            "validation": split_2["test"],
            "test": split_1["test"],
        })

    raise ValueError("Could not normalize dataset into train/validation/test splits.")


def prepare_data(args):
    ds = load_klingon_dataset(args)

    train_df = infer_parallel_df(ds["train"])
    valid_df = infer_parallel_df(ds["validation"])
    test_df = infer_parallel_df(ds["test"])

    train_df = train_df.dropna().reset_index(drop=True)
    valid_df = valid_df.dropna().reset_index(drop=True)
    test_df = test_df.dropna().reset_index(drop=True)

    train_df = train_df.sample(min(args.max_train, len(train_df)), random_state=args.seed).reset_index(drop=True)
    valid_df = valid_df.sample(min(args.max_valid, len(valid_df)), random_state=args.seed).reset_index(drop=True)
    test_df = test_df.sample(min(args.max_test, len(test_df)), random_state=args.seed).reset_index(drop=True)

    for df in (train_df, valid_df, test_df):
        df["src"] = df["src"].apply(clean_text)
        df["trg"] = df["trg"].apply(clean_text)
        df["src_tokens"] = df["src"].apply(lambda x: tokenize(x, lowercase=args.lowercase))
        df["trg_tokens"] = df["trg"].apply(lambda x: tokenize(x, lowercase=args.lowercase))

    src_vocab = build_vocab(train_df["src_tokens"], min_freq=args.min_freq)
    trg_vocab = build_vocab(train_df["trg_tokens"], min_freq=args.min_freq)

    for df in (train_df, valid_df, test_df):
        df["src_ids"] = df["src_tokens"].apply(lambda x: numericalize(x, src_vocab))
        df["trg_ids"] = df["trg_tokens"].apply(lambda x: numericalize(x, trg_vocab))

    train_df = train_df[train_df["src_ids"].apply(len).le(args.max_len) & train_df["trg_ids"].apply(len).le(args.max_len)].reset_index(drop=True)
    valid_df = valid_df[valid_df["src_ids"].apply(len).le(args.max_len) & valid_df["trg_ids"].apply(len).le(args.max_len)].reset_index(drop=True)
    test_df = test_df[test_df["src_ids"].apply(len).le(args.max_len) & test_df["trg_ids"].apply(len).le(args.max_len)].reset_index(drop=True)

    src_itos = {v: k for k, v in src_vocab.items()}
    trg_itos = {v: k for k, v in trg_vocab.items()}

    meta = {
        "src_vocab": src_vocab,
        "trg_vocab": trg_vocab,
        "src_itos": src_itos,
        "trg_itos": trg_itos,
        "src_pad_idx": src_vocab["<pad>"],
        "trg_pad_idx": trg_vocab["<pad>"],
        "src_sos_idx": src_vocab["<sos>"],
        "src_eos_idx": src_vocab["<eos>"],
        "src_unk_idx": src_vocab["<unk>"],
        "trg_sos_idx": trg_vocab["<sos>"],
        "trg_eos_idx": trg_vocab["<eos>"],
        "trg_unk_idx": trg_vocab["<unk>"],
    }

    return train_df, valid_df, test_df, meta




class TranslationDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src_ids = torch.tensor(self.df.iloc[idx]["src_ids"], dtype=torch.long)
        trg_ids = torch.tensor(self.df.iloc[idx]["trg_ids"], dtype=torch.long)
        return src_ids, trg_ids


def make_collate_fn(src_pad_idx, trg_pad_idx):
    def collate_fn(batch):
        src_batch = [x[0] for x in batch]
        trg_batch = [x[1] for x in batch]
        src_lengths = torch.tensor([len(x) for x in src_batch], dtype=torch.long)
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=src_pad_idx)
        trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=trg_pad_idx)
        return {"src": src_padded, "src_lengths": src_lengths, "trg": trg_padded}
    return collate_fn




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
        prediction = self.fc_out(torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1))
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

        input_tokens = torch.full((batch_size,), self.sos_idx, dtype=torch.long, device=self.device)
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




def epoch_time(start, end):
    elapsed = end - start
    return int(elapsed // 60), int(elapsed % 60)


def train_epoch(model, dataloader, optimizer, criterion, clip, device, teacher_forcing_ratio=0.5):
    model.train()
    epoch_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        src = batch["src"].to(device, non_blocking=True)
        src_lengths = batch["src_lengths"].to(device, non_blocking=True)
        trg = batch["trg"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        output = model(src, src_lengths, trg, teacher_forcing_ratio=teacher_forcing_ratio)
        output_dim = output.shape[-1]
        output = output[:, 1:, :].reshape(-1, output_dim)
        trg_y = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        src = batch["src"].to(device, non_blocking=True)
        src_lengths = batch["src_lengths"].to(device, non_blocking=True)
        trg = batch["trg"].to(device, non_blocking=True)

        output = model(src, src_lengths, trg, teacher_forcing_ratio=0.0)
        output_dim = output.shape[-1]
        output = output[:, 1:, :].reshape(-1, output_dim)
        trg_y = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg_y)
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


@torch.no_grad()
def compute_bleu(model, dataloader, trg_itos, meta, device, max_len=50):
    model.eval()
    predictions, references = [], []

    for batch in tqdm(dataloader, desc="Greedy BLEU", leave=False):
        src = batch["src"].to(device, non_blocking=True)
        src_lengths = batch["src_lengths"].to(device, non_blocking=True)
        trg = batch["trg"].to(device, non_blocking=True)

        pred_ids = model.greedy_decode(src, src_lengths, max_len=max_len)

        for i in range(src.size(0)):
            pred_tokens = ids_to_tokens(
                pred_ids[i].tolist(), trg_itos,
                meta["trg_eos_idx"], meta["trg_pad_idx"], meta["trg_sos_idx"]
            )
            gold_tokens = ids_to_tokens(
                trg[i].tolist(), trg_itos,
                meta["trg_eos_idx"], meta["trg_pad_idx"], meta["trg_sos_idx"]
            )
            predictions.append(detokenize(pred_tokens))
            references.append(detokenize(gold_tokens))

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score, predictions, references


@torch.no_grad()
def beam_search_decode_attention(model, src, src_lengths, beam_width=5, max_len=50):
    model.eval()
    device = src.device

    encoder_outputs, hidden = model.encoder(src, src_lengths)
    seq_len = encoder_outputs.size(1)
    mask = (torch.arange(seq_len, device=device).unsqueeze(0) < src_lengths.unsqueeze(1))

    beams = [([model.sos_idx], 0.0, hidden)]
    completed = []

    for _ in range(max_len):
        new_beams = []

        for seq, score, hidden_state in beams:
            last_token = seq[-1]

            if last_token == model.eos_idx:
                completed.append((seq, score))
                new_beams.append((seq, score, hidden_state))
                continue

            input_token = torch.tensor([last_token], dtype=torch.long, device=device)
            output, new_hidden, _ = model.decoder(input_token, hidden_state, encoder_outputs, mask)
            log_probs = F.log_softmax(output, dim=1)
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_width, dim=1)

            for k in range(beam_width):
                next_token = topk_ids[0, k].item()
                next_score = score + topk_log_probs[0, k].item()
                next_seq = seq + [next_token]
                new_beams.append((next_seq, next_score, new_hidden))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        if all(seq[-1] == model.eos_idx for seq, _, _ in beams):
            break

    if completed:
        best_seq = sorted(completed, key=lambda x: x[1], reverse=True)[0][0]
    else:
        best_seq = beams[0][0]
    return best_seq


@torch.no_grad()
def compute_bleu_beam(model, dataloader, trg_itos, meta, device, beam_width=5, max_len=50):
    model.eval()
    predictions, references = [], []

    for batch in tqdm(dataloader, desc=f"Beam={beam_width} BLEU", leave=False):
        src = batch["src"].to(device, non_blocking=True)
        src_lengths = batch["src_lengths"].to(device, non_blocking=True)
        trg = batch["trg"].to(device, non_blocking=True)

        for i in range(src.size(0)):
            pred_ids = beam_search_decode_attention(
                model,
                src[i].unsqueeze(0),
                src_lengths[i].unsqueeze(0),
                beam_width=beam_width,
                max_len=max_len,
            )

            pred_tokens = ids_to_tokens(
                pred_ids, trg_itos,
                meta["trg_eos_idx"], meta["trg_pad_idx"], meta["trg_sos_idx"]
            )
            gold_tokens = ids_to_tokens(
                trg[i].tolist(), trg_itos,
                meta["trg_eos_idx"], meta["trg_pad_idx"], meta["trg_sos_idx"]
            )

            predictions.append(detokenize(pred_tokens))
            references.append(detokenize(gold_tokens))

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score, predictions, references




def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Loading Klingon dataset...")

    train_df, valid_df, test_df, meta = prepare_data(args)

    print(f"Train size: {len(train_df)}")
    print(f"Valid size: {len(valid_df)}")
    print(f"Test size : {len(test_df)}")
    print(f"Src vocab : {len(meta['src_vocab'])}")
    print(f"Trg vocab : {len(meta['trg_vocab'])}")

    collate_fn = make_collate_fn(meta["src_pad_idx"], meta["trg_pad_idx"])

    train_loader = DataLoader(TranslationDataset(train_df), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(TranslationDataset(valid_df), batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(TranslationDataset(test_df), batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=2, pin_memory=torch.cuda.is_available())

    attention = Attention(args.hid_dim)
    encoder = Encoder(
        input_dim=len(meta["src_vocab"]),
        emb_dim=args.emb_dim,
        hid_dim=args.hid_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_idx=meta["src_pad_idx"],
    )
    decoder = Decoder(
        output_dim=len(meta["trg_vocab"]),
        emb_dim=args.emb_dim,
        hid_dim=args.hid_dim,
        attention=attention,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_idx=meta["trg_pad_idx"],
    )
    model = Seq2SeqAttention(
        encoder,
        decoder,
        meta["src_pad_idx"],
        device,
        meta["trg_sos_idx"],
        meta["trg_eos_idx"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=meta["trg_pad_idx"])

    best_valid_loss = float("inf")
    epochs_no_improve = 0
    checkpoint_path = os.path.join(args.out_dir, "best_klingon_attention.pt")
    history = []

    print("\nTraining model...")
    for epoch in range(args.epochs):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.clip, device, args.teacher_forcing_ratio)
        valid_loss = evaluate_epoch(model, valid_loader, criterion, device)
        mins, secs = epoch_time(start, time.time())

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "valid_loss": valid_loss})
        print(f"Epoch: {epoch+1:02d} | Time: {mins}m {secs}s")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Valid Loss: {valid_loss:.4f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print("Early stopping triggered.")
            break

    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "klingon_attention_history.csv"), index=False)

    print("\nLoading best checkpoint...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print("\nEvaluating greedy BLEU...")
    greedy_bleu, _, _ = compute_bleu(model, test_loader, meta["trg_itos"], meta, device, max_len=args.max_len)
    print(f"Greedy BLEU: {greedy_bleu:.2f}")

    print("\nEvaluating beam=5 BLEU...")
    beam_bleu, _, _ = compute_bleu_beam(model, test_loader, meta["trg_itos"], meta, device, beam_width=args.beam_width, max_len=args.max_len)
    print(f"Beam={args.beam_width} BLEU: {beam_bleu:.2f}")

    summary = {
        "dataset": args.dataset,
        "fallback_dataset": args.fallback_dataset,
        "train_size": len(train_df),
        "valid_size": len(valid_df),
        "test_size": len(test_df),
        "architecture": {
            "model": "Seq2Seq+Attention",
            "num_layers": args.num_layers,
            "hid_dim": args.hid_dim,
            "emb_dim": args.emb_dim,
            "dropout": args.dropout,
        },
        "greedy_bleu": greedy_bleu,
        f"beam_bleu_{args.beam_width}": beam_bleu,
    }

    with open(os.path.join(args.out_dir, "klingon_attention_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved outputs to:", args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
