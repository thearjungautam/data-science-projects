import os
import re
import random
from collections import Counter
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    text = clean_text(text)
    if lowercase:
        text = text.lower()
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+'", "'", text)
    return text.strip()


def build_vocab(token_lists, min_freq: int = 2) -> dict[str, int]:
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    for token, freq in counter.items():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def numericalize(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    unk = vocab["<unk>"]
    sos = vocab["<sos>"]
    eos = vocab["<eos>"]
    return [sos] + [vocab.get(tok, unk) for tok in tokens] + [eos]


class TranslationDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        src_ids = torch.tensor(self.df.iloc[idx]["src_ids"], dtype=torch.long)
        trg_ids = torch.tensor(self.df.iloc[idx]["trg_ids"], dtype=torch.long)
        return src_ids, trg_ids


def make_collate_fn(src_pad_idx: int, trg_pad_idx: int):
    def collate_fn(batch):
        src_batch = [item[0] for item in batch]
        trg_batch = [item[1] for item in batch]

        src_lengths = torch.tensor([len(x) for x in src_batch], dtype=torch.long)
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=src_pad_idx)
        trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=trg_pad_idx)

        return {
            "src": src_padded,
            "src_lengths": src_lengths,
            "trg": trg_padded,
        }

    return collate_fn


def _prepare_split(df: pd.DataFrame, lowercase: bool, max_len: int, src_vocab=None, trg_vocab=None):
    df = df[["en", "de"]].dropna().reset_index(drop=True)
    df.columns = ["src", "trg"]

    df["src"] = df["src"].apply(clean_text)
    df["trg"] = df["trg"].apply(clean_text)

    df["src_tokens"] = df["src"].apply(lambda x: tokenize(x, lowercase=lowercase))
    df["trg_tokens"] = df["trg"].apply(lambda x: tokenize(x, lowercase=lowercase))

    if src_vocab is None or trg_vocab is None:
        return df

    df["src_ids"] = df["src_tokens"].apply(lambda x: numericalize(x, src_vocab))
    df["trg_ids"] = df["trg_tokens"].apply(lambda x: numericalize(x, trg_vocab))

    df = df[
        df["src_ids"].apply(len).le(max_len) &
        df["trg_ids"].apply(len).le(max_len)
    ].reset_index(drop=True)

    return df


def load_data(cfg):
    dataset_path = os.path.expandvars(cfg.dataset_path)
    ds = load_dataset("wmt14", "de-en")

    train_df = pd.DataFrame(ds["train"]["translation"])
    valid_df = pd.DataFrame(ds["validation"]["translation"])
    test_df = pd.DataFrame(ds["test"]["translation"])

    train_df = train_df.sample(min(cfg.max_train, len(train_df)), random_state=cfg.seed).reset_index(drop=True)
    valid_df = valid_df.sample(min(cfg.max_valid, len(valid_df)), random_state=cfg.seed).reset_index(drop=True)
    test_df = test_df.sample(min(cfg.max_test, len(test_df)), random_state=cfg.seed).reset_index(drop=True)

    train_df = _prepare_split(train_df, cfg.lowercase, cfg.max_len)
    valid_df = _prepare_split(valid_df, cfg.lowercase, cfg.max_len)
    test_df = _prepare_split(test_df, cfg.lowercase, cfg.max_len)

    src_vocab = build_vocab(train_df["src_tokens"], min_freq=2)
    trg_vocab = build_vocab(train_df["trg_tokens"], min_freq=2)

    train_df = _prepare_split(train_df.rename(columns={"src": "en", "trg": "de"}), cfg.lowercase, cfg.max_len, src_vocab, trg_vocab)
    valid_df = _prepare_split(valid_df.rename(columns={"src": "en", "trg": "de"}), cfg.lowercase, cfg.max_len, src_vocab, trg_vocab)
    test_df = _prepare_split(test_df.rename(columns={"src": "en", "trg": "de"}), cfg.lowercase, cfg.max_len, src_vocab, trg_vocab)

    src_itos = {idx: tok for tok, idx in src_vocab.items()}
    trg_itos = {idx: tok for tok, idx in trg_vocab.items()}

    src_pad_idx = src_vocab["<pad>"]
    trg_pad_idx = trg_vocab["<pad>"]

    collate_fn = make_collate_fn(src_pad_idx, trg_pad_idx)

    train_loader = DataLoader(
        TranslationDataset(train_df),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        TranslationDataset(valid_df),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        TranslationDataset(test_df),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    meta = {
        "src_vocab": src_vocab,
        "trg_vocab": trg_vocab,
        "src_itos": src_itos,
        "trg_itos": trg_itos,
        "src_pad_idx": src_pad_idx,
        "trg_pad_idx": trg_pad_idx,
        "src_sos_idx": src_vocab["<sos>"],
        "src_eos_idx": src_vocab["<eos>"],
        "src_unk_idx": src_vocab["<unk>"],
        "trg_sos_idx": trg_vocab["<sos>"],
        "trg_eos_idx": trg_vocab["<eos>"],
        "trg_unk_idx": trg_vocab["<unk>"],
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
    }

    return train_loader, valid_loader, test_loader, meta