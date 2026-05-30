import os
import pandas as pd
import torch

from configs import Config
from data_utils import load_data, set_seed, clean_text, tokenize, detokenize
import seq2seq
import seq2seq_attention


def ids_to_tokens(ids, itos, eos_idx, pad_idx, sos_idx):
    tokens = []
    for idx in ids:
        if idx == eos_idx:
            break
        if idx in (pad_idx, sos_idx):
            continue
        tokens.append(itos.get(idx, "<unk>"))
    return tokens


def length_bucket(n):
    if n <= 15:
        return "short"
    elif n <= 35:
        return "medium"
    else:
        return "long"


@torch.no_grad()
def translate_sentence_rnn(model, sentence, src_vocab, trg_itos, meta, device, max_len=50):
    model.eval()

    tokens = tokenize(clean_text(sentence), lowercase=True)
    src_ids = [meta["src_sos_idx"]] + [src_vocab.get(tok, meta["src_unk_idx"]) for tok in tokens] + [meta["src_eos_idx"]]

    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    src_lengths = torch.tensor([len(src_ids)], dtype=torch.long).to(device)

    pred_ids = model.greedy_decode(src_tensor, src_lengths, max_len=max_len)

    pred_tokens = ids_to_tokens(
        pred_ids.squeeze(0).tolist(),
        trg_itos,
        eos_idx=meta["trg_eos_idx"],
        pad_idx=meta["trg_pad_idx"],
        sos_idx=meta["trg_sos_idx"],
    )

    return detokenize(pred_tokens)


def make_error_df(model, model_name, test_df, meta, device, max_len=50, seed=42):
    df = test_df.copy()
    df["src_len"] = df["src_ids"].apply(lambda x: len(x) - 2)
    df["length_bucket"] = df["src_len"].apply(length_bucket)

    short_examples = df[df["length_bucket"] == "short"].sample(5, random_state=seed)
    medium_examples = df[df["length_bucket"] == "medium"].sample(5, random_state=seed)
    long_examples = df[df["length_bucket"] == "long"].sample(5, random_state=seed)

    error_df = pd.concat([short_examples, medium_examples, long_examples]).reset_index(drop=True)

    error_df["prediction"] = error_df["src"].apply(
        lambda x: translate_sentence_rnn(
            model,
            x,
            meta["src_vocab"],
            meta["trg_itos"],
            meta,
            device=device,
            max_len=max_len,
        )
    )

    error_df["word_order_error"] = ""
    error_df["missing_content_words"] = ""
    error_df["incorrect_morphology"] = ""
    error_df["hallucinated_tokens"] = ""
    error_df["notes"] = ""

    error_df = error_df[
        [
            "length_bucket",
            "src_len",
            "src",
            "trg",
            "prediction",
            "word_order_error",
            "missing_content_words",
            "incorrect_morphology",
            "hallucinated_tokens",
            "notes",
        ]
    ].copy()

    return error_df


def load_best_model(model_type, cfg, meta, device):
    if model_type == "seq2seq":
        model = seq2seq.build_model(cfg, meta, device)
        ckpt = "checkpoints/best_seq2seq.pt"
    elif model_type == "attention":
        model = seq2seq_attention.build_model(cfg, meta, device)
        ckpt = "checkpoints/best_attention.pt"
    else:
        raise ValueError("model_type must be 'seq2seq' or 'attention'")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()
    return model


def main():
    cfg = Config()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    
    train_loader, valid_loader, test_loader, meta = load_data(cfg)

    os.makedirs(cfg.results_dir, exist_ok=True)

    
    print("Loading best Seq2Seq model...")
    cfg.model_name = "seq2seq"
    seq_model = load_best_model("seq2seq", cfg, meta, device)

    seq_error_df = make_error_df(
        seq_model,
        "seq2seq",
        meta["test_df"],
        meta,
        device=device,
        max_len=cfg.max_len,
        seed=cfg.seed,
    )

    seq_out = os.path.join(cfg.results_dir, "seq2seq_error_analysis.csv")
    seq_error_df.to_csv(seq_out, index=False)
    print(f"Saved: {seq_out}")
    print(seq_error_df.head())

    del seq_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    
    print("Loading best Seq2Seq + Attention model...")
    cfg.model_name = "attention"
    attn_model = load_best_model("attention", cfg, meta, device)

    attn_error_df = make_error_df(
        attn_model,
        "attention",
        meta["test_df"],
        meta,
        device=device,
        max_len=cfg.max_len,
        seed=cfg.seed,
    )

    attn_out = os.path.join(cfg.results_dir, "seq2seq_attention_error_analysis.csv")
    attn_error_df.to_csv(attn_out, index=False)
    print(f"Saved: {attn_out}")
    print(attn_error_df.head())

    del attn_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Done.")


if __name__ == "__main__":
    main()