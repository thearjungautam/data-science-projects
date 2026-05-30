import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from configs import Config
from data_utils import load_data, set_seed, make_collate_fn
from train_utils import fit_model
from eval_utils import compute_bleu, compute_bleu_beam, length_analysis

import seq2seq
import seq2seq_attention
import transformer_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="seq2seq", choices=["seq2seq", "attention", "transformer"])
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(model_name=args.model)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, valid_loader, test_loader, meta = load_data(cfg)

    if cfg.model_name == "seq2seq":
        model = seq2seq.build_model(cfg, meta, device)
        lr = cfg.lr_rnn
    elif cfg.model_name == "attention":
        model = seq2seq_attention.build_model(cfg, meta, device)
        lr = cfg.lr_rnn
    else:
        model = transformer_model.build_model(cfg, meta, device)
        lr = cfg.lr_tf

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=meta["trg_pad_idx"])

    checkpoint_path = os.path.join(cfg.checkpoint_dir, f"best_{cfg.model_name}.pt")

    if args.train:
        history = fit_model(cfg, model, train_loader, valid_loader, optimizer, criterion, device, checkpoint_path)
        pd.DataFrame(history).to_csv(
            os.path.join(cfg.results_dir, f"{cfg.model_name}_history.csv"), index=False
        )

    if args.eval:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        greedy_bleu, _, _ = compute_bleu(
            model,
            test_loader,
            meta["trg_itos"],
            device,
            meta["trg_eos_idx"],
            meta["trg_pad_idx"],
            meta["trg_sos_idx"],
            cfg.max_len,
        )

        beam_results = {}
        for bw in [3, 5, 10]:
            score, _, _ = compute_bleu_beam(
                model,
                test_loader,
                meta["trg_itos"],
                device,
                meta["trg_eos_idx"],
                meta["trg_pad_idx"],
                meta["trg_sos_idx"],
                cfg.model_name,
                beam_width=bw,
                max_len=cfg.max_len,
            )
            beam_results[bw] = score

        collate_fn = make_collate_fn(meta["src_pad_idx"], meta["trg_pad_idx"])
        length_df = length_analysis(
            model,
            meta["test_df"],
            meta["trg_itos"],
            device,
            collate_fn,
            meta["trg_eos_idx"],
            meta["trg_pad_idx"],
            meta["trg_sos_idx"],
            cfg.model_name,
            cfg.max_len,
        )
        length_df.to_csv(os.path.join(cfg.results_dir, f"{cfg.model_name}_length_analysis.csv"), index=False)

        summary = {
            "model": cfg.model_name,
            "greedy_bleu": greedy_bleu,
            "beam_bleu_3": beam_results[3],
            "beam_bleu_5": beam_results[5],
            "beam_bleu_10": beam_results[10],
        }

        with open(os.path.join(cfg.results_dir, f"{cfg.model_name}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(summary)
        print(length_df)


if __name__ == "__main__":
    main()