import math
import torch
import torch.nn.functional as F
import pandas as pd
import sacrebleu
from tqdm import tqdm
from data_utils import detokenize


def ids_to_tokens(ids, itos, eos_idx, pad_idx, sos_idx):
    tokens = []
    for idx in ids:
        if idx == eos_idx:
            break
        if idx in (pad_idx, sos_idx):
            continue
        tokens.append(itos.get(idx, "<unk>"))
    return tokens


@torch.no_grad()
def compute_bleu(model, dataloader, trg_itos, device, eos_idx, pad_idx, sos_idx, max_len=50):
    model.eval()
    predictions, references = [], []

    for batch in tqdm(dataloader, desc="BLEU", leave=False):
        src = batch["src"].to(device, non_blocking=True)
        src_lengths = batch["src_lengths"].to(device, non_blocking=True)
        trg = batch["trg"].to(device, non_blocking=True)

        pred_ids = model.greedy_decode(src, src_lengths, max_len=max_len)

        for i in range(src.size(0)):
            pred_tokens = ids_to_tokens(pred_ids[i].tolist(), trg_itos, eos_idx, pad_idx, sos_idx)
            gold_tokens = ids_to_tokens(trg[i].tolist(), trg_itos, eos_idx, pad_idx, sos_idx)
            predictions.append(detokenize(pred_tokens))
            references.append(detokenize(gold_tokens))

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score, predictions, references


@torch.no_grad()
def beam_search_decode_seq2seq(model, src, src_lengths, beam_width=5, max_len=50):
    hidden = model.encoder(src, src_lengths)
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

            input_token = torch.tensor([last_token], dtype=torch.long, device=src.device)
            output, new_hidden = model.decoder(input_token, hidden_state)

            log_probs = F.log_softmax(output, dim=1)
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_width, dim=1)

            for k in range(beam_width):
                next_token = topk_ids[0, k].item()
                next_score = score + topk_log_probs[0, k].item()
                new_beams.append((seq + [next_token], next_score, new_hidden))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        if all(seq[-1] == model.eos_idx for seq, _, _ in beams):
            break

    return sorted(completed if completed else [(beams[0][0], beams[0][1])], key=lambda x: x[1], reverse=True)[0][0]


@torch.no_grad()
def beam_search_decode_attention(model, src, src_lengths, beam_width=5, max_len=50):
    encoder_outputs, hidden = model.encoder(src, src_lengths)
    seq_len = encoder_outputs.size(1)
    mask = (torch.arange(seq_len, device=src.device).unsqueeze(0) < src_lengths.unsqueeze(1))

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

            input_token = torch.tensor([last_token], dtype=torch.long, device=src.device)
            output, new_hidden, _ = model.decoder(input_token, hidden_state, encoder_outputs, mask)

            log_probs = F.log_softmax(output, dim=1)
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_width, dim=1)

            for k in range(beam_width):
                next_token = topk_ids[0, k].item()
                next_score = score + topk_log_probs[0, k].item()
                new_beams.append((seq + [next_token], next_score, new_hidden))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        if all(seq[-1] == model.eos_idx for seq, _, _ in beams):
            break

    return sorted(completed if completed else [(beams[0][0], beams[0][1])], key=lambda x: x[1], reverse=True)[0][0]


@torch.no_grad()
def beam_search_decode_transformer(model, src, beam_width=5, max_len=50):
    src_key_padding_mask = model.make_src_key_padding_mask(src)
    src_emb = model.pos_encoder(model.src_embedding(src) * math.sqrt(model.emb_dim))
    memory = model.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    beams = [([model.sos_idx], 0.0)]
    completed = []

    for _ in range(max_len):
        candidates = []

        for seq, score in beams:
            if seq[-1] == model.eos_idx:
                completed.append((seq, score))
                candidates.append((seq, score))
                continue

            ys = torch.tensor(seq, dtype=torch.long, device=src.device).unsqueeze(0)
            tgt_mask = model.make_tgt_mask(ys.size(1), src.device)
            tgt_key_padding_mask = model.make_trg_key_padding_mask(ys)
            tgt_emb = model.pos_decoder(model.trg_embedding(ys) * math.sqrt(model.emb_dim))

            out = model.transformer.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            out = model.fc_out(out[:, -1, :])
            log_probs = F.log_softmax(out, dim=-1)
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_width, dim=-1)

            for k in range(beam_width):
                candidates.append((seq + [topk_ids[0, k].item()], score + topk_log_probs[0, k].item()))

        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        if all(seq[-1] == model.eos_idx for seq, _ in beams):
            break

    return sorted(completed if completed else beams, key=lambda x: x[1], reverse=True)[0][0]


@torch.no_grad()
def compute_bleu_beam(model, dataloader, trg_itos, device, eos_idx, pad_idx, sos_idx, model_name, beam_width=5, max_len=50):
    model.eval()
    predictions, references = [], []

    for batch in tqdm(dataloader, desc=f"Beam {beam_width} BLEU", leave=False):
        src = batch["src"].to(device, non_blocking=True)
        src_lengths = batch["src_lengths"].to(device, non_blocking=True)
        trg = batch["trg"].to(device, non_blocking=True)

        for i in range(src.size(0)):
            src_i = src[i].unsqueeze(0)
            len_i = src_lengths[i].unsqueeze(0)

            if model_name == "seq2seq":
                pred_ids = beam_search_decode_seq2seq(model, src_i, len_i, beam_width, max_len)
            elif model_name == "attention":
                pred_ids = beam_search_decode_attention(model, src_i, len_i, beam_width, max_len)
            else:
                pred_ids = beam_search_decode_transformer(model, src_i, beam_width, max_len)

            pred_tokens = ids_to_tokens(pred_ids, trg_itos, eos_idx, pad_idx, sos_idx)
            gold_tokens = ids_to_tokens(trg[i].tolist(), trg_itos, eos_idx, pad_idx, sos_idx)

            predictions.append(detokenize(pred_tokens))
            references.append(detokenize(gold_tokens))

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score, predictions, references


def add_length_buckets(test_df):
    def length_bucket(n):
        if n <= 15:
            return "short"
        elif n <= 35:
            return "medium"
        return "long"

    out = test_df.copy()
    out["src_len"] = out["src_ids"].apply(lambda x: len(x) - 2)
    out["length_bucket"] = out["src_len"].apply(length_bucket)
    return out


def length_analysis(model, test_df, trg_itos, device, collate_fn, eos_idx, pad_idx, sos_idx, model_name, max_len=50):
    from torch.utils.data import DataLoader
    from data_utils import TranslationDataset

    test_df = add_length_buckets(test_df)
    results = []

    for label in ["short", "medium", "long"]:
        subset = test_df[test_df["length_bucket"] == label].reset_index(drop=True)
        loader = DataLoader(TranslationDataset(subset), batch_size=64, shuffle=False, collate_fn=collate_fn)

        greedy_bleu, _, _ = compute_bleu(model, loader, trg_itos, device, eos_idx, pad_idx, sos_idx, max_len)
        beam_bleu, _, _ = compute_bleu_beam(model, loader, trg_itos, device, eos_idx, pad_idx, sos_idx, model_name, 5, max_len)

        results.append({
            "Sentence Length": {
                "short": "Short (<=15)",
                "medium": "Medium (16-35)",
                "long": "Long (36-50)",
            }[label],
            "Greedy BLEU": greedy_bleu,
            "Beam BLEU (beam=5)": beam_bleu,
        })

    return pd.DataFrame(results)