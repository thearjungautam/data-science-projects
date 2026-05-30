import os
import time
import torch
from tqdm import tqdm


def epoch_time(start_time, end_time):
    elapsed = end_time - start_time
    return int(elapsed // 60), int(elapsed % 60)


def train_epoch_rnn(model, dataloader, optimizer, criterion, clip, device, teacher_forcing_ratio=0.5):
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
def evaluate_epoch_rnn(model, dataloader, criterion, device):
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


def train_epoch_transformer(model, dataloader, optimizer, criterion, clip, device, use_amp=True):
    model.train()
    epoch_loss = 0.0
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and torch.cuda.is_available()))

    for batch in tqdm(dataloader, desc="Training", leave=False):
        src = batch["src"].to(device, non_blocking=True)
        src_lengths = batch["src_lengths"].to(device, non_blocking=True)
        trg = batch["trg"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(use_amp and torch.cuda.is_available())):
            output = model(src, src_lengths, trg)
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            trg_y = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg_y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


@torch.no_grad()
def evaluate_epoch_transformer(model, dataloader, criterion, device, use_amp=True):
    model.eval()
    epoch_loss = 0.0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        src = batch["src"].to(device, non_blocking=True)
        src_lengths = batch["src_lengths"].to(device, non_blocking=True)
        trg = batch["trg"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(use_amp and torch.cuda.is_available())):
            output = model(src, src_lengths, trg)
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            trg_y = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg_y)

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def fit_model(cfg, model, train_loader, valid_loader, optimizer, criterion, device, checkpoint_path):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    best_valid_loss = float("inf")
    epochs_no_improve = 0
    history = []

    for epoch in range(cfg.epochs):
        start_time = time.time()

        if cfg.model_name == "transformer":
            train_loss = train_epoch_transformer(model, train_loader, optimizer, criterion, cfg.clip, device, cfg.use_amp)
            valid_loss = evaluate_epoch_transformer(model, valid_loader, criterion, device, cfg.use_amp)
        else:
            train_loss = train_epoch_rnn(
                model, train_loader, optimizer, criterion, cfg.clip, device, cfg.teacher_forcing_ratio
            )
            valid_loss = evaluate_epoch_rnn(model, valid_loader, criterion, device)

        mins, secs = epoch_time(start_time, time.time())
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

        if epochs_no_improve >= cfg.patience:
            print("Early stopping triggered.")
            break

    return history