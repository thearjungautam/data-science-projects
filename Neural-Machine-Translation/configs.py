from dataclasses import dataclass


@dataclass
class Config:
    model_name: str = "seq2seq"  # seq2seq, attention, transformer

    # data
    dataset_path: str = "/home/cmm4yj/wmt14_local"
    max_train: int = 300000
    max_valid: int = 3000
    max_test: int = 3000
    max_len: int = 50
    seed: int = 42

    # tokenization
    lowercase: bool = True

    # seq2seq / attention
    emb_dim: int = 512
    hid_dim: int = 512
    num_layers: int = 2
    dropout_rnn: float = 0.3
    teacher_forcing_ratio: float = 0.5
    lr_rnn: float = 1e-3

    # transformer
    tf_emb_dim: int = 512
    tf_heads: int = 8
    tf_enc_layers: int = 4
    tf_dec_layers: int = 4
    tf_ff_dim: int = 2048
    tf_dropout: float = 0.1
    lr_tf: float = 1e-4

    # training
    batch_size: int = 64
    epochs: int = 12
    clip: float = 1.0
    patience: int = 2
    use_amp: bool = True

    # file outputs
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"