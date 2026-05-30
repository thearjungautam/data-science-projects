# Neural Machine Translation: Seq2Seq, Attention, and Transformer

## Overview
This project explores neural machine translation (NMT) using three model architectures:

- Seq2Seq (GRU)
- Seq2Seq + Attention
- Transformer

We evaluate these models on:
- English to German (WMT14 subset, 300K sample set)
- English to Klingon (custom low-resource dataset, 33K sample set)

The goal is to compare performance across architectures using BLEU scores, sentence-length analysis, and qualitative error analysis.

## Results Summary

### BLEU Scores (English to German)

Model                  | Greedy BLEU | Beam BLEU (5)
-----------------------|-------------|--------------
Seq2Seq (GRU)          | ~8.8        | ~9.0
Seq2Seq + Attention    | ~10.2       | ~9.4
Transformer            | ~9.7        | ~11.4

### BLEU Scores (English to Klingon)

Model                  | Greedy BLEU | Beam BLEU (5)
-----------------------|-------------|--------------
Seq2Seq + Attention    | 17.33       | 18.12

## Model Architectures

### Seq2Seq (Baseline)
- GRU encoder-decoder
- 2 layers
- Hidden size: 512
- Embedding size: 512

### Seq2Seq + Attention
- Same as baseline with attention mechanism
- Improves alignment between source and target sequences

### Transformer
- Multi-head attention
- Better handling of long-range dependencies

## Klingon Experiment

Dataset: MihaiPopa-1/custom-klingon-33k   

Data split:
- Train: 23,818
- Validation: 1,999
- Test: 1,999  

Architecture:
- Seq2Seq + Attention
- 2 GRU layers
- Hidden size: 512
- Dropout: 0.3

Results:
- Greedy BLEU: 17.33
- Beam BLEU (5): 18.12

## Error Analysis

We categorize errors into:
- Word order errors
- Missing content words
- Incorrect morphology
- Hallucinated tokens

Key findings:
- Seq2Seq: high hallucination and missing word errors
- Attention: reduces major errors
- Transformer: most fluent and consistent outputs

## Setup

Clone the repository:
git clone https://github.com/thearjungautam/G16-NeuralMachineTranslation
cd DS6050

Create environment:
conda create -n ds6050 python=3.11
conda activate ds6050

Install dependencies:
pip install -r requirements.txt

If needed:
pip install torch torchvision torchaudio datasets sacrebleu sentencepiece

## Running Experiments

Train and evaluate (German):
python runner.py --model seq2seq --train --eval
python runner.py --model attention --train --eval
python runner.py --model transformer --train --eval

Klingon experiment:
python klingon_attention_runner.py --max_train 30000 --epochs 10

Error analysis:
python error_analysis.py

## HPC Usage (Rivanna)

Request GPU:
salloc -A shakeri_ds6050 -p gpu --gres=gpu:1 -c 4 --mem=16G -t 4:00:00
srun --pty bash

Run:
module load miniforge
conda activate ds6050
python runner.py

## Project Structure

G16-NeuralMachineTranslation/
├── runner.py
├── klingon_attention_runner.py
├── error_analysis.py
├── data_utils.py
├── configs.py
├── models/
│   ├── seq2seq.py
│   ├── seq2seq_attention.py
│   └── transformer.py
├── requirements.txt
├── README.md

## Notes

Results and model checkpoints are not included due to size constraints but can be reproduced using the provided scripts.

## Key Takeaways

- Attention improves Seq2Seq but does not fully resolve translation errors
- Transformer produces the most fluent outputs
- Beam search provides modest improvements
- Models generalize well to low-resource languages like Klingon
