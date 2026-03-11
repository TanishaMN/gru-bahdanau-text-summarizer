# Amazon Review Title Generator
### Seq2Seq with Bidirectional GRU + Bahdanau Attention — Built from Scratch

A complete implementation of abstractive text summarization trained on 
Amazon product reviews. The model reads a review body and automatically 
generates a short title for it.

---

## What This Project Does

Amazon has millions of product reviews, each with a long body and a short 
title. This system reads the body and writes the title automatically — 
this is called abstractive text summarization.

**Example:**
- Input: *"This book is absolutely fantastic. I have read it three times 
already and learn something new every time..."*
- Output: *"this book"*

---

## Project Architecture

The model is a Sequence-to-Sequence (Seq2Seq) architecture with three 
main components:

**Encoder** — A Bidirectional GRU reads the review forwards and backwards 
simultaneously. Produces one context vector per source word (81 vectors 
total) plus one compressed summary of the entire review.

**Attention (Bahdanau)** — Before generating each title word, attention 
scores all 81 source positions and returns a weighted average. This lets 
the decoder focus on relevant parts of the review at each step.

**Decoder** — A GRU generates the title one word at a time using the 
attention context. Stops when it predicts the end-of-sequence token.

---

## Project Structure
```
amazon-review-summarizer/
├── notebooks/              # Step-by-step Jupyter notebooks (Steps 1–8)
├── src/                    # Demo script for live inference
├── checkpoints/            # Trained model weights (epoch 22)
│   ├── best_model_v2/      # PyTorch folder format
├── dataset/                # Train / val / test CSV files
├── vocab/                  # vocabulary.pkl (33,379 tokens)
├── docs/                   # PDF documentation for each step
├── vizs/                   # All visualizations organized by step
└── training_artifacts/     # training_history_v2.npy (loss curves)
```

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Training samples | 28,830 |
| Vocabulary size | 33,379 tokens |
| Model parameters | ~4 million |
| Best validation loss | 6.5778 |
| Best validation PPL | 719 |
| ROUGE-1 | 0.064 (greedy) / 0.059 (beam) |
| ROUGE-2 | 0.005 (greedy) / 0.003 (beam) |
| Best epoch | 22 of 29 |
| Training time | ~29 minutes on NVIDIA T4 GPU |

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the demo
Open `src/demo_checking.ipynb` in Jupyter and run all cells.
Make sure `checkpoints/best_model_v2_fix.pt` and `vocab/vocabulary.pkl` 
are in place before running.

### 3. Explore step by step
Open notebooks in order from `notebooks/step1_vocabulary_builder.ipynb` 
through `notebooks/step8_inference.ipynb`.

---

## The 8 Steps

| Step | Notebook | What It Does |
|------|----------|--------------|
| 1 | step1_vocabulary_builder | Scans all reviews, builds 33,379 word dictionary |
| 2 | step2_dataset_dataloader | Converts words to numbers, pads sequences, creates batches |
| 3 | step3_encoder | Bidirectional GRU encoder |
| 4 | step4_attention_mechanism | Bahdanau attention mechanism |
| 5 | step5_decoder | GRU decoder with attention |
| 6 | step6_seq2seq_architecture | Connects encoder + decoder into one model |
| 7 | *(run on Colab)* | Training loop — GPU required |
| 8 | step8_inference | Inference, ROUGE evaluation, attention heatmaps |

---

## Training Details

The model required two training runs. The first run revealed severe 
overfitting (val loss peaked at epoch 6). Three fixes were applied:

- Reduced HIDDEN_DIM from 256 → 128 (cut parameters from 16M to 4M)
- Reduced NUM_LAYERS from 2 → 1
- Added teacher forcing linear decay (0.70 → 0.30 over 20 epochs)

The fixed v2 run trained for 29 epochs on Google Colab (T4 GPU, ~59 
seconds per epoch). Early stopping triggered at epoch 29, best checkpoint 
saved at epoch 22.

---

## Known Limitations

- **Sentiment polarity** — The model identifies product domain (book, cd, 
dvd) reliably but does not reliably distinguish positive from negative 
reviews. This requires more training data.
- **Vocabulary gaps** — Product names not seen during training appear as 
 in outputs.
- **ROUGE scores** — ROUGE-1 of 0.064 is low compared to state-of-the-art 
(0.35–0.45), but those systems use millions of samples and Transformer 
architectures. For a from-scratch GRU on 28K samples, any non-zero score 
confirms the model has learned meaningful patterns.

---

## Built With

- Python 3.10
- PyTorch 2.0
- Google Colab (T4 GPU for training)
- NumPy, Pandas, Matplotlib, ROUGE-score
