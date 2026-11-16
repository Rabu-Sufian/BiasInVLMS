# Towards Counterfactual Bias Quantification in Vision-Language Models

**Published at the 8th Northern Lights Deep Learning Conference (NLDL 2027), PMLR 352**

Rabia Abu Sufian\* · Ayat Khudoir\* · Stella Graßhof · Luca Rossi
IT University of Copenhagen

\*Equal contribution

This work builds on our MSc Data Science thesis at the IT University of Copenhagen (June 2026).

---

## Overview

This repository contains the code and results for our paper, which studies counterfactual bias in vision-language models (VLMs) at the level of output probabilities rather than final-answer accuracy alone.

We evaluate three open-source VLMs on the [VLMBias benchmark](https://huggingface.co/datasets/anvo25/vlms-are-biased):
- Qwen2.5-VL-3B-Instruct
- Qwen2.5-VL-7B-Instruct
- LLaVA-OneVision-7B

The analysis is organized into two phases:
- **Phase 1**: Sequence-level candidate scoring (with PMI correction) and token-step probability diagnostics
- **Phase 2**: A VCD-style contrast of original and distorted-image candidate scores (Gaussian noise and blur) as a diagnostic probe

---

## Repository Structure

```
BiasInVLMS/
│
├── main_split/                          # Counterfactual images (Main Split)
│   ├── pipelines/                       # Inference notebooks
│   │   ├── qwen_3b_seq.ipynb            # Sequence scoring - Qwen-3B
│   │   ├── qwen_7b_seq.ipynb            # Sequence scoring - Qwen-7B
│   │   ├── llava_7b_seq.ipynb           # Sequence scoring - LLaVA-7B
│   │   ├── qwen_3b_token.ipynb          # Token-step scoring - Qwen-3B
│   │   ├── qwen_7b_token.ipynb          # Token-step scoring - Qwen-7B
│   │   └── llava_7b_token.ipynb         # Token-step scoring - LLaVA-7B
│   ├── sequence_results/                # Sequence scoring parquets (per model, incl. PMI)
│   └── token_results/                   # Token-step parquets (per model)
│
├── original_split/                      # Control condition (unmodified images)
│   ├── pipelines/                       # Inference notebooks (sequence + token, per model)
│   ├── sequence_level/                  # Sequence results and candidate scores (CSV)
│   ├── token_level/                     # Model outputs and numeric signals (CSV)
│   └── sanity_checks/
│       ├── token_sanity_check.ipynb     # Token-level sanity check + accuracy
│       ├── sequence_sanity_check.ipynb  # Sequence-level analysis
│       ├── token_sanity_results/        # Processed outputs and figures
│       └── seq_sanity_results/          # Processed outputs
│
├── phase_2/                             # VCD-style distortion probe
│   ├── phase_2.ipynb
│   ├── phase2_full_run.py               # Full run: 3 models x 2 distortions
│   └── phase_2_results/                 # Parquets (per model, per distortion)
│
└── analysis.ipynb                       # Main analysis notebook (all paper figures and tables)
```

---

## How to Run

Run in this order:

**Step 1: Main Split inference.**
Run the 6 notebooks in `main_split/pipelines/` (3 sequence + 3 token, one per model). These produce the parquet files in `sequence_results/` and `token_results/`.

**Step 2: Original Split control.**
Run the notebooks in `original_split/pipelines/`, then `original_split/sanity_checks/token_sanity_check.ipynb` and `sequence_sanity_check.ipynb`. These produce the processed CSVs in `token_sanity_results/` and `seq_sanity_results/`.

**Step 3: Phase 2 (VCD-style probe).**
Run `phase_2/phase2_full_run.py`. This produces the parquets in `phase_2_results/`.

**Step 4: Analysis.**
Run `analysis.ipynb`. This loads all results and generates the paper figures and tables.

> Before running, update the `BASE` path in the config cell of each notebook to match your local directory.

### Key settings

- Decoding: greedy (sampling disabled), float16, automatic device mapping
- Token-step prompt suffix (all models, both splits): `Answer with exactly ONE digit (0-9). No text, no multiple digits.`, with `max_new_tokens=3`
- PMI null prompt: the literal prefix `Answer:` followed by the candidate (no image, question, or chat template)
- Distortions: Gaussian noise (σ = 100, fixed seed) and Gaussian blur (radius 10); contrast strengths α ∈ {0.5, 1.0, 2.0}

---

## Requirements

```bash
pip install torch transformers accelerate pillow datasets
pip install pandas numpy matplotlib scipy
pip install pyarrow fastparquet
```

All models are loaded from Hugging Face. You may need a Hugging Face token for model access:

```python
from huggingface_hub import notebook_login
notebook_login()
```

---

## Dataset

We use the [VLMBias benchmark](https://huggingface.co/datasets/anvo25/vlms-are-biased) (`anvo25/vlms-are-biased`), loaded via the Hugging Face `datasets` library. Optical Illusions are excluded from all experiments.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{abusufian2027counterfactual,
  title     = {Towards Counterfactual Bias Quantification in Vision-Language Models},
  author    = {Abu Sufian, Rabia and Khudoir, Ayat and Gra{\ss}hof, Stella and Rossi, Luca},
  booktitle = {Proceedings of the 8th Northern Lights Deep Learning Conference},
  series    = {Proceedings of Machine Learning Research},
  volume    = {352},
  publisher = {PMLR},
  year      = {2027}
}
```

---

## Contact

- Rabia Abu Sufian: rabiasufian0@gmail.com
- Ayat Khudoir: aykhudoir@hotmail.com
- Stella Graßhof: stgr@itu.dk
- Luca Rossi: lucr@itu.dk
