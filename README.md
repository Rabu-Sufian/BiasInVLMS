# VLMs Are Confidently Biased Before Producing the Final Answer

**Master Thesis: MSc Data Science, IT University of Copenhagen, June 2026**

Ayat Khudoir · Rabia Abu Sufian · Supervisor: Luca Rossi

---

## Overview

This repository contains the full code for our thesis, which investigates prior-driven bias in vision-language models (VLMs) as a probability-level phenomenon. Rather than measuring bias through final-answer accuracy alone, we examine the probability distributions that produce answers during decoding.

We evaluate three open-source VLMs on the [VLMBias benchmark](https://huggingface.co/datasets/anvo25/vlms-are-biased):
- Qwen2.5-VL-3B-Instruct
- Qwen2.5-VL-7B-Instruct
- LLaVA-OneVision-7B

The analysis is organized into two phases:
- **Phase 1**: Sequence-level and token-step probability diagnostics
- **Phase 2**: Visual Contrastive Decoding (VCD) as an inference-time diagnostic probe

---

## Repository Structure

```
BiasInVLMS/
│
├── main_split/                        # Phase 1 inference on counterfactual images
│   ├── qwen_7b_seq.ipynb              # Sequence scoring - Qwen-7B
│   ├── qwen_3b_seq.ipynb              # Sequence scoring - Qwen-3B
│   ├── llava_7b_seq.ipynb             # Sequence scoring - LLaVA-7B
│   ├── qwen_7b_token.ipynb            # Token-step scoring - Qwen-7B
│   ├── qwen_3b_token.ipynb            # Token-step scoring - Qwen-3B
│   ├── llava_7b_token.ipynb           # Token-step scoring - LLaVA-7B
│   ├── sequence_results/              # Sequence scoring parquets (per model)
│   └── token_results/                 # Token-step parquets (per model)
│
├── original_split/                    # Control condition (unmodified images)
│   └── sanity_checks/
│       ├── token_sanity_check.ipynb   # Token-level sanity check + accuracy
│       ├── sequence_sanity_check.ipynb# Sequence-level analysis
│       ├── token_level/               # Input CSVs (model outputs + numeric signals)
│       ├── sequence_level/            # Input CSVs (sequence results + candidates)
│       ├── token_sanity_results/      # Processed outputs from token notebook
│       └── seq_sanity_results/        # Processed outputs from sequence notebook
│
├── phase_2/                           # VCD inference
│   ├── phase2_full_run.py             # VCD pipeline script
│   └── phase_2_results/              # VCD parquets (per model, per distortion)
│
└── analysis/                          # Final analysis and paper figures
    └── analysis.ipynb                 # Main analysis notebook (all figures)
```

---

## How to Run

Run notebooks in this order:

**Step 1: Main split inference**
Run the 6 notebooks in `main_split/` (3 sequence + 3 token), one per model. These produce the parquet files in `sequence_results/` and `token_results/`.

**Step 2: Original split sanity check**
Run `original_split/sanity_checks/token_sanity_check.ipynb` then `sequence_sanity_check.ipynb`. These produce the processed CSVs in `token_sanity_results/` and `seq_sanity_results/`.

**Step 3: Phase 2 VCD**
Run `phase_2/phase2_full_run.py`. This produces the VCD parquets in `phase_2_results/`.

**Step 4: Analysis**
Run `analysis/analysis.ipynb`. This stacks all parquets, loads the original split CSVs, and generates all paper figures and tables into `analysis/outputs/`.

> Before running, update the `BASE` path in the config cell of each notebook to match your local directory.

---

## Requirements

```bash
pip install torch transformers accelerate pillow datasets
pip install pandas numpy matplotlib scipy
pip install pyarrow fastparquet
```

All models are loaded from HuggingFace. You may need to set up a HuggingFace token for model access:

```python
from huggingface_hub import notebook_login
notebook_login()
```

---

## Dataset

We use the [VLMBias benchmark](https://huggingface.co/datasets/anvo25/vlms-are-biased) (`anvo25/vlms-are-biased`), loaded directly via the HuggingFace `datasets` library. Optical Illusions are excluded from all experiments.

---

## Authors

- Ayat Khudoir: aykh@itu.dk
- Rabia Abu Sufian: rabs@itu.dk
- Supervisor: Luca Rossi: lucr@itu.dk

IT University of Copenhagen, Rued Langgaards Vej 7, 2300 Copenhagen, Denmark
