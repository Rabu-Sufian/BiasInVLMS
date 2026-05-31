#!/usr/bin/env python3
"""
Phase 2 Full Run: Visual Contrastive Decoding
All 3 models x both distortions x full dataset (~1992 samples)
Checkpoints every 100 samples. Safe to restart if interrupted.
"""

import gc
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from datasets import load_dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    LlavaOnevisionForConditionalGeneration,
    AutoProcessor,
)
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
SAVE_DIR = "/home/rabiasufian0/Phase_1"
CHECKPOINT_EVERY = 100
VALID_TOPICS = {"Animals", "Chess Pieces", "Flags", "Game Boards", "Logos", "Patterned Grid"}

MODELS = [
    {"name": "Qwen2.5-VL-3B", "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",               "arch": "qwen"},
    {"name": "Qwen2.5-VL-7B", "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",               "arch": "qwen"},
    {"name": "LLaVA-OneVision-7B", "model_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf", "arch": "llava"},
]

DISTORTIONS = [
    {"name": "noise_s100", "type": "noise", "param": 100},
    {"name": "blur_r10",   "type": "blur",  "param": 10},
]

# ── Distortion functions ──────────────────────────────────────────────────────
def add_gaussian_noise(image: Image.Image, sigma: float = 100.0, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

def add_gaussian_blur(image: Image.Image, radius: int = 10) -> Image.Image:
    return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))

def apply_distortion(image, distortion):
    if distortion["type"] == "noise":
        return add_gaussian_noise(image, sigma=distortion["param"])
    return add_gaussian_blur(image, radius=distortion["param"])

# ── Model management ──────────────────────────────────────────────────────────
model = None
processor = None

def load_model(model_cfg):
    global model, processor
    free_model()
    print(f"\nLoading {model_cfg['name']}...")
    processor = AutoProcessor.from_pretrained(model_cfg["model_id"])
    if model_cfg["arch"] == "qwen":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_cfg["model_id"], torch_dtype=torch.float16, device_map="auto"
        )
    else:
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_cfg["model_id"], torch_dtype=torch.float16, device_map="auto"
        )
    model.eval()
    print(f"{model_cfg['name']} loaded")

def free_model():
    global model, processor
    for obj in [model, processor]:
        try:
            del obj
        except Exception:
            pass
    model = None
    processor = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("GPU memory freed")

# ── Scoring ───────────────────────────────────────────────────────────────────
def score_sequence(image, prompt, answer):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt}
    ]}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + str(answer)

    prompt_inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
    prompt_len = prompt_inputs["input_ids"].shape[1]

    inputs = processor(text=[full_text], images=[image], return_tensors="pt").to("cuda")
    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    with torch.no_grad():
        outputs = model(**inputs)

    log_probs = torch.log_softmax(outputs.logits, dim=-1)
    answer_ids = inputs["input_ids"][0, prompt_len:]

    per_tok_lp, entropies = [], []
    for i, tok_id in enumerate(answer_ids):
        pos = prompt_len - 1 + i
        per_tok_lp.append(log_probs[0, pos, tok_id].item())
        d = log_probs[0, pos]
        entropies.append(-(d.exp() * d).sum().item())

    return {
        "seq_logprob":    sum(per_tok_lp),
        "avg_logprob":    sum(per_tok_lp) / len(per_tok_lp),
        "entropy_step1":  entropies[0],
        "entropy_mean":   sum(entropies) / len(entropies),
        "n_tokens":       len(answer_ids),
    }

def score_vcd(image, noisy_image, prompt, answer):
    real  = score_sequence(image,       prompt, answer)
    noisy = score_sequence(noisy_image, prompt, answer)
    return {
        "real_seq_logprob":    real["seq_logprob"],
        "real_avg_logprob":    real["avg_logprob"],
        "real_entropy_step1":  real["entropy_step1"],
        "real_entropy_mean":   real["entropy_mean"],
        "noisy_seq_logprob":   noisy["seq_logprob"],
        "noisy_avg_logprob":   noisy["avg_logprob"],
        "noisy_entropy_step1": noisy["entropy_step1"],
        "noisy_entropy_mean":  noisy["entropy_mean"],
        "n_tokens":            real["n_tokens"],
    }

def analyze_sample(sample, distortion, seed=0):
    gt       = str(sample["ground_truth"])
    bias     = str(sample["expected_bias"])
    real_img = sample["image"].convert("RGB")
    noisy_img = apply_distortion(real_img, distortion)

    gold   = score_vcd(real_img, noisy_img, sample["prompt"], gt)
    biased = score_vcd(real_img, noisy_img, sample["prompt"], bias)

    return {
        "ID":            sample.get("ID", ""),
        "topic":         sample["topic"],
        "question_type": sample.get("type_of_question", ""),
        "ground_truth":  gt,
        "expected_bias": bias,
        "distortion":    distortion["name"],
        "gold_real_seq_logprob":    gold["real_seq_logprob"],
        "gold_real_avg_logprob":    gold["real_avg_logprob"],
        "gold_noisy_seq_logprob":   gold["noisy_seq_logprob"],
        "gold_noisy_avg_logprob":   gold["noisy_avg_logprob"],
        "gold_real_entropy_step1":  gold["real_entropy_step1"],
        "gold_noisy_entropy_step1": gold["noisy_entropy_step1"],
        "gold_n_tokens":            gold["n_tokens"],
        "bias_real_seq_logprob":    biased["real_seq_logprob"],
        "bias_real_avg_logprob":    biased["real_avg_logprob"],
        "bias_noisy_seq_logprob":   biased["noisy_seq_logprob"],
        "bias_noisy_avg_logprob":   biased["noisy_avg_logprob"],
        "bias_real_entropy_step1":  biased["real_entropy_step1"],
        "bias_noisy_entropy_step1": biased["noisy_entropy_step1"],
        "bias_n_tokens":            biased["n_tokens"],
        "same_length":              gold["n_tokens"] == biased["n_tokens"],
    }

def compute_derived(df, model_name):
    df["real_avg_margin"]  = df["gold_real_avg_logprob"]  - df["bias_real_avg_logprob"]
    df["noisy_avg_margin"] = df["gold_noisy_avg_logprob"] - df["bias_noisy_avg_logprob"]
    df["margin_shift"]     = df["noisy_avg_margin"] - df["real_avg_margin"]
    df["gold_lp_shift"]    = df["gold_noisy_avg_logprob"] - df["gold_real_avg_logprob"]
    df["bias_lp_shift"]    = df["bias_noisy_avg_logprob"] - df["bias_real_avg_logprob"]
    for alpha in [0.5, 1.0, 2.0]:
        gold_c = (1 + alpha) * df["gold_real_avg_logprob"] - alpha * df["gold_noisy_avg_logprob"]
        bias_c = (1 + alpha) * df["bias_real_avg_logprob"] - alpha * df["bias_noisy_avg_logprob"]
        df[f"vcd_margin_a{alpha}"] = gold_c - bias_c
    df["helped_a1"] = df["vcd_margin_a1.0"] > df["real_avg_margin"]
    df["model"] = model_name
    return df

# ── Dataset ───────────────────────────────────────────────────────────────────
print("Loading dataset...")
ds = load_dataset("anvo25/vlms-are-biased", split="main")
samples = [s for s in ds if s.get("topic") in VALID_TOPICS]
print(f"{len(samples)} samples loaded across {len(VALID_TOPICS)} topics")

# ── Main loop ─────────────────────────────────────────────────────────────────
for model_cfg in MODELS:
    load_model(model_cfg)
    model_tag = model_cfg["name"].replace(".", "").replace(" ", "_").replace("-", "_")

    for distortion in DISTORTIONS:
        out_path  = os.path.join(SAVE_DIR, f"vcd_full_{model_tag}_{distortion['name']}.parquet")
        ckpt_path = out_path.replace(".parquet", "_ckpt.parquet")

        # Resume from checkpoint
        records, done_ids = [], set()
        if os.path.exists(ckpt_path):
            ckpt_df  = pd.read_parquet(ckpt_path)
            records  = ckpt_df.to_dict("records")
            done_ids = set(ckpt_df["ID"].tolist())
            print(f"Resuming: {len(done_ids)} already done")

        remaining = [s for s in samples if s.get("ID", "") not in done_ids]
        print(f"{model_cfg['name']} / {distortion['name']}: {len(remaining)} samples to run")

        for i, sample in enumerate(tqdm(remaining, desc=f"{model_cfg['name']} {distortion['name']}")):
            try:
                records.append(analyze_sample(sample, distortion))
            except Exception as e:
                print(f"  Failed {sample.get('ID', i)}: {e}")

            if (i + 1) % CHECKPOINT_EVERY == 0:
                pd.DataFrame(records).to_parquet(ckpt_path)
                print(f"  Checkpoint: {len(records)} records saved")

        # Final save
        df = compute_derived(pd.DataFrame(records), model_cfg["name"])
        df.to_parquet(out_path)
        print(f"Saved {len(df)} rows -> {out_path}")

        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    free_model()

print("\nPhase 2 full run complete.")
