# Ronchi‑ML Two‑Head POC (Flat `src/` Layout)

This POC predicts both **parabolic correction** (`p_corr`) and **Ronchi offset in inches** (`offset_in`) from a Ronchi image, conditioning on your mirror/grating parameters (`f`, `lpi`). It uses your existing JSONL manifest unchanged.

---

## 🚀 Quick Start

### 1) Environment
```bash
python3.11 -m venv .venv --copies
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install torch torchvision opencv-python numpy tqdm
```

### 2) Train (single manifest — auto split)
If you have **one** combined manifest (e.g., `data/manifest.jsonl`), the trainer will split it internally into train/val (80/20 by default):

```bash
python src/train.py --manifest data/manifest.jsonl --resize 320
```

**Optional split controls:**
- `--val-ratio 0.2`
- `--split-seed 42`

### 2b) Train (explicit two-file mode — optional)
If you prefer to manage the split yourself:
```bash
python src/train.py --train-jsonl data/train.jsonl --val-jsonl data/val.jsonl --resize 320
```

### 3) Inference
```bash
python src/infer.py --image data/images/ronchi_3.0_-0.5_0.0.png --f 3.0 --lpi 100 --resize 320
```

Example output:
```json
{
  "p_corr": 0.9731,
  "offset_in": -0.5123,
  "offset_mm": -13.01,
  "notes": "twohead-inch"
}
```

---

## 📦 Expected JSONL Format

Each line in your manifest looks like:
```json
{"id":"ronchi_3.0_-0.5_0.0",
 "image":"data/images/ronchi_3.0_-0.5_0.0.png",
 "meta":{"f":3.0,"offset":-0.5,"lpi":100.0},
 "labels":{"p_corr":0.0}}
```

- `meta.offset` is in **inches** (negative = inside ROC).
- `labels.p_corr` is your scalar parabolic correction.
- Trainer computes normalization stats from the **train** split only.

---

## 🔧 Flags (train)

| Flag | Default | Meaning |
|------|---------|---------|
| `--manifest` | — | Single JSONL; trainer auto-splits into train/val |
| `--train-jsonl` / `--val-jsonl` | — | Explicit split files (alternative to `--manifest`) |
| `--resize` | `320` | Image size used for both train and infer |
| `--bs` | `32` | Batch size |
| `--lr` | `3e-4` | Learning rate |
| `--epochs` | `20` | Training epochs |
| `--lambda_z` | `0.25` | Offset-loss weight |
| `--val-ratio` | `0.2` | (Only with `--manifest`) validation fraction |
| `--split-seed` | `42` | (Only with `--manifest`) reproducible shuffling |

---

## 🧠 Implementation Notes

- Two-head CNN (`p_corr`, `offset_in`) with conditioning on (`f`, `lpi`).
- Loss: `SmoothL1Loss` for both heads; offset head scaled by `--lambda_z`.
- Norm stats saved to `models/label_norm.json` and `models/cond_norm.json` inside the checkpoint as well.
- **Keep `--resize` the same** for both training and inference (default 320).

---

## 🗂 Layout

```
src/
├─ data.py
├─ model.py
├─ train.py   # supports --manifest auto-split OR explicit --train-jsonl/--val-jsonl
├─ infer.py
└─ README.md
```
