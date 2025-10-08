# Ronchi-ML (p_corr + auxiliary offset)

This pipeline predicts **parabolic correction** (`p_corr`) from a Ronchi image, conditioning on your mirror/grating parameters (`f`, `lpi`).  
During training, it also learns an **auxiliary offset head** (offset in inches, normalized by the train split). At inference, you can return just `p_corr` (default) or include the estimated **offset** with `--report-offset`.

> Invocation style is **path-based** to avoid import fragility:
> ```bash
> python src/train.py ...
> python src/infer.py ...
> ```

---

## Quick start

### 1) Environment
```bash
python3.11 -m venv .venv --copies
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# Core deps
pip install torch torchvision opencv-python numpy
```

### 2) Data format (JSONL)

Each line (example):
```json
{
  "id": "ronchi_3.0_-0.5_0.0",
  "image": "data/images/ronchi_3.0_-0.5_0.0.png",
  "meta":  { "f": 3.0, "offset": -0.5, "lpi": 100.0 },
  "labels": { "p_corr": 0.0 }
}
```

Notes:
- `labels.p_corr` is **natural units** in `[0,1]`.  
- `meta.offset` is in **inches** and is **normalized** internally using the **train** split’s mean/std for the auxiliary head.  
- `f` and `lpi` are **normalized** and used as conditioning inputs.

### 3) Train (single manifest — auto split 80/20)

Recommended settings that converged well:
```bash
python src/train.py --manifest data/manifest.jsonl \
  --resize 320 --epochs 80 --lr 0.003 --bs 8 \
  --off-weight 0.75 --sched onecycle
```

Artifacts produced:
- `models/ronchi_auxoffset.pt` (weights + norms)
- `models/cond_norm.json`, `models/offset_norm.json`

Optional flags:
- `--val-ratio 0.2` (validation fraction for auto split)
- `--split-seed 42` (deterministic shuffle)
- `--augment` (enable light phase-jitter; helpful for robustness once learning is healthy)
- `--wd` (weight decay), `--clip` (grad clip max-norm)

### 3b) Train (explicit train/val files)
```bash
python src/train.py \
  --train-jsonl data/train.jsonl \
  --val-jsonl   data/val.jsonl \
  --resize 320 --epochs 80 --lr 0.003 --bs 8 \
  --off-weight 0.75 --sched onecycle
```

### 4) Inference

**p_corr only (default):**
```bash
python src/infer.py --image data/holdout/your_image.png --f 3.0 --lpi 100 --resize 320
```
Output:
```json
{ "p_corr": 0.7363, "notes": "aux-offset trained, offset ignored at inference" }
```

**Include offset estimate (inches):**
```bash
python src/infer.py --image data/holdout/your_image.png \
  --f 3.0 --lpi 100 --resize 320 --report-offset
```
Output:
```json
{
  "p_corr": 0.7363,
  "notes": "aux-offset trained",
  "offset_in": 0.118
}
```

> `--f` and `--lpi` provide **conditioning** context. If you pass inaccurate values, predictions can shift. If you want to ignore conditioning entirely at inference, we can add a flag to zero it.

---

## Code layout (current)

- `src/data.py`
  - Loads JSONL, reads `image` (grayscale), resizes to `--resize`, scales to `[-1,1]`.
  - Normalizes `f`, `lpi` (conditioning) and **offset** (aux target).
  - Optional light **phase-jitter** augmentation with `--augment`.

- `src/model.py`
  - CNN backbone → fuse image features with projected conditioning → **two heads**:
    - `head_p`: predicts `p_corr` (raw scalar; clamped to `[0,1]` only for reporting).
    - `head_off`: predicts **normalized offset** (denormalized to inches for reporting).

- `src/train.py`
  - Loss: `SmoothL1(p_corr)` **+** `off_weight × SmoothL1(offset_norm)`.
  - Scheduler: **OneCycleLR** by default (`--sched onecycle`), warmup + cosine anneal **per batch**.
  - Prints epoch summary:
    - `train| loss=..., p_loss=..., off_loss=...`
    - `val| p_corr_MAE=...  offset_in_MAE=..."`
  - Saves best checkpoint by **p_corr MAE**.

- `src/infer.py`
  - Loads checkpoint + norms, runs both heads, reports only `p_corr` by default.
  - With `--report-offset`, also returns `offset_in` (inches).

---

## Hyperparameters & tips

- **Image size**: keep `--resize` the same for train/infer (default **320**).
- **Batch size**: small batches (e.g., `8`, `4`) give more updates/epoch and better convergence than `32` on this dataset.
- **LR**: `0.003` with OneCycle worked well; if you change batch size a lot, adjust LR modestly.
- **`--off-weight`**: strength of auxiliary supervision; `0.5–1.0` are good starting points (we used **0.75**).
- **Augmentation**: add `--augment` for robustness once learning trends down; you may see small MAE improvements and better generalization to real captures.
- **Capacity**: if you plateau, bump the backbone width in `model.py` (e.g., `Backbone(C=80)`).

---

## Tiny sanity checks

- **1-sample overfit** (should go ~0 quickly):
  ```bash
  head -n 1 data/manifest.jsonl > data/tiny1.jsonl
  python src/train.py --train-jsonl data/tiny1.jsonl --val-jsonl data/tiny1.jsonl \
    --resize 320 --epochs 150 --lr 0.01 --bs 1 --off-weight 0.5 --sched onecycle
  ```
- **8-sample** (beats mean baseline; commonly ~0.01 MAE on synthetic tiny-8):
  ```bash
  head -n 8 data/manifest.jsonl > data/tiny8.jsonl
  python src/train.py --train-jsonl data/tiny8.jsonl --val-jsonl data/tiny8.jsonl \
    --resize 320 --epochs 120 --lr 0.003 --bs 4 --off-weight 0.5 --sched onecycle
  ```

---

## Repo hygiene

- **Commit code**, not checkpoints. Add `models/` to `.gitignore`:
  ```
  models/
  ```
- Capture exact commands used (training & inference) in your commit message or a short section here for reproducibility.

---

## References

- PyTorch docs: https://pytorch.org/docs/stable/  
- OneCycleLR: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html  
- SmoothL1Loss: https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html  
- OpenCV (Python): https://pypi.org/project/opencv-python/  
- NumPy: https://numpy.org/  
