# Implementation Reference

## Model Architecture

`src/model.py` defines a two-head custom CNN (`Net`).

**Backbone** (`Backbone`, C=64):

```
Conv2d(1→64,   k=5, s=2, p=2) + ReLU
Conv2d(64→64,  k=3, s=1, p=1) + ReLU
Conv2d(64→128, k=3, s=2, p=1) + ReLU
Conv2d(128→128,k=3, s=1, p=1) + ReLU
Conv2d(128→256,k=3, s=2, p=1) + ReLU
Conv2d(256→256,k=3, s=1, p=1) + ReLU
AdaptiveAvgPool2d(4×4)
→ flatten → 4096 dims
```

**Conditioning projection**: `[f_norm, lines_norm]` (2D) → Linear(2→32) + ReLU → 32 dims

**Fusion**: concat(4096 + 32) → Linear(4128→256) + ReLU

**Heads**:
- `head_p`: Linear(256→128) + ReLU → Linear(128→1) — parabolic correction (clamped to [0,1] at inference)
- `head_off`: Linear(256→128) + ReLU → Linear(128→1) — offset in normalized units

Input images are single-channel float32, normalized to [-1, 1] (`(pixel/255 - 0.5) / 0.5`).

---

## Normalization

Computed from the **train split only** and stored in three places:

| Stats | Saved to | Checkpoint key |
|---|---|---|
| `f`, `lines` z-score | `models/cond_norm.json` | `ckpt["cond_norm"]` |
| dimensionless `offset` z-score | `models/offset_norm.json` | `ckpt["offset_norm"]` |

`p_corr` is kept in natural units [0, 1]; no normalization applied.

---

## Training (`src/train.py`)

### Loss

```
loss = SmoothL1Loss(p_hat, y_p) + off_weight × SmoothL1Loss(off_hat, y_off_normalized)
```

The best checkpoint is saved on validation `p_corr MAE` improvement.

Validation also reports `offset_dimless_MAE` (offset denormalized back to dimensionless units).

### Flags

| Flag | Default | Description |
|---|---|---|
| `--manifest` | — | Single JSONL; auto-split into train/val (mutually exclusive with `--train-jsonl`) |
| `--train-jsonl` | — | Training JSONL (requires `--val-jsonl`) |
| `--val-jsonl` | — | Validation JSONL |
| `--out-ckpt` | `models/ronchi_auxoffset.pt` | Checkpoint output path |
| `--norm-cond` | `models/cond_norm.json` | Conditioning norm stats output |
| `--norm-offset` | `models/offset_norm.json` | Offset norm stats output |
| `--resize` | `320` | Square image size (must match at inference) |
| `--bs` | `8` | Batch size |
| `--lr` | `3e-3` | Max learning rate (OneCycleLR peak) |
| `--epochs` | `80` | Training epochs |
| `--wd` | `0.0` | Adam weight decay |
| `--clip` | `0.0` | Gradient clip max-norm (0 = disabled) |
| `--sched` | `onecycle` | LR scheduler: `onecycle` or `none` |
| `--pct-start` | `0.3` | OneCycleLR warmup fraction |
| `--div-factor` | `25.0` | `initial_lr = max_lr / div_factor` |
| `--final-div-factor` | `1e3` | `min_lr = initial_lr / final_div_factor` |
| `--off-weight` | `0.75` | Auxiliary offset loss weight |
| `--augment` | false | Phase-jitter augmentation: random vertical roll ±5 px (at resize=320) |
| `--val-ratio` | `0.2` | Validation fraction (with `--manifest`) |
| `--split-seed` | `42` | Shuffle seed (with `--manifest`) |

---

## Inference (`src/infer.py`)

### Usage

```bash
# Basic
python src/infer.py --image real.png \
  --f 3.0 --lpi 100 --diameter 8.0 \
  --ckpt models/ronchi_auxoffset.pt \
  --binarize auto --report-offset

# With TTA
python src/infer.py --image real.png \
  --f 3.0 --lpi 100 --diameter 8.0 \
  --ckpt models/ronchi_auxoffset.pt \
  --binarize auto --report-offset --tta 8

# Hint blend (blend physical measurement with prediction)
python src/infer.py --image real.png \
  --f 3.0 --lpi 100 --diameter 8.0 \
  --ckpt models/ronchi_auxoffset.pt \
  --binarize auto --report-offset \
  --offset-hint-in 0.15 --offset-hint-weight 0.5

# Adaptive hint scan (searches vertical shifts to best match hint phase)
python src/infer.py --image real.png \
  --f 3.0 --lpi 100 --diameter 8.0 \
  --ckpt models/ronchi_auxoffset.pt \
  --binarize auto --report-offset \
  --offset-hint-in 0.15 --hint-scan --tta 4
```

`lines` is computed as `lpi × diameter`. Pass `--lines` directly to skip this.

### Flags

| Flag | Default | Description |
|---|---|---|
| `--image` | required | Input image path |
| `--f` | required | Mirror f/# |
| `--lines` | — | Lines across diameter (or use `--lpi` + `--diameter`) |
| `--lpi` | — | Grating lines per inch |
| `--diameter` | — | Mirror diameter in inches (for computing `lines` and offset conversion) |
| `--ckpt` | `models/ronchi_auxoffset.pt` | Checkpoint path |
| `--resize` | `320` | Must match training resize |
| `--report-offset` | false | Include offset prediction in output |
| `--binarize` | `auto` | `auto` \| `always` \| `off` — Otsu binarize after resize |
| `--tta` | `0` | TTA jitter variants to average (0 = disabled) |
| `--offset-hint-in` | — | Physical offset hint in inches to blend with prediction |
| `--offset-hint-weight` | `0.5` | Blend weight: `final = (1-w)×pred + w×hint` |
| `--hint-scan` | false | Adaptive vertical-roll search around hint (requires `--offset-hint-in` + `--diameter`) |
| `--hint-scan-scale` | `6.0` | `radius_px ≈ |hint_dimless| × resize × scale` |
| `--hint-scan-cap` | — | Cap on scan radius in pixels |

### Output fields

| Field | Condition | Description |
|---|---|---|
| `p_corr` | always | Predicted parabolic correction, clamped [0, 1] |
| `notes` | always | Processing summary string |
| `offset_in` | `--report-offset` + `--diameter` | Predicted offset in inches |
| `offset_dimless` | `--report-offset` | Predicted dimensionless offset |
| `offset_in_pred_raw` | `--report-offset` + hint blend | Raw model prediction before hint blend |
| `hint_weight` | hint blend | Weight applied to hint |
| `warning` | `--report-offset` without `--diameter` | Prompt to provide `--diameter` |
| `hint_scan` | `--hint-scan` | `best_shift_px`, `radius_px`, `hint_dimless`, `best_err_dimless` |

---

## Preprocessing Tools

Four preprocessors handle the simulated-to-real domain gap. All output a square {0, 255} PNG.

### `preprocess.py` — inference-matched

Replicates the exact pipeline used during inference: grayscale → resize → optional Otsu binarize. Use this to inspect or prepare real images to match the training distribution.

```bash
python src/preprocess.py --in real.png --out proc.png \
  --resize 320 --binarize auto
```

| Flag | Default | Description |
|---|---|---|
| `--resize` | `320` | Output square size |
| `--binarize` | `auto` | `auto` \| `always` \| `off` |
| `--force-1bit` | false | Strict {0, 255} clamp after binarize step |
| `--report-json` | false | Print JSON processing stats |

`auto` mode samples ~1% of pixels; if more than 3 unique values are found it applies Otsu.

### `preprocess_v2.py` — signed distance map

Binarizes at full resolution, computes a signed distance map, resizes the SDM with linear interpolation, then re-thresholds at zero. This preserves crisp edges when downscaling from high-resolution captures.

```bash
python src/preprocess_v2.py --in real.png --out proc.png \
  --strategy signed --bin otsu --resize 320
```

| Flag | Default | Description |
|---|---|---|
| `--strategy` | `signed` | `signed` \| `resize_then_thresh` \| `thresh_then_resize` |
| `--bin` | `otsu` | `otsu` \| `gauss` \| `sauvola` |
| `--bin-params` | `""` | e.g. `"block=35,C=5"` or `"window=31,k=0.2,R=128"` |
| `--inter` | `area` | Resize interpolation for non-signed strategies |
| `--no-rethreshold` | — | Skip final hard clamp after resize |
| `--despeckle` | false | NlMeans denoise before threshold + morphological opening after |
| `--report-json` | false | Print JSON processing stats |

### `preprocess_60.py` — fixed-range threshold

Binarizes using a fixed 60%-range threshold (`black + 0.6 × (light - black)`), then resizes with NEAREST interpolation. Useful when Otsu misjudges the black/white balance on low-contrast frames.

```bash
python src/preprocess_60.py --in real.png --out proc.png \
  --resize 320 --robust
```

| Flag | Default | Description |
|---|---|---|
| `--invert` | false | Swap black/white assignment |
| `--robust` | false | Use 1st/99th percentiles instead of min/max for black/light levels |
| `--report-json` | false | Print computed stats (black, light, threshold) |

### `preprocess_real.py` — full real-frame pipeline

Most capable preprocessor. Pipeline: auto-crop to pupil circle via Hough → illumination correction → denoise → adaptive threshold (Sauvola by default) → optional signed-distance resize.

```bash
python src/preprocess_real.py --in real.png --out proc.png \
  --illum division --thresh sauvola --resize 320
```

| Flag | Default | Description |
|---|---|---|
| `--illum` | `division` | `division` \| `homomorphic` \| `none` |
| `--illum-ksize` | `151` | Gaussian kernel size for illumination background |
| `--denoise` | `bilateral` | `bilateral` \| `gaussian` \| `none` |
| `--denoise-strength` | `8` | Bilateral filter sigmaColor multiplier |
| `--thresh` | `sauvola` | `sauvola` \| `gauss` \| `otsu` |
| `--thr-params` | `window=41,k=0.18,R=128` | Threshold parameters (comma-separated `key=val`) |
| `--no-signed` | — | Use NEAREST resize instead of signed distance map |
| `--no-despeckle` | — | Skip morphological opening after threshold |
| `--report-json` | false | Print JSON processing report |

If Hough circle detection fails, the pipeline falls back to a center crop of the shortest image dimension.

---

## Checkpoint Format

```python
{
    "model":       model.state_dict(),
    "cond_norm":   {"f_mu": ..., "f_sigma": ..., "lines_mu": ..., "lines_sigma": ...},
    "offset_norm": {"off_mu": ..., "off_sigma": ...},
}
```

Norm stats are also written as standalone JSON files (`models/cond_norm.json`, `models/offset_norm.json`) for use without loading the full checkpoint tensor.

---

## `topfix.py`

`recommend_fix(p_corr, tol=0.03)` maps a scalar prediction to a next-action suggestion:

| Condition | Action | Rationale |
|---|---|---|
| `|p_corr - 1.0| ≤ tol` | `no_action` | Near ideal parabola |
| `p_corr < 1.0` | `increase_correction` | Undercorrected |
| `p_corr > 1.0` | `reduce_correction` | Overcorrected |
