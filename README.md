# Ronchi-ML

CNN-based estimator for Ronchi test analysis. Predicts **parabolic correction** (`p_corr`) and **Ronchi offset** from a Ronchi fringe image, conditioned on mirror and grating geometry.

See [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) for the full architecture and flag reference.

---

## Environment

```bash
python3.11 -m venv .venv --copies
source .venv/bin/activate
pip install torch torchvision opencv-python-headless numpy tqdm
```

---

## Dataset Schema

Each line in a JSONL manifest:

```json
{"image": "output/images/ronchi_2.5_-0.050_0.4_680.png",
 "meta":  {"f": 2.5, "offset": -0.05, "lines": 680.0},
 "labels": {"p_corr": 0.4}}
```

| Field | Description |
|---|---|
| `meta.f` | Mirror f/# |
| `meta.offset` | Dimensionless offset: `physical_offset_in / diameter_in` |
| `meta.lines` | Grating lines across mirror diameter: `lpi × diameter_in` |
| `labels.p_corr` | Parabolic correction in [0, 1] |

---

## Train

```bash
python src/train.py --manifest output/manifest.jsonl \
  --resize 320 --epochs 80 --lr 3e-3 --bs 8 \
  --off-weight 0.75 --sched onecycle --augment
```

Outputs: `models/ronchi_auxoffset.pt`, `models/cond_norm.json`, `models/offset_norm.json`

---

## Infer

```bash
python src/infer.py \
  --image real.png \
  --f 3.0 --lpi 100 --diameter 8.0 \
  --ckpt models/ronchi_auxoffset.pt \
  --binarize auto --report-offset
```

`lines` is computed as `lpi × diameter`. Pass `--lines` directly if you already have it.

Example output:

```json
{
  "p_corr": 0.8312,
  "offset_in": -0.1234,
  "offset_dimless": -0.01543,
  "notes": "aux-offset trained; cond=[f, lines=lpi*diameter]; binarize=auto; tta=0; hint_scan=off"
}
```

---

## Layout

```
src/
├─ model.py            custom CNN backbone + two-head net
├─ data.py             JSONL dataset, normalization dataclasses
├─ train.py            training loop
├─ infer.py            single-image inference with TTA and hint scan
├─ preprocess.py       inference-matched preprocessor (grayscale + Otsu)
├─ preprocess_v2.py    signed-distance-map preprocessor
├─ preprocess_60.py    fixed-range threshold binarizer (60% of dynamic range)
├─ preprocess_real.py  robust real-frame pipeline (crop, illumination, Sauvola)
└─ topfix.py           action recommendation from p_corr
docs/
├─ IMPLEMENTATION.md   full architecture, flags, and checkpoint format
└─ EVALUATION.md       evaluation metrics and targets
models/                saved checkpoints and norm stats (git-ignored except .gitkeep)
weights/               ONNX and TorchScript exports
```
