
# ronchi-ml (dimensionless training, real-parameter inference)

Estimate **parabolic correction (`p_corr`)** and **offset** from Ronchi images.

- **Training** uses dimensionless metadata:
  - `meta.f` (f/#)
  - `meta.lines` (grating lines across mirror diameter)
  - `meta.offset` = physical_offset_in / diameter_in
- **Inference** takes real params and converts internally:
  - `lines = lpi * diameter`
  - offset prediction is returned in **inches** when `--report-offset` & `--diameter` are provided.

## Dataset schema (JSONL)
```
{"image":"output/images/ronchi_2.5_-0.050_0.4_680.png",
 "meta":{"f":2.5, "offset":-0.05, "lines":680.0},
 "labels":{"p_corr":0.4}}
```

## Train
```bash
python src/train.py --manifest output/manifest.jsonl \
  --resize 320 --epochs 20 --lr 0.003 --bs 8 \
  --off-weight 0.75 --sched onecycle --augment
```

Outputs:
- `models/ronchi_auxoffset.pt` (checkpoint)
- `models/cond_norm.json`, `models/offset_norm.json`

## Inference
```bash
python src/infer.py --image real.png \
  --f 3.0 --lpi 100 --diameter 8.0 --resize 320 \
  --ckpt models/ronchi_auxoffset.pt \
  --binarize auto --report-offset
```
Options:
- `--binarize {auto,always,off}`: 1‑bit Otsu after resize (default `auto`).
- `--tta N`: average over N jitter variants + base.
- `--offset-hint-in <in>` and `--offset-hint-weight <0..1>`: blend your physical hint with prediction.
- `--hint-scan`: **adaptive** local search around the hint (requires `--offset-hint-in`, `--diameter`).
  - `radius_px ≈ |offset_hint_in/diameter| * resize * --hint-scan-scale` (default 6.0)
  - optional cap: `--hint-scan-cap <px>`

Examples:
```bash
# TTA
python src/infer.py --image real.png --f 3.0 --lpi 100 --diameter 8.0 \
  --resize 320 --ckpt models/ronchi_auxoffset.pt --binarize auto --report-offset --tta 8

# Hint blend
python src/infer.py --image real.png --f 3.0 --lpi 100 --diameter 8.0 \
  --resize 320 --ckpt models/ronchi_auxoffset.pt --binarize auto --report-offset \
  --offset-hint-in 0.15 --offset-hint-weight 0.5

# Hint + adaptive scan
python src/infer.py --image real.png --f 3.0 --lpi 100 --diameter 8.0 \
  --resize 320 --ckpt models/ronchi_auxoffset.pt --binarize auto --report-offset \
  --offset-hint-in 0.15 --hint-scan --tta 4
# tweak radius:
#   --hint-scan-scale 6.0   # default
#   --hint-scan-cap 20
```

## Tips
- Crop to the pupil before resize for best results.
- If lighting is uneven, try `--binarize always` and/or `--tta 8`.
- Use a real holdout to assess generalization.
