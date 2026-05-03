# Preprocessing Pipeline

> This document is a stub — to be filled in.

## Sections to be completed

- Overview of existing preprocessors: src/preprocess.py, src/preprocess_v2.py, src/preprocess_60.py, src/preprocess_real.py
- Canonical pipeline stages: load, grayscale, normalize, illumination correction, threshold/binarize, invert, resize/crop/pad
- Which preprocessor is used during training vs inference
- Thresholding strategies compared: Otsu (auto), fixed-range 60%, Sauvola adaptive, percentile-clipped
- Illumination correction options: division, homomorphic, none
- Signed-distance-map resize path vs nearest-neighbor resize
- Decisions that need human judgment (inversion, threshold strategy)
- How to save intermediate stages for inspection
