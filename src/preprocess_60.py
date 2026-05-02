#!/usr/bin/env python3
import argparse, json
import cv2 as cv
import numpy as np

def to_gray(img):
    if img.ndim == 2:
        if img.dtype != np.uint8:
            img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {img.shape!r}")

def preprocess_60(src, dst, resize=320, invert=False, robust=False, report_json=False):
    # Steps:
    # - Convert to grayscale.
    # - Find blackest (min) and lightest (max) pixels (or robust percentiles).
    # - thr = black + 0.6*(light - black).
    # - If pixel <= thr, set 1 (255); else 0. If --invert, swap.
    # - Resize after binarization with NEAREST; hard clamp to {0,255}.
    img_raw = cv.imread(src, cv.IMREAD_UNCHANGED)
    if img_raw is None:
        raise FileNotFoundError(src)
    g = to_gray(img_raw)

    if robust:
        black = float(np.percentile(g, 1))
        light = float(np.percentile(g, 99))
    else:
        black = float(g.min())
        light = float(g.max())

    thr = black + 0.6 * (light - black)

    if not invert:
        bw_full = (g <= thr).astype(np.uint8) * 255
    else:
        bw_full = (g > thr).astype(np.uint8) * 255

    if resize is not None and resize > 0:
        bw = cv.resize(bw_full, (int(resize), int(resize)), interpolation=cv.INTER_NEAREST)
        bw = (bw > 127).astype(np.uint8) * 255
    else:
        bw = bw_full

    ok = cv.imwrite(dst, bw)
    if not ok:
        raise RuntimeError(f"Failed to write {dst}")

    if report_json:
        stats = {
            "input": src,
            "output": dst,
            "resize": int(resize),
            "invert": bool(invert),
            "robust": bool(robust),
            "black": round(black, 3),
            "light": round(light, 3),
            "threshold": round(thr, 3),
            "unique_values": int(len(np.unique(bw)))
        }
        print(json.dumps(stats, indent=2))
    else:
        print(f"wrote {dst}  black={black:.1f} light={light:.1f} thr={thr:.1f} invert={invert} robust={robust}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Simple 60% threshold preprocessor (threshold first, then resize NEAREST).")
    ap.add_argument("--in", dest="src", required=True, help="Source image")
    ap.add_argument("--out", dest="dst", required=True, help="Output image (PNG recommended)")
    ap.add_argument("--resize", type=int, default=320, help="Output size (square). Default: 320")
    ap.add_argument("--invert", action="store_true", help="Invert the binary assignment")
    ap.add_argument("--robust", action="store_true", help="Use 1/99 percentiles for black/light instead of min/max")
    ap.add_argument("--report-json", action="store_true", help="Print computed stats (black, light, threshold)")
    args = ap.parse_args()

    preprocess_60(args.src, args.dst, resize=args.resize, invert=args.invert, robust=args.robust, report_json=args.report_json)
