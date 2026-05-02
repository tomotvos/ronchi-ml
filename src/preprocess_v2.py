
#!/usr/bin/env python3
import argparse, json
import numpy as np
import cv2 as cv

def _to_gray(img):
    if img.ndim == 2:
        if img.dtype != np.uint8:
            img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {img.shape!r}")

def otsu(gray):
    _, bw = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return bw

def adaptive_gauss(gray, block=35, C=5):
    return cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv.THRESH_BINARY, max(3, block | 1), C)

def sauvola(gray, window=31, k=0.2, R=128):
    # Simple Sauvola implementation
    # Compute mean and std using box filters
    gray32 = gray.astype(np.float32)
    win = max(3, window | 1)
    mean = cv.boxFilter(gray32, ddepth=-1, ksize=(win, win), normalize=True)
    mean_sq = cv.boxFilter(gray32*gray32, ddepth=-1, ksize=(win, win), normalize=True)
    var = np.clip(mean_sq - mean*mean, 0, None)
    std = np.sqrt(var)
    thresh = mean * (1 + k * ((std / max(R,1e-5)) - 1))
    bw = (gray32 > thresh).astype(np.uint8) * 255
    return bw

def to_binary(gray, method="otsu", **kw):
    if method == "otsu":
        return otsu(gray)
    elif method == "gauss":
        return adaptive_gauss(gray, kw.get("block", 35), kw.get("C", 5))
    elif method == "sauvola":
        return sauvola(gray, kw.get("window", 31), kw.get("k", 0.2), kw.get("R", 128))
    else:
        raise ValueError("Unknown binarize method")

def signed_distance(bw):
    # bw expected {0,255}
    mask = (bw > 127).astype(np.uint8)
    # Distance to background for foreground pixels
    dist_fg = cv.distanceTransform(mask, cv.DIST_L2, 3)
    # Distance to foreground for background pixels
    dist_bg = cv.distanceTransform(1 - mask, cv.DIST_L2, 3)
    sdm = dist_fg - dist_bg  # positive inside white regions
    return sdm

def preprocess(
    src, dst, resize=320,
    strategy="signed",           # 'signed' | 'resize_then_thresh' | 'thresh_then_resize'
    bin_method="otsu",           # 'otsu' | 'gauss' | 'sauvola'
    bin_params="",               # JSON-like "key=val,key=val" parsed cheaply
    inter="area",                # 'area' | 'nearest' | 'linear'
    rethreshold=True,            # for non-signed strategies, clamp to {0,255} after resize
    despeckle=False,             # optional small opening to reduce hot pixels
    save_json=False
):
    img_raw = cv.imread(src, cv.IMREAD_UNCHANGED)
    if img_raw is None:
        raise FileNotFoundError(src)
    gray0 = _to_gray(img_raw)

    # Parse params like "block=35,C=5" or "window=31,k=0.2,R=128"
    params = {}
    if bin_params:
        for kv in bin_params.split(","):
            if not kv.strip():
                continue
            k, v = kv.split("=")
            k = k.strip()
            v = v.strip()
            try:
                if "." in v:
                    params[k] = float(v)
                else:
                    params[k] = int(v)
            except ValueError:
                try:
                    params[k] = float(v)
                except ValueError:
                    params[k] = v

    if despeckle:
        gray0 = cv.fastNlMeansDenoising(gray0, None, 7, 7, 21)

    if inter == "area":
        interp = cv.INTER_AREA
    elif inter == "nearest":
        interp = cv.INTER_NEAREST
    elif inter == "linear":
        interp = cv.INTER_LINEAR
    else:
        raise ValueError("--inter must be one of: area, nearest, linear")

    # Strategy 1: recommended 'signed' — preserves crisp edges on downscale
    if strategy == "signed":
        bw = to_binary(gray0, bin_method, **params)
        if despeckle:
            # remove tiny specks
            bw = cv.morphologyEx(bw, cv.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        sdm = signed_distance(bw).astype(np.float32)
        sdm_resized = cv.resize(sdm, (resize, resize), interpolation=cv.INTER_LINEAR)
        out = (sdm_resized > 0).astype(np.uint8) * 255

    # Strategy 2: resize then threshold (baseline)
    elif strategy == "resize_then_thresh":
        gray = cv.resize(gray0, (resize, resize), interpolation=interp)
        out = to_binary(gray, bin_method, **params)

    # Strategy 3: threshold then resize, then rethreshold
    elif strategy == "thresh_then_resize":
        bw = to_binary(gray0, bin_method, **params)
        if despeckle:
            bw = cv.morphologyEx(bw, cv.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        # nearest retains edges, area may blur; allow choice
        bw_small = cv.resize(bw, (resize, resize), interpolation=interp)
        out = (bw_small > 127).astype(np.uint8) * 255 if rethreshold else bw_small
    else:
        raise ValueError("Unknown strategy")

    ok = cv.imwrite(dst, out)
    if not ok:
        raise RuntimeError(f"Failed to write {dst}")

    if save_json:
        info = {
            "input": src,
            "output": dst,
            "resize": int(resize),
            "strategy": strategy,
            "binarize": bin_method,
            "params": params,
            "interpolation": inter,
            "unique_values": int(len(np.unique(out))),
        }
        print(json.dumps(info, indent=2))
    else:
        print(f"wrote {dst}  strategy={strategy} bin={bin_method} inter={inter} unique={len(np.unique(out))}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Loss-minimized Ronchi preprocessor (binary-preserving downscale).")
    ap.add_argument("--in", dest="src", required=True, help="Source image")
    ap.add_argument("--out", dest="dst", required=True, help="Output image (PNG recommended)")
    ap.add_argument("--resize", type=int, default=320, help="Output resolution (square). Default: 320")
    ap.add_argument("--strategy", choices=["signed","resize_then_thresh","thresh_then_resize"], default="signed", help="Preprocess strategy")
    ap.add_argument("--bin", dest="bin_method", choices=["otsu","gauss","sauvola"], default="otsu", help="Binarization method")
    ap.add_argument("--bin-params", default="", help="Extra params, e.g. 'block=35,C=5' or 'window=31,k=0.2,R=128'")
    ap.add_argument("--inter", choices=["area","nearest","linear"], default="area", help="Resize interpolation (for non-signed)")
    ap.add_argument("--no-rethreshold", dest="rethreshold", action="store_false", help="Skip final hard clamp after resize (non-signed strategies)")
    ap.add_argument("--despeckle", action="store_true", help="Denoise before threshold (fastNlMeans) and small opening after binarization")
    ap.add_argument("--report-json", action="store_true", help="Print JSON report of settings")
    args = ap.parse_args()

    preprocess(
        args.src, args.dst, resize=args.resize,
        strategy=args.strategy, bin_method=args.bin_method, bin_params=args.bin_params,
        inter=args.inter, rethreshold=args.rethreshold, despeckle=args.despeckle,
        save_json=args.report_json
    )
