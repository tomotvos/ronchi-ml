#!/usr/bin/env python3
import argparse, json, math
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

def detect_pupil(gray):
    h, w = gray.shape
    blur = cv.medianBlur(gray, 5)
    small = cv.resize(blur, (512, int(512*h/w)), interpolation=cv.INTER_AREA) if max(h,w) > 600 else blur
    circles = cv.HoughCircles(small, cv.HOUGH_GRADIENT, dp=1.2, minDist=small.shape[0]//4,
                              param1=100, param2=30,
                              minRadius=int(min(small.shape)/4.5), maxRadius=int(min(small.shape)/1.8))
    if circles is None:
        return None
    c = circles[0][0]
    if small is not blur:
        scale = h / small.shape[0]
        x, y, r = c[0]*scale, c[1]*scale, c[2]*scale
    else:
        x, y, r = c
    return int(round(x)), int(round(y)), int(round(r))

def crop_to_circle(gray, cx, cy, r):
    h, w = gray.shape
    x0, x1 = max(0, cx-r), min(w, cx+r)
    y0, y1 = max(0, cy-r), min(h, cy+r)
    crop = gray[y0:y1, x0:x1]
    mask = np.zeros_like(crop, dtype=np.uint8)
    cv.circle(mask, (min(r, x1-x0-1), min(r, y1-y0-1)), int(0.98*r), 255, -1)
    crop = cv.bitwise_and(crop, crop, mask=mask)
    H, W = crop.shape
    side = max(H, W)
    sq = np.zeros((side, side), dtype=crop.dtype)
    y_off = (side - H)//2
    x_off = (side - W)//2
    sq[y_off:y_off+H, x_off:x_off+W] = crop
    return sq

def illumination_correct(gray, method="division", ksize=151):
    k = int(ksize) | 1
    if method == "division":
        bg = cv.GaussianBlur(gray, (k,k), 0)
        bg = np.clip(bg, 1, 255)
        norm = (gray.astype(np.float32) / bg.astype(np.float32)) * 128.0
        return np.clip(norm, 0, 255).astype(np.uint8)
    elif method == "homomorphic":
        g = gray.astype(np.float32) + 1.0
        log = np.log(g)
        low = cv.GaussianBlur(log, (k,k), 0)
        hi = log - low
        out = np.exp(hi)
        out = out / out.max() * 255.0
        return out.astype(np.uint8)
    else:
        return gray

def sauvola(gray, window=41, k=0.2, R=128):
    gray32 = gray.astype(np.float32)
    win = max(3, int(window) | 1)
    mean = cv.boxFilter(gray32, ddepth=-1, ksize=(win, win), normalize=True)
    mean_sq = cv.boxFilter(gray32*gray32, ddepth=-1, ksize=(win, win), normalize=True)
    var = np.clip(mean_sq - mean*mean, 0, None)
    std = np.sqrt(var)
    thresh = mean * (1 + k * ((std / max(R,1e-5)) - 1))
    bw = (gray32 > thresh).astype(np.uint8) * 255
    return bw

def signed_distance(bw):
    mask = (bw > 127).astype(np.uint8)
    dist_fg = cv.distanceTransform(mask, cv.DIST_L2, 3)
    dist_bg = cv.distanceTransform(1 - mask, cv.DIST_L2, 3)
    return dist_fg - dist_bg

def preprocess_real(
    src, dst, resize=320,
    illum="division", illum_ksize=151,
    denoise="bilateral", denoise_strength=8,
    thresh="sauvola", thr_params="window=41,k=0.18,R=128",
    use_signed=True, despeckle=True, report_json=False
):
    img_raw = cv.imread(src, cv.IMREAD_UNCHANGED)
    if img_raw is None:
        raise FileNotFoundError(src)
    gray0 = _to_gray(img_raw)

    circ = detect_pupil(gray0)
    if circ is not None:
        cx, cy, r = circ
        roi = crop_to_circle(gray0, cx, cy, r)
        crop_used = True
    else:
        h, w = gray0.shape
        side = min(h, w)
        y0 = (h - side)//2
        x0 = (w - side)//2
        roi = gray0[y0:y0+side, x0:x0+side]
        crop_used = False

    roi = illumination_correct(roi, method=illum, ksize=illum_ksize)

    if denoise == "bilateral":
        roi = cv.bilateralFilter(roi, d=7, sigmaColor=denoise_strength*8, sigmaSpace=denoise_strength)
    elif denoise == "gaussian":
        roi = cv.GaussianBlur(roi, (5,5), 0)

    params = {}
    if thr_params:
        for kv in thr_params.split(","):
            k,v = kv.split("=")
            k = k.strip(); v = v.strip()
            if "." in v:
                params[k] = float(v)
            else:
                try:
                    params[k] = int(v)
                except ValueError:
                    params[k] = float(v)
    if thresh == "sauvola":
        bw = sauvola(roi, params.get("window", 41), params.get("k", 0.18), params.get("R", 128))
    elif thresh == "gauss":
        bw = cv.adaptiveThreshold(roi, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv.THRESH_BINARY, max(3, int(params.get("block", 35)) | 1), int(params.get("C", 5)))
    elif thresh == "otsu":
        _, bw = cv.threshold(roi, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    else:
        raise ValueError("Unknown threshold method")

    if despeckle:
        bw = cv.morphologyEx(bw, cv.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)

    if use_signed:
        sdm = signed_distance(bw).astype(np.float32)
        sdm_small = cv.resize(sdm, (resize, resize), interpolation=cv.INTER_LINEAR)
        out = (sdm_small > 0).astype(np.uint8) * 255
    else:
        out_small = cv.resize(bw, (resize, resize), interpolation=cv.INTER_NEAREST)
        out = (out_small > 127).astype(np.uint8) * 255

    ok = cv.imwrite(dst, out)
    if not ok:
        raise RuntimeError(f"Failed to write {dst}")

    if report_json:
        rep = dict(
            input=src, output=dst, resize=int(resize),
            crop_detected=bool(circ is not None), illumination=illum, illum_ksize=int(illum_ksize),
            denoise=denoise, thresh=thresh, thr_params=params, signed=int(use_signed),
            unique=int(len(np.unique(out)))
        )
        print(json.dumps(rep, indent=2))
    else:
        print(f"wrote {dst}  crop={crop_used} illum={illum} denoise={denoise} thr={thresh} signed={use_signed} unique={len(np.unique(out))}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Robust Ronchi preprocessor for real frames (crop, illumination, adaptive threshold, SDM resize).");
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--resize", type=int, default=320)
    ap.add_argument("--illum", choices=["division","homomorphic","none"], default="division")
    ap.add_argument("--illum-ksize", type=int, default=151)
    ap.add_argument("--denoise", choices=["bilateral","gaussian","none"], default="bilateral")
    ap.add_argument("--denoise-strength", type=int, default=8)
    ap.add_argument("--thresh", choices=["sauvola","gauss","otsu"], default="sauvola")
    ap.add_argument("--thr-params", default="window=41,k=0.18,R=128")
    ap.add_argument("--no-signed", dest="use_signed", action="store_false")
    ap.add_argument("--no-despeckle", dest="despeckle", action="store_false")
    ap.add_argument("--report-json", action="store_true")
    args = ap.parse_args()

    preprocess_real(
        args.src, args.dst, resize=args.resize,
        illum=args.illum, illum_ksize=args.illum_ksize,
        denoise=args.denoise, denoise_strength=args.denoise_strength,
        thresh=args.thresh, thr_params=args.thr_params,
        use_signed=args.use_signed, despeckle=args.despeckle,
        report_json=args.report_json
    )
