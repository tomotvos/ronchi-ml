#!/usr/bin/env python3
import argparse, cv2 as cv, numpy as np, json

def _to_gray(img):
    if img.ndim == 2:
        if img.dtype != np.uint8:
            img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        return img
    if img.ndim == 3:
        if img.shape[2] == 3:
            return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if img.shape[2] == 4:
            return cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {img.shape!r}")

def preprocess_image(src_path, dst_path, resize=320, binarize_mode="auto", force_1bit=False, report_json=False):
    img_raw = cv.imread(src_path, cv.IMREAD_UNCHANGED)
    if img_raw is None:
        raise FileNotFoundError(src_path)
    gray = _to_gray(img_raw)

    gray = cv.resize(gray, (resize, resize), interpolation=cv.INTER_AREA)

    did_binarize = False
    if binarize_mode not in ("always","auto","off"):
        raise ValueError("--binarize must be one of: auto, always, off")
    if binarize_mode == "always":
        did_binarize = True
    elif binarize_mode == "auto":
        h, w = gray.shape
        step = max(1, (h*w)//100)
        sample = gray.reshape(-1)[::step]
        uniq = np.unique(sample)
        if len(uniq) > 3:
            did_binarize = True

    if did_binarize:
        _, bw = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        gray = bw

    if force_1bit:
        gray = (gray > 127).astype(np.uint8) * 255

    ok = cv.imwrite(dst_path, gray)
    if not ok:
        raise RuntimeError(f"Failed to write {dst_path}")

    if report_json:
        stats = {
            "input": src_path,
            "output": dst_path,
            "shape": [int(gray.shape[0]), int(gray.shape[1])],
            "resize": int(resize),
            "binarize": binarize_mode,
            "did_binarize": bool(did_binarize),
            "force_1bit": bool(force_1bit),
            "unique_values_count": int(len(np.unique(gray))),
        }
        print(json.dumps(stats, indent=2))
    else:
        print(f"wrote {dst_path}  (resize={resize}, binarize={binarize_mode}, did_binarize={did_binarize}, 1bit={force_1bit})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Preprocess a Ronchi image exactly like inference (grayscale+resize+optional Otsu binarize).")
    ap.add_argument("--in", dest="src", required=True, help="Source image path")
    ap.add_argument("--out", dest="dst", required=True, help="Destination PNG path")
    ap.add_argument("--resize", type=int, default=320, help="Output size (square). Default: 320")
    ap.add_argument("--binarize", choices=["auto","always","off"], default="auto", help="Binarization mode (Otsu after resize). Default: auto")
    ap.add_argument("--force-1bit", action="store_true", help="Force strict {0,255} output values after binarization step")
    ap.add_argument("--report-json", action="store_true", help="Print a JSON report with processing details")
    args = ap.parse_args()

    preprocess_image(args.src, args.dst, resize=args.resize, binarize_mode=args.binarize, force_1bit=args.force_1bit, report_json=args.report_json)
