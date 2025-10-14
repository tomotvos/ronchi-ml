
import argparse, json, torch, numpy as np, cv2 as cv, random
from dataclasses import dataclass
from model import Net

@dataclass
class CondNorm:
    f_mu: float; f_sigma: float; lines_mu: float; lines_sigma: float

@dataclass
class OffsetNorm:
    off_mu: float; off_sigma: float

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

def _prep_gray(path, resize=320, binarize_mode="auto"):
    img_raw = cv.imread(path, cv.IMREAD_UNCHANGED)
    if img_raw is None:
        raise FileNotFoundError(path)
    gray = _to_gray(img_raw)
    gray = cv.resize(gray, (resize, resize), interpolation=cv.INTER_AREA)

    do_binarize = False
    if binarize_mode == "always":
        do_binarize = True
    elif binarize_mode == "auto":
        h, w = gray.shape
        step = max(1, (h*w)//100)
        sample = gray.reshape(-1)[::step]
        uniq = np.unique(sample)
        if len(uniq) > 3:
            do_binarize = True

    if do_binarize:
        _, bw = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        gray = bw

    return gray

def _to_tensor(gray_u8):
    img = gray_u8.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return torch.from_numpy(img[None, None, ...])

def _prep_cond(f, lines, C: CondNorm):
    f_n     = (f     - C.f_mu)     / (C.f_sigma     + 1e-6)
    lines_n = (lines - C.lines_mu) / (C.lines_sigma + 1e-6)
    return torch.tensor([[f_n, lines_n]], dtype=torch.float32)

def _tta_variants(gray_u8, N):
    outs = []
    for _ in range(N):
        g = gray_u8.astype(np.float32)
        dy = random.choice([-1, 0, 1])
        if dy != 0:
            g = np.roll(g, shift=dy, axis=0)
        alpha = 0.95 + 0.10 * random.random()
        g = g * alpha
        sigma = random.uniform(0.0, 3.0)
        if sigma > 0:
            g = g + np.random.normal(0.0, sigma, size=g.shape).astype(np.float32)
        if random.random() < 0.25:
            g = cv.GaussianBlur(g, (3,3), 0)
        g = np.clip(g, 0, 255).astype(np.uint8)
        outs.append(g)
    return outs

def _predict_batch(model, x_batch, cond_batch):
    with torch.no_grad():
        p, off_n = model(x_batch, cond_batch)
        p = torch.clamp(p, 0, 1).squeeze(1)
        off_n = off_n.squeeze(1)
        return p.mean().item(), off_n.mean().item()

def infer(
    image, f, lines, ckpt, resize=320,
    report_offset=False, diameter=None,
    binarize_mode="auto", tta=0,
    offset_hint_in=None, offset_hint_weight=0.5,
    hint_scan=False, hint_scan_scale=6.0, hint_scan_cap=None
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(ckpt, map_location=device)
    C = CondNorm(**ck['cond_norm'])
    O = OffsetNorm(**ck['offset_norm'])

    model = Net().to(device)
    model.load_state_dict(ck['model'])
    model.eval()

    gray0 = _prep_gray(image, resize, binarize_mode=binarize_mode)
    cond = _prep_cond(f, lines, C).to(device)

    def eval_gray(gray_u8):
        imgs = [gray_u8] + _tta_variants(gray_u8, int(tta)) if tta and tta > 0 else [gray_u8]
        x_batch = torch.cat([_to_tensor(g) for g in imgs], dim=0).to(device)
        cond_batch = cond.repeat(x_batch.size(0), 1)
        return _predict_batch(model, x_batch, cond_batch)

    # Adaptive hint scan (requires hint + diameter)
    scan_info = None
    if hint_scan:
        if offset_hint_in is None or diameter is None:
            raise ValueError("--hint-scan requires both --offset-hint-in and --diameter")
        hint_dimless = float(offset_hint_in) / float(diameter)
        radius_px = int(round(abs(hint_dimless) * float(resize) * float(hint_scan_scale)))
        if radius_px < 1: radius_px = 1
        if hint_scan_cap is not None:
            radius_px = min(radius_px, int(hint_scan_cap))

        best = None
        for dy in range(-radius_px, radius_px + 1):
            rolled = np.roll(gray0, shift=dy, axis=0)
            p_m, off_n_m = eval_gray(rolled)
            pred_dimless = off_n_m * O.off_sigma + O.off_mu
            err = abs(pred_dimless - hint_dimless)
            if best is None or err < best["err"]:
                best = {"dy": dy, "p_mean": p_m, "off_n_mean": off_n_m, "pred_dimless": pred_dimless, "err": err}
        p_mean, off_n_mean = best["p_mean"], best["off_n_mean"]
        scan_info = {"hint_dimless": hint_dimless, "radius_px": radius_px, "best_shift_px": best["dy"], "best_err": best["err"]}
    else:
        p_mean, off_n_mean = eval_gray(gray0)

    out = {
        "p_corr": round(float(p_mean), 4),
        "notes": f"aux-offset trained; cond=[f, lines=lpi*diameter]; binarize={binarize_mode}; tta={int(tta)}; hint_scan={'on' if hint_scan else 'off'}"
    }
    if scan_info is not None:
        out["hint_scan"] = {
            "best_shift_px": int(scan_info["best_shift_px"]),
            "radius_px": int(scan_info["radius_px"]),
            "hint_dimless": round(float(scan_info["hint_dimless"]), 5),
            "best_err_dimless": round(float(scan_info["best_err"]), 6),
        }

    if report_offset:
        off_dimless_pred = off_n_mean * O.off_sigma + O.off_mu
        if offset_hint_in is not None and diameter is not None:
            off_in_pred = off_dimless_pred * diameter
            off_in_final = (1.0 - offset_hint_weight) * off_in_pred + offset_hint_weight * offset_hint_in
            out["offset_in"] = round(float(off_in_final), 5)
            out["offset_in_pred_raw"] = round(float(off_in_pred), 5)
            out["offset_dimless"] = round(float(off_in_final / diameter), 5)
            out["hint_weight"] = float(offset_hint_weight)
        else:
            if diameter is None:
                out["offset_dimless"] = round(float(off_dimless_pred), 5)
                out["warning"] = "Provide --diameter to convert offset to inches."
            else:
                out["offset_in"] = round(float(off_dimless_pred * diameter), 5)
                out["offset_dimless"] = round(float(off_dimless_pred), 5)

    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--f', type=float, required=True)
    ap.add_argument('--lines', type=float, default=None, help='Number of grating lines across mirror diameter (optional if using --lpi and --diameter)')
    ap.add_argument('--lpi', type=float, default=None, help='Grating lines per inch (used with --diameter)')
    ap.add_argument('--diameter', type=float, default=None, help='Mirror diameter in inches (used with --lpi); also used to convert offset to inches')
    ap.add_argument('--ckpt', default='models/ronchi_auxoffset.pt')
    ap.add_argument('--resize', type=int, default=320)
    ap.add_argument('--report-offset', action='store_true', help='Output predicted offset; converts to inches if --diameter provided')
    ap.add_argument('--binarize', choices=['auto','always','off'], default='auto', help='Binarize to 1-bit using Otsu after resize')
    ap.add_argument('--tta', type=int, default=0, help='Number of TTA jitter variants to average (0 disables)')
    ap.add_argument('--offset-hint-in', type=float, default=None, help='Physical offset hint (inches) to blend with prediction')
    ap.add_argument('--offset-hint-weight', type=float, default=0.5, help='Blend weight for offset hint (0..1); final=(1-w)*pred + w*hint')
    ap.add_argument('--hint-scan', action='store_true', help='Search vertical rolls around the hint to best match its phase (requires --offset-hint-in and --diameter)')
    ap.add_argument('--hint-scan-scale', type=float, default=6.0, help='Adaptive scale: radius_px ≈ |hint_dimless| * resize * scale')
    ap.add_argument('--hint-scan-cap', type=int, default=None, help='Optional cap on scan radius in pixels')
    args = ap.parse_args()

    lines = args.lines if args.lines is not None else (args.lpi * args.diameter if (args.lpi is not None and args.diameter is not None) else None)
    if lines is None:
        ap.error('Either provide --lines, or provide BOTH --lpi and --diameter to compute lines = lpi * diameter.')

    res = infer(
        image=args.image, f=args.f, lines=lines, ckpt=args.ckpt, resize=args.resize,
        report_offset=args.report_offset, diameter=args.diameter,
        binarize_mode=args.binarize, tta=args.tta,
        offset_hint_in=args.offset_hint_in, offset_hint_weight=args.offset_hint_weight,
        hint_scan=args.hint_scan, hint_scan_scale=args.hint_scan_scale, hint_scan_cap=args.hint_scan_cap
    )
    print(json.dumps(res, indent=2))
