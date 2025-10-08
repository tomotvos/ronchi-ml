
import argparse, json, torch, numpy as np, cv2 as cv
from data import CondNorm, OffsetNorm
from model import Net

def _prep_image(path, resize=320):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(path)
    img = cv.resize(img, (resize, resize), interpolation=cv.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return torch.from_numpy(img[None, None, ...])

def _prep_cond(f, lpi, C: CondNorm):
    f_n   = (f   - C.f_mu)   / (C.f_sigma   + 1e-6)
    lpi_n = (lpi - C.lpi_mu) / (C.lpi_sigma + 1e-6)
    return torch.tensor([[f_n, lpi_n]], dtype=torch.float32)

def infer(image, f, lpi, ckpt, resize=320, report_offset=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(ckpt, map_location=device)
    C = CondNorm(**ck['cond_norm'])
    O = OffsetNorm(**ck['offset_norm'])

    model = Net().to(device)
    model.load_state_dict(ck['model'])
    model.eval()

    x = _prep_image(image, resize).to(device)
    cond = _prep_cond(f, lpi, C).to(device)

    out = {}
    with torch.no_grad():
        p, off_n = model(x, cond)
        p_corr = torch.clamp(p, 0, 1).squeeze().item()
        out['p_corr'] = round(float(p_corr), 4)
        out['notes'] = 'aux-offset trained' if report_offset else 'aux-offset trained, offset ignored at inference'
        if report_offset:
            # de-normalize to inches
            off_in = (off_n.squeeze() * O.off_sigma + O.off_mu).item()
            out['offset_in'] = round(float(off_in), 3)

    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--f', type=float, required=True)
    ap.add_argument('--lpi', type=float, required=True)
    ap.add_argument('--ckpt', default='models/ronchi_auxoffset.pt')
    ap.add_argument('--resize', type=int, default=320)
    ap.add_argument('--report-offset', action='store_true', help='Also output estimated offset (inches).')
    args = ap.parse_args()
    res = infer(args.image, args.f, args.lpi, args.ckpt, resize=args.resize, report_offset=args.report_offset)
    print(json.dumps(res, indent=2))
