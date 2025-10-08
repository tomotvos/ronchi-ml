
import argparse, json, numpy as np, torch, torch.nn as nn, tempfile, os, random, math
from torch.utils.data import DataLoader
from data import RonchiJSONLDataset, CondNorm, OffsetNorm
from model import Net

def _ensure_parent(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _scan_stats(jsonl_path):
    P, F, L, OFF = [], [], [], []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            o = json.loads(line)
            P.append(float(o['labels']['p_corr']))
            F.append(float(o['meta']['f']))
            L.append(float(o['meta']['lpi']))
            OFF.append(float(o['meta']['offset']))
    def stats(a):
        return float(np.mean(a)), float(np.std(a) + 1e-6)
    (f_mu, f_sd)   = stats(F)
    (l_mu, l_sd)   = stats(L)
    (off_mu, off_sd) = stats(OFF)
    Cnorm = CondNorm(f_mu=f_mu, f_sigma=f_sd, lpi_mu=l_mu, lpi_sigma=l_sd)
    Onorm = OffsetNorm(off_mu=off_mu, off_sigma=off_sd)
    return Cnorm, Onorm

def _split_manifest_inline(manifest_path, val_ratio=0.2, seed=42):
    with open(manifest_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    rng = random.Random(seed)
    rng.shuffle(lines)
    cut = int((1.0 - val_ratio) * len(lines))
    train_lines, val_lines = lines[:cut], lines[cut:]
    tf_tr = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, prefix='_tmp_train_', dir='.')
    tf_va = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, prefix='_tmp_val_', dir='.')
    tf_tr.write("\n".join(train_lines) + "\n"); tf_tr.flush(); tf_tr.close()
    tf_va.write("\n".join(val_lines) + "\n"); tf_va.flush(); tf_va.close()
    return tf_tr.name, tf_va.name

def _make_scheduler(opt, dl_tr_len, args):
    if args.sched == "onecycle":
        from torch.optim.lr_scheduler import OneCycleLR
        steps_per_epoch = max(1, dl_tr_len)
        sched = OneCycleLR(
            opt,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=args.pct_start,
            div_factor=args.div_factor,
            final_div_factor=args.final_div_factor,
            three_phase=False,
            anneal_strategy="cos"
        )
        return sched, "batch"
    else:
        return None, None

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cleanup_paths = []
    if args.manifest:
        tr_path, va_path = _split_manifest_inline(args.manifest, args.val_ratio, args.split_seed)
        cleanup_paths.extend([tr_path, va_path])
    else:
        tr_path, va_path = args.train_jsonl, args.val_jsonl

    Cnorm, Onorm = _scan_stats(tr_path)

    # Ensure output dirs exist
    _ensure_parent(args.norm_cond)
    _ensure_parent(args.norm_offset)
    _ensure_parent(args.out_ckpt)

    with open(args.norm_cond, 'w') as f: json.dump(Cnorm.__dict__, f, indent=2)
    with open(args.norm_offset, 'w') as f: json.dump(Onorm.__dict__, f, indent=2)

    ds_tr = RonchiJSONLDataset(tr_path, Cnorm, Onorm, resize=args.resize, augment=args.augment)
    ds_va = RonchiJSONLDataset(va_path, Cnorm, Onorm, resize=args.resize, augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=0, pin_memory=False)
    dl_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=False)

    model = Net().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched, sched_mode = _make_scheduler(opt, len(dl_tr), args)
    l1 = nn.SmoothL1Loss()

    best = math.inf
    try:
        for ep in range(args.epochs):
            model.train()
            running = {"loss":0.0, "p":0.0, "off":0.0, "n":0}
            for img, cond, y_p, y_off in dl_tr:
                img, cond, y_p, y_off = img.to(device), cond.to(device), y_p.to(device), y_off.to(device)
                p_hat, off_hat = model(img, cond)
                loss_p   = l1(p_hat, y_p)                 # p_corr in natural units
                loss_off = l1(off_hat, y_off)             # offset in normalized units
                loss = loss_p + args.off_weight * loss_off

                opt.zero_grad()
                loss.backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
                opt.step()
                if sched and sched_mode == "batch":
                    sched.step()

                # accum
                bs = img.size(0)
                running["loss"] += loss.item() * bs
                running["p"]    += loss_p.item() * bs
                running["off"]  += loss_off.item() * bs
                running["n"]    += bs

            # Epoch summary
            denom = max(running["n"],1)
            avg_loss = running["loss"]/denom
            avg_p    = running["p"]/denom
            avg_off  = running["off"]/denom

            # Validation (report p_corr MAE in natural units, clamped to [0,1], and offset MAE in inches)
            model.eval()
            n, p_mae, off_mae_in = 0, 0.0, 0.0
            with torch.no_grad():
                for img, cond, y_p, y_off in dl_va:
                    img, cond, y_p, y_off = img.to(device), cond.to(device), y_p.to(device), y_off.to(device)
                    p_hat, off_hat = model(img, cond)
                    # p_corr MAE (clamped for readability)
                    p_mae += torch.abs(torch.clamp(p_hat, 0, 1) - y_p).sum().item()
                    # offset MAE in inches (denormalize)
                    off_pred_in = off_hat * Onorm.off_sigma + Onorm.off_mu
                    off_true_in = y_off * Onorm.off_sigma + Onorm.off_mu
                    off_mae_in += torch.abs(off_pred_in - off_true_in).sum().item()
                    n += img.size(0)
            p_mae /= max(n,1)
            off_mae_in /= max(n,1)

            print(f"Epoch {ep+1}/{args.epochs}"
                  f"  train| loss={avg_loss:.4f} p_loss={avg_p:.4f} off_loss={avg_off:.4f}"
                  f"  val| p_corr_MAE={p_mae:.4f}  offset_in_MAE={off_mae_in:.3f}\""
                  f"  lr={opt.param_groups[0]['lr']:.2e}")

            if p_mae < best:
                best = p_mae
                torch.save({
                    "model": model.state_dict(),
                    "cond_norm": Cnorm.__dict__,
                    "offset_norm": Onorm.__dict__
                }, args.out_ckpt)
                print(f"✔ saved {args.out_ckpt}")
    finally:
        for p in cleanup_paths:
            try: os.remove(p)
            except Exception: pass

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--manifest', help='Single JSONL file; will be split into train/val internally.')
    mode.add_argument('--train-jsonl', help='Training JSONL (if not using --manifest).')
    ap.add_argument('--val-jsonl', help='Validation JSONL (required if using --train-jsonl).')

    ap.add_argument('--out-ckpt',       default='models/ronchi_auxoffset.pt')
    ap.add_argument('--norm-cond',      default='models/cond_norm.json')
    ap.add_argument('--norm-offset',    default='models/offset_norm.json')

    ap.add_argument('--resize', type=int, default=320)
    ap.add_argument('--bs',     type=int, default=8)
    ap.add_argument('--lr',     type=float, default=3e-3)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--wd',     type=float, default=0.0, help='Weight decay (Adam)')
    ap.add_argument('--clip',   type=float, default=0.0, help='Gradient clip max-norm (0 disables)')

    ap.add_argument('--sched',  choices=['onecycle','none'], default='onecycle')
    ap.add_argument('--pct-start', type=float, default=0.3, help='OneCycleLR warmup fraction')
    ap.add_argument('--div-factor', type=float, default=25.0, help='OneCycleLR initial_lr = max_lr/div_factor')
    ap.add_argument('--final-div-factor', type=float, default=1e3, help='OneCycleLR min_lr = initial_lr/final_div_factor')

    ap.add_argument('--off-weight', type=float, default=0.75, help='Aux offset loss weight')
    ap.add_argument('--augment', action='store_true', help='Enable phase-jitter augmentation during training.')

    ap.add_argument('--val-ratio', type=float, default=0.2, help='Validation fraction (default 0.2).')
    ap.add_argument('--split-seed', type=int, default=42, help='Shuffle seed for reproducibility.')

    args = ap.parse_args()
    if args.train_jsonl and not args.val_jsonl:
        ap.error('--val-jsonl is required when --train-jsonl is used.')

    train(args)
