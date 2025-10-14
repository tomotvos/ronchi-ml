
from dataclasses import dataclass
import json, cv2 as cv, numpy as np, torch
from torch.utils.data import Dataset

@dataclass
class CondNorm:
    f_mu: float
    f_sigma: float
    lines_mu: float
    lines_sigma: float

@dataclass
class OffsetNorm:
    off_mu: float
    off_sigma: float

class RonchiJSONLDataset(Dataset):
    """
    JSONL format (new schema):
    {
      "image":  "path/to/img.png",
      "meta":   {"f": <float>, "offset": <float_dimless>, "lines": <float_num_lines_across_diameter>},
      "labels": {"p_corr": <float in [0,1]>}
    }
    - p_corr remains in natural units [0,1].
    - offset is now dimensionless (normalized to mirror diameter); we still normalize by train-split stats.
    - conditioning uses f and lines (replaces prior lpi).
    """
    def __init__(self, jsonl_path, cond_norm: CondNorm, off_norm: OffsetNorm, resize=320, augment=False):
        self.items = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                self.items.append(json.loads(line))
        self.C = cond_norm
        self.O = off_norm
        self.resize = int(resize)
        self.augment = augment

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = cv.imread(it['image'], cv.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(it['image'])
        img = cv.resize(img, (self.resize, self.resize), interpolation=cv.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5

        # Optional light phase jitter (disabled by default; set augment=True)
        if self.augment:
            max_px = max(1, self.resize // 64)  # ~5 px at 320
            d = int(np.random.randint(-max_px, max_px+1))
            if d != 0:
                img = np.roll(img, shift=d, axis=0)

        x = torch.from_numpy(img[None, ...])  # [1,H,W]

        f     = float(it['meta']['f'])
        lines = float(it['meta']['lines'])  # replaces lpi
        off   = float(it['meta']['offset']) # now dimensionless

        # Normalize conditioning
        f_n     = (f     - self.C.f_mu)     / (self.C.f_sigma     + 1e-6)
        lines_n = (lines - self.C.lines_mu) / (self.C.lines_sigma + 1e-6)
        cond = torch.tensor([f_n, lines_n], dtype=torch.float32)

        # Targets
        p_corr = float(it['labels']['p_corr'])  # natural [0,1]
        y_p = torch.tensor([p_corr], dtype=torch.float32)
        y_off = torch.tensor([(off - self.O.off_mu)/(self.O.off_sigma + 1e-6)], dtype=torch.float32)

        return x, cond, y_p, y_off
