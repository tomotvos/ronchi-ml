from dataclasses import dataclass
import json, cv2 as cv, numpy as np, torch
from torch.utils.data import Dataset

@dataclass
class CondNorm:
    f_mu: float
    f_sigma: float
    lpi_mu: float
    lpi_sigma: float

@dataclass
class OffsetNorm:
    off_mu: float
    off_sigma: float

class RonchiJSONLDataset(Dataset):
    """JSONL format:
    { "image": "...png",
      "meta": {"f":..., "lpi":..., "offset": ...},
      "labels": {"p_corr": ...}
    }
    - p_corr is used in natural units [0..1] (no label normalization).
    - offset is normalized using train-split stats: (offset - mu)/sigma.
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

        # Optional phase jitter (disabled by default, set augment=True to enable)
        if self.augment:
            max_px = max(1, self.resize // 64)  # ~5 px at 320
            d = int(np.random.randint(-max_px, max_px+1))
            if d != 0:
                img = np.roll(img, shift=d, axis=0)  # vertical shift ~ stripe-normal

        x = torch.from_numpy(img[None, ...])  # [1,H,W]

        f   = float(it['meta']['f']); lpi = float(it['meta']['lpi'])
        off = float(it['meta']['offset'])  # inches

        # Normalize cond
        f_n   = (f   - self.C.f_mu)   / (self.C.f_sigma   + 1e-6)
        lpi_n = (lpi - self.C.lpi_mu) / (self.C.lpi_sigma + 1e-6)
        cond = torch.tensor([f_n, lpi_n], dtype=torch.float32)

        # Targets
        p_corr = float(it['labels']['p_corr'])                  # natural [0,1]
        y_p = torch.tensor([p_corr], dtype=torch.float32)
        y_off = torch.tensor([(off - self.O.off_mu)/(self.O.off_sigma + 1e-6)], dtype=torch.float32)

        return x, cond, y_p, y_off
