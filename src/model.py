import torch, torch.nn as nn

class Backbone(nn.Module):
    def __init__(self, C=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, C, 5, 2, 2), nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(C, 2*C, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(2*C, 2*C, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(2*C, 4*C, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(4*C, 4*C, 3, 1, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4,4))  # preserve a 4x4 grid of spatial cues
        )
        self.out_dim = 4*4*4*C

    def forward(self, x):
        return self.features(x).flatten(1)  # [B, D]

class Net(nn.Module):
    """Backbone + cond projection -> two heads:
        - head_p: predicts p_corr (raw, unclipped). We'll clamp only for reporting.
        - head_off: predicts normalized offset (zero-mean, unit-std).
    Inference can ignore the offset head entirely.
    """
    def __init__(self, cond_dim=2):
        super().__init__()
        self.backbone = Backbone(C=64)
        D = self.backbone.out_dim
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim, 32), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Linear(D + 32, 256), nn.ReLU(inplace=True))
        self.head_p = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        self.head_off = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))

    def forward(self, img, cond):
        h_img = self.backbone(img)      # [B,D]
        h_c   = self.cond_proj(cond)    # [B,32]
        h     = torch.cat([h_img, h_c], dim=1)
        h     = self.fuse(h)
        p     = self.head_p(h)          # raw scalar
        off_n = self.head_off(h)        # normalized offset
        return p, off_n
