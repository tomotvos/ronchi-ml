
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
            nn.AdaptiveAvgPool2d((4,4))
        )
        self.out_dim = 4*4*4*C

    def forward(self, x):
        return self.features(x).flatten(1)

class Net(nn.Module):
    def __init__(self, cond_dim=2):
        super().__init__()
        self.backbone = Backbone(C=64)
        D = self.backbone.out_dim
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim, 32), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Linear(D + 32, 256), nn.ReLU(inplace=True))
        self.head_p = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        self.head_off = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))

    def forward(self, img, cond):
        h_img = self.backbone(img)
        h_c   = self.cond_proj(cond)
        h     = torch.cat([h_img, h_c], dim=1)
        h     = self.fuse(h)
        p     = self.head_p(h)
        off_n = self.head_off(h)
        return p, off_n
