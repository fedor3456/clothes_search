"""ClothNet: свой ResNet-подобный энкодер, написанный с нуля.

Вход: картинка 3x224x224. Выход: L2-нормированный вектор длины EMB_DIM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False),
                nn.BatchNorm2d(cout),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.skip(x), inplace=True)


class GeM(nn.Module):
    """Generalized Mean Pooling — стандарт в задачах поиска по картинкам."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p).flatten(1)


class ClothNet(nn.Module):
    def __init__(self, emb_dim=128, width=32):
        super().__init__()
        w = width
        self.stem = nn.Sequential(                       # 224 -> 56
            nn.Conv2d(3, w, 3, 2, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.stage1 = nn.Sequential(ResBlock(w, 2 * w, 1), ResBlock(2 * w, 2 * w, 1))          # 56
        self.stage2 = nn.Sequential(ResBlock(2 * w, 4 * w, 2), ResBlock(4 * w, 4 * w, 1))      # 28
        self.stage3 = nn.Sequential(ResBlock(4 * w, 8 * w, 2), ResBlock(8 * w, 8 * w, 1))      # 14
        self.stage4 = nn.Sequential(ResBlock(8 * w, 16 * w, 2), ResBlock(16 * w, 16 * w, 1))   # 7
        self.pool = GeM()
        self.neck = nn.BatchNorm1d(16 * w)
        self.head = nn.Linear(16 * w, emb_dim)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # последний BN в каждом блоке = 0: блок стартует как тождественный, обучение стабильнее
        for m in self.modules():
            if isinstance(m, ResBlock):
                nn.init.zeros_(m.bn2.weight)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage4(self.stage3(self.stage2(self.stage1(x))))
        x = self.neck(self.pool(x))
        return F.normalize(self.head(x), dim=1)


if __name__ == "__main__":
    net = ClothNet()
    n = sum(p.numel() for p in net.parameters())
    print(f"Параметров: {n / 1e6:.2f}M")
    print(net(torch.randn(2, 3, 224, 224)).shape)
