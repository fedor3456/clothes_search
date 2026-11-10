"""Лоссы для metric learning: batch-hard triplet и ArcFace.

Triplet: для каждого якоря берём самый далёкий позитив и самый близкий негатив.
ArcFace: классификация вещей по углу между эмбеддингом и центром класса
с добавочным угловым зазором m. Даёт более компактные классы, чем один triplet.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, emb, labels):
        # emb уже L2-нормированы -> квадрат евклидова расстояния = 2 - 2*cos
        dist = (2 - 2 * emb @ emb.T).clamp(min=1e-12).sqrt()
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)

        pos_mask = same & ~eye
        hardest_pos = (dist * pos_mask).max(dim=1).values
        hardest_neg = dist.masked_fill(same, float("inf")).min(dim=1).values

        loss = torch.relu(hardest_pos - hardest_neg + self.margin)
        valid = pos_mask.any(dim=1)
        # доля троек, где модель ещё ошибается — полезно смотреть в логах
        active = (loss[valid] > 0).float().mean()
        return loss[valid].mean(), active


class ArcFaceLoss(nn.Module):
    """ArcFace: cos(theta + m) для верного класса, масштаб s.

    Хранит центры классов (num_classes x emb_dim). Нужен только при обучении,
    при поиске используется один энкодер.
    """

    def __init__(self, num_classes, emb_dim, s=30.0, m=0.4):
        super().__init__()
        self.num_classes, self.s, self.m = num_classes, s, m
        self.weight = nn.Parameter(torch.empty(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb, labels, m=None):
        m = self.m if m is None else m
        cos = F.linear(emb, F.normalize(self.weight, dim=1)).clamp(-1 + 1e-7, 1 - 1e-7)
        target = torch.cos(torch.acos(cos) + m)
        # если theta + m > pi, cos(theta + m) перестаёт убывать; берём линейное продолжение
        th, mm = math.cos(math.pi - m), math.sin(math.pi - m) * m
        target = torch.where(cos > th, target, cos - mm)
        onehot = F.one_hot(labels, self.num_classes).bool()
        logits = torch.where(onehot, target, cos) * self.s
        loss = F.cross_entropy(logits, labels)
        acc = (cos.argmax(dim=1) == labels).float().mean()
        return loss, acc
