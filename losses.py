"""Batch-hard triplet loss.

Для каждого якоря в батче берём:
  - самый далёкий позитив (то же item_id),
  - самый близкий негатив (другое item_id).
Это сильно эффективнее случайных троек.
"""
import torch
import torch.nn as nn


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
