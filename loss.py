import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_cross_entropy(p, q):
    q = F.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return - loss.mean()


def stablize_logits(logits):
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits

class MultiPosConLossMM(nn.Module):
    """Multi-positive contrastive loss, single GPU version"""

    def __init__(self, temperature=0.1, w1=1.0, w2=1.0):
        """
        Args:
            temperature: temperature for image-image contrastive loss
            w1: weight for the image contrastive part
            w2: weight for the image-text contrastive part
        """
        super(MultiPosConLossMM, self).__init__()
        self.temperature = temperature
        self.w1 = w1
        self.w2 = w2

    def forward(self, outputs):
        v_feats = outputs['image_emb']         # [B, D]
        t_feats = outputs['text_emb']          # [2, D]
        feats = outputs['image_feats']         # [B, D]
        v_labels = outputs['image_labels']     # [B]
        t_labels = outputs['text_labels']      # [2]
        logit_scale = outputs['logit_scale']   # scalar tensor

        device = v_feats.device

        # Normalize features
        v_feats = F.normalize(v_feats, dim=-1, p=2)
        t_feats = F.normalize(t_feats, dim=-1, p=2)
        feats = F.normalize(feats, dim=-1, p=2)

        # Logits
        logits_v = logit_scale * torch.matmul(v_feats, t_feats.T)  # [B, 2]
        logits_t = logit_scale * torch.matmul(t_feats, v_feats.T)  # [2, B]

        # Image-only contrastive logits
        logits = torch.matmul(feats, feats.T) / self.temperature   # [B, B]

        # ======== 构建 label mask ========
        # image-image mask
        mask = (v_labels.unsqueeze(1) == v_labels.unsqueeze(0)).float().to(device)
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=device)
        mask = mask * logits_mask  # 去除自身对比

        # image-text masks
        v_label_matrix = (v_labels.unsqueeze(1) == t_labels.unsqueeze(0)).float().to(device)  # [B, 2]
        t_label_matrix = (t_labels.unsqueeze(1) == v_labels.unsqueeze(0)).float().to(device)  # [2, B]

        # ======== 稳定 logits ========
        logits = logits - (1 - logits_mask) * 1e9
        logits = stablize_logits(logits)

        # 归一化标签概率分布
        p_img = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
        p_v = v_label_matrix / v_label_matrix.sum(1, keepdim=True).clamp(min=1.0)
        p_t = t_label_matrix / t_label_matrix.sum(1, keepdim=True).clamp(min=1.0)

        # ======== 损失计算 ========
        img_loss = compute_cross_entropy(p_img, logits)
        img_txt_loss = (compute_cross_entropy(p_v, logits_v) + compute_cross_entropy(p_t, logits_t)) / 2

        total_loss = self.w1 * img_loss + self.w2 * img_txt_loss

        return {
            'loss': total_loss,
            'image_loss': img_loss,
            'img_txt_loss': img_txt_loss
        }
