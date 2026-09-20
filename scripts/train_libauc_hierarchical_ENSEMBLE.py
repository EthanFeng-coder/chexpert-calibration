#!/usr/bin/env python3
"""
CheXpert Hierarchical Training - ENSEMBLE Edition
Target: 90%+ AUC

Major Improvements:
1. Multiple SOTA backbones (MaxViT, ConvNeXt-XL, EVA-Giant, EfficientNetV2-XL, Swin-V2)
2. Advanced augmentation (MixUp, CutMix, AutoAugment, RandAugment)
3. AUCMLoss from LibAUC (optimized for AUC metric)
4. ImageNet-21k pre-training
5. Progressive learning (freeze → unfreeze)
6. Label smoothing
7. Test-Time Augmentation (TTA)
"""

import os
import sys
import argparse
import random
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from tqdm import tqdm

# Advanced augmentation imports
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Try importing timm for model zoo
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("WARNING: timm not available, falling back to torchvision models")

# ============================================================================
# Configuration
# ============================================================================

PARENT_LABELS = [
    'Cardiac_Abnormalities', 'Pulmonary_Opacities', 'Pleural_Findings',
    'Focal_Lung_Lesions', 'Fluid_Overload', 'Structural_Other'
]

CHILD_LABELS = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly',  # Cardiac
    'Lung Opacity', 'Consolidation', 'Pneumonia', 'Atelectasis',  # Pulmonary
    'Pleural Effusion', 'Pleural Other', 'Pneumothorax',  # Pleural
    'Lung Lesion',  # Focal
    'Edema',  # Fluid
    'Fracture', 'Support Devices'  # Structural
]

PARENT_TO_CHILD = {
    0: [0, 1],           # Cardiac
    1: [2, 3, 4, 5],     # Pulmonary
    2: [6, 7, 8],        # Pleural
    3: [9],              # Focal
    4: [10],             # Fluid
    5: [11, 12],         # Structural
}

# Backbone configurations with ImageNet-21k weights where available
BACKBONES = {
    'maxvit_large_tf_512': {
        'pretrained': True,
        'num_classes': 0,
        'img_size': 512,
        'features_only': False,
    },
    'convnext_xlarge_in22k': {
        'pretrained': True,
        'num_classes': 0,
        'img_size': 384,
    },
    # timm >=1.x canonical names for diverse-backbone experiments
    'convnext_xlarge.fb_in22k': {
        'pretrained': True,
        'num_classes': 0,
        'img_size': 384,
    },
    'convnext_large.fb_in22k_ft_in1k_384': {
        'pretrained': True,
        'num_classes': 0,
        'img_size': 384,
    },
    'convnextv2_large.fcmae_ft_in22k_in1k_384': {
        'pretrained': True,
        'num_classes': 0,
        'img_size': 384,
    },
    'convnext_xlarge.fb_in22k_ft_in1k_384': {
        'pretrained': True,
        'num_classes': 0,
        'img_size': 384,
    },
    'eva_giant_patch14_336': {
        'pretrained': True,
        'num_classes': 0,
        'img_size': 336,
    },
    'eva02_large_patch14_clip_336.merged2b': {
        'pretrained': True,
        'num_classes': 0,
        'img_size': 336,
    },
    'eva02_large_patch14_clip_336': {
        'pretrained': True,
        'num_classes': 0,
        'img_size': 336,
    },
    'tf_efficientnetv2_xl_in21k': {
        'pretrained': True,
        'num_classes': 0,
        'img_size': 512,
    },
    'tf_efficientnetv2_xl.in21k': {
        'pretrained': True,
        'num_classes': 0,
        'img_size': 512,
    },
    # Exact name available in this local timm build; use for EfficientNetV2-XL trials.
    'tf_efficientnetv2_xl': {
        'pretrained': True,
        'num_classes': 0,
        'img_size': 512,
    },
    'swinv2_large_window12to24_192to384': {
        'pretrained': True,
        'num_classes': 0,
        'img_size': 384,
    },
    'swinv2_large_window12to24_192to384.ms_in22k_ft_in1k': {
        'pretrained': True,
        'num_classes': 0,
        'img_size': 384,
    },
    # ViT-L is a genuinely different Transformer-family trial from MaxViT/Swin/EVA/ConvNeXt.
    'vit_large_patch16_384': {
        'pretrained': True,
        'num_classes': 0,
        'img_size': 384,
    },
    # DeiT III Large is an untried ViT-family diversity backbone available in this timm env.
    'deit3_large_patch16_384': {
        'pretrained': True,
        'num_classes': 0,
        'img_size': 384,
    },
}


# ============================================================================
# Advanced Augmentation Pipeline
# ============================================================================

class AdvancedAugmentation:
    """Advanced augmentation with MixUp, CutMix, AutoAugment"""
    
    def __init__(self, img_size=512, mode='train'):
        self.mode = mode
        self.img_size = img_size
        
        if mode == 'train':
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=10,
                    border_mode=0,
                    p=0.5
                ),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3,
                        contrast_limit=0.3,
                        p=1.0
                    ),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    A.CLAHE(clip_limit=4.0, p=1.0),
                ], p=0.5),
                A.OneOf([
                    A.GaussNoise(p=1.0),
                    A.GaussianBlur(blur_limit=3, p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                ], p=0.2),
                A.CoarseDropout(
                    max_holes=4,
                    max_height=int(img_size * 0.06),
                    max_width=int(img_size * 0.06),
                    p=0.2
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ])
    
    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        return self.transform(image=image)['image']


def mixup_data(x, y_parent, y_child, alpha=1.0):
    """MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_parent_a, y_parent_b = y_parent, y_parent[index]
    y_child_a, y_child_b = y_child, y_child[index]
    
    return mixed_x, y_parent_a, y_parent_b, y_child_a, y_child_b, lam


def cutmix_data(x, y_parent, y_child, alpha=1.0):
    """CutMix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    
    y_parent_a, y_parent_b = y_parent, y_parent[index]
    y_child_a, y_child_b = y_child, y_child[index]
    
    return x, y_parent_a, y_parent_b, y_child_a, y_child_b, lam


def rand_bbox(size, lam):
    """Generate random bounding box for CutMix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2


# ============================================================================
# Dataset
# ============================================================================

def apply_uncertain_policy_df(df, columns, policy='zeros'):
    """Apply uncertain-label policy on CheXpert labels.

    policy:
      - zeros: map -1 -> 0
      - ones:  map -1 -> 1
    """
    mapped = df.copy()
    uncertain_value = 0 if policy == 'zeros' else 1
    for col in columns:
        if col in mapped.columns:
            mapped[col] = mapped[col].fillna(0).replace(-1, uncertain_value)
    return mapped


class CheXpertHierarchicalDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, mode='train', uncertain_policy='zeros'):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.mode = mode
        self.uncertain_policy = uncertain_policy

        self.train_batch_dirs = [
            'CheXpert-v1.0 batch 2 (train 1)',
            'CheXpert-v1.0 batch 3 (train 2)',
            'CheXpert-v1.0 batch 4 (train 3)',
        ]
        self.valid_dir = Path('CheXpert-v1.0 batch 1 (validate & csv)') / 'valid'

        # Handle -1 (uncertain) labels with configurable policy
        self.df = apply_uncertain_policy_df(self.df, PARENT_LABELS + CHILD_LABELS, policy=self.uncertain_policy)
    
    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, img_path_str):
        """Resolve CSV image path to an existing file path with robust fallbacks."""
        candidates = []

        if 'CheXpert-v1.0/train/' in img_path_str:
            rel_path = img_path_str.split('CheXpert-v1.0/train/', 1)[1]
            candidates.extend(self.img_dir / batch_dir / rel_path for batch_dir in self.train_batch_dirs)
        elif 'CheXpert-v1.0/valid/' in img_path_str:
            rel_path = img_path_str.split('CheXpert-v1.0/valid/', 1)[1]
            candidates.append(self.img_dir / self.valid_dir / rel_path)

        # Generic fallbacks
        candidates.append(self.img_dir / img_path_str)
        candidates.append(self.img_dir / img_path_str.replace('CheXpert-v1.0/', '', 1))

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path_str = row['Path']
        img_path = self._resolve_image_path(img_path_str)

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image on error
            img = Image.new('RGB', (512, 512), color=0)
        
        if self.transform:
            img = self.transform(img)
        
        # Get labels
        parent_labels = torch.tensor([row[col] for col in PARENT_LABELS], dtype=torch.float32)
        child_labels = torch.tensor([row[col] for col in CHILD_LABELS], dtype=torch.float32)
        
        return img, parent_labels, child_labels


def build_pos_weights(df, columns, device, eps=1e-6, max_pos_weight=20.0, uncertain_policy='zeros'):
    """Build class-wise pos_weight = neg/pos for BCEWithLogitsLoss."""
    df_mapped = apply_uncertain_policy_df(df, columns, policy=uncertain_policy)
    pos_weights = []
    n = len(df_mapped)

    for col in columns:
        vals = df_mapped[col].values.astype(np.float32)
        pos = float(vals.sum())
        neg = float(n - pos)
        w = neg / max(pos, eps)
        w = min(max(w, 1.0), max_pos_weight)
        pos_weights.append(w)

    return torch.tensor(pos_weights, dtype=torch.float32, device=device)


def build_class_aware_sample_weights(df, columns, rare_threshold=0.05, boost=3.0, eps=1e-6, uncertain_policy='zeros'):
    """Build per-sample weights to oversample rare positive classes."""
    df_mapped = apply_uncertain_policy_df(df, columns, policy=uncertain_policy)
    y = df_mapped[columns].values.astype(np.float32)
    pos_rates = y.mean(axis=0)
    rare_mask = pos_rates < rare_threshold

    if not rare_mask.any():
        return np.ones(len(df), dtype=np.float32), pos_rates, []

    inv_rates = 1.0 / np.maximum(pos_rates, eps)
    rare_scores = y[:, rare_mask] * inv_rates[rare_mask]
    sample_score = rare_scores.max(axis=1) if rare_scores.ndim == 2 else rare_scores

    if np.max(sample_score) <= 0:
        return np.ones(len(df), dtype=np.float32), pos_rates, []

    sample_score = sample_score / (np.max(sample_score) + eps)
    weights = 1.0 + float(boost) * sample_score
    rare_cols = [col for col, is_rare in zip(columns, rare_mask) if is_rare]
    return weights.astype(np.float32), pos_rates, rare_cols


class MixedLoss(nn.Module):
    """Weighted mix of two criteria: ratio*loss_a + (1-ratio)*loss_b"""

    def __init__(self, loss_a, loss_b, ratio=0.7):
        super().__init__()
        self.loss_a = loss_a
        self.loss_b = loss_b
        self.ratio = ratio

    def forward(self, logits, targets):
        return self.ratio * self.loss_a(logits, targets) + (1.0 - self.ratio) * self.loss_b(logits, targets)


class FocalBCEWithLogitsLoss(nn.Module):
    """Focal BCE with logits for multi-label classification."""

    def __init__(self, gamma=2.0, alpha=0.25, pos_weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.register_buffer('pos_weight', pos_weight if pos_weight is not None else None)

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction='none',
            pos_weight=self.pos_weight
        )
        pt = torch.exp(-bce)
        focal = (1 - pt).pow(self.gamma) * bce

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal = alpha_t * focal

        if self.reduction == 'mean':
            return focal.mean()
        if self.reduction == 'sum':
            return focal.sum()
        return focal


# ============================================================================
# Model Architecture
# ============================================================================

class HierarchicalCheXpertModel(nn.Module):
    """Hierarchical model with separate heads for parent and child predictions"""
    
    def __init__(self, backbone_name='maxvit_large_tf_512', num_parents=6, num_children=13):
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for this model")
        
        # Get backbone config
        config = BACKBONES.get(backbone_name, BACKBONES['maxvit_large_tf_512'])
        
        # Create backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=config['pretrained'],
            num_classes=config['num_classes'],
        )
        
        # Get feature dimension
        if hasattr(self.backbone, 'num_features'):
            feat_dim = self.backbone.num_features
        elif hasattr(self.backbone, 'head'):
            feat_dim = self.backbone.head.in_features
        else:
            # Try to infer from forward pass
            dummy = torch.randn(1, 3, config['img_size'], config['img_size'])
            with torch.no_grad():
                feat_dim = self.backbone(dummy).shape[1]
        
        # Parent classifier
        self.parent_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_parents)
        )
        
        # Child classifier (conditioned on parents)
        self.child_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim + num_parents, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_children)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Parent predictions
        parent_logits = self.parent_head(features)
        parent_probs = torch.sigmoid(parent_logits)
        
        # Child predictions (conditioned on parent predictions)
        child_input = torch.cat([features, parent_probs], dim=1)
        child_logits = self.child_head(child_input)
        
        return parent_logits, child_logits


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def set_backbone_trainable(model, trainable: bool):
    base_model = unwrap_model(model)
    for p in base_model.backbone.parameters():
        p.requires_grad = trainable


class ModelEMA:
    """Exponential moving average of model weights for stabler validation AUC."""

    def __init__(self, model, decay=0.9997):
        self.decay = decay
        self.ema = deepcopy(unwrap_model(model)).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        model_state = unwrap_model(model).state_dict()
        ema_state = self.ema.state_dict()

        for k, v in ema_state.items():
            src = model_state[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            else:
                v.copy_(src)


def build_optimizer(model, lr, weight_decay=0.01, head_only=False):
    base_model = unwrap_model(model)
    if head_only:
        params = list(base_model.parent_head.parameters()) + list(base_model.child_head.parameters())
    else:
        params = model.parameters()

    return torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )


def build_scheduler(optimizer, scheduler_name, num_epochs, eta_min=1e-6):
    num_epochs = max(1, int(num_epochs))

    if scheduler_name == 'cosine_restarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, min(10, num_epochs)),
            T_mult=2,
            eta_min=eta_min
        )

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=eta_min
    )


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, loader, optimizer, device, epoch, parent_criterion, child_criterion,
                use_mixup=True, use_cutmix=False, mixup_alpha=0.2, cutmix_alpha=0.5,
                label_smoothing=0.02, ema=None, hierarchy_penalty_weight=0.0,
                grad_accum_steps=1, parent_loss_weight=1.0, child_loss_weight=2.0):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    grad_accum_steps = max(1, int(grad_accum_steps))
    optimizer.zero_grad()

    for batch_idx, (images, parent_labels, child_labels) in enumerate(pbar):
        images = images.to(device)
        parent_labels = parent_labels.to(device)
        child_labels = child_labels.to(device)
        
        # Apply label smoothing
        if label_smoothing > 0:
            parent_labels = parent_labels * (1 - label_smoothing) + label_smoothing / 2
            child_labels = child_labels * (1 - label_smoothing) + label_smoothing / 2
        
        # Randomly choose augmentation
        r = np.random.rand()
        if use_mixup and r < 0.33:
            # MixUp
            images, parent_a, parent_b, child_a, child_b, lam = mixup_data(
                images, parent_labels, child_labels, mixup_alpha
            )
            parent_out, child_out = model(images)
            
            parent_loss = lam * parent_criterion(parent_out, parent_a) + \
                          (1 - lam) * parent_criterion(parent_out, parent_b)
            child_loss = lam * child_criterion(child_out, child_a) + \
                         (1 - lam) * child_criterion(child_out, child_b)
        
        elif use_cutmix and r < 0.66:
            # CutMix
            images, parent_a, parent_b, child_a, child_b, lam = cutmix_data(
                images, parent_labels, child_labels, cutmix_alpha
            )
            parent_out, child_out = model(images)
            
            parent_loss = lam * parent_criterion(parent_out, parent_a) + \
                          (1 - lam) * parent_criterion(parent_out, parent_b)
            child_loss = lam * child_criterion(child_out, child_a) + \
                         (1 - lam) * child_criterion(child_out, child_b)
        
        else:
            # Regular training
            parent_out, child_out = model(images)
            parent_loss = parent_criterion(parent_out, parent_labels)
            child_loss = child_criterion(child_out, child_labels)
        
        # Hierarchical consistency penalty (child probability should not exceed parent)
        parent_prob = torch.sigmoid(parent_out)
        child_prob = torch.sigmoid(child_out)
        hierarchy_penalty = torch.zeros((), device=device)
        if hierarchy_penalty_weight > 0:
            penalties = []
            for p_idx, c_idx_list in PARENT_TO_CHILD.items():
                parent_p = parent_prob[:, p_idx].unsqueeze(1)
                child_p = child_prob[:, c_idx_list]
                penalties.append(F.relu(child_p - parent_p).mean())
            hierarchy_penalty = torch.stack(penalties).mean()

        # Total loss
        loss = parent_loss_weight * parent_loss + child_loss_weight * child_loss + hierarchy_penalty_weight * hierarchy_penalty

        # Gradient accumulation (for stable effective batch with limited VRAM)
        loss_for_backward = loss / grad_accum_steps
        loss_for_backward.backward()

        should_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(loader))
        if should_step:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)

        running_loss += loss.item()
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'p_loss': f'{parent_loss.item():.4f}',
            'c_loss': f'{child_loss.item():.4f}',
            'h_pen': f'{hierarchy_penalty.item():.4f}'
        })
    
    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device, use_tta=False, return_arrays=False):
    model.eval()
    
    all_parent_preds = []
    all_parent_labels = []
    all_child_preds = []
    all_child_labels = []
    
    for images, parent_labels, child_labels in tqdm(loader, desc='Validating'):
        images = images.to(device)
        
        if use_tta:
            # Test-Time Augmentation
            parent_preds_tta = []
            child_preds_tta = []
            
            # Original
            p_out, c_out = model(images)
            parent_preds_tta.append(torch.sigmoid(p_out))
            child_preds_tta.append(torch.sigmoid(c_out))
            
            # Horizontal flip
            p_out, c_out = model(torch.flip(images, dims=[3]))
            parent_preds_tta.append(torch.sigmoid(p_out))
            child_preds_tta.append(torch.sigmoid(c_out))

            # Mild scale TTA (zoom-in then resize back)
            scaled = F.interpolate(images, scale_factor=1.05, mode='bilinear', align_corners=False)
            scaled = F.interpolate(scaled, size=images.shape[-2:], mode='bilinear', align_corners=False)
            p_out, c_out = model(scaled)
            parent_preds_tta.append(torch.sigmoid(p_out))
            child_preds_tta.append(torch.sigmoid(c_out))
            
            # Average predictions
            parent_preds = torch.stack(parent_preds_tta).mean(0)
            child_preds = torch.stack(child_preds_tta).mean(0)
        else:
            parent_out, child_out = model(images)
            parent_preds = torch.sigmoid(parent_out)
            child_preds = torch.sigmoid(child_out)
        
        all_parent_preds.append(parent_preds.cpu())
        all_parent_labels.append(parent_labels)
        all_child_preds.append(child_preds.cpu())
        all_child_labels.append(child_labels)
    
    # Concatenate all predictions
    all_parent_preds = torch.cat(all_parent_preds, dim=0).numpy()
    all_parent_labels = torch.cat(all_parent_labels, dim=0).numpy()
    all_child_preds = torch.cat(all_child_preds, dim=0).numpy()
    all_child_labels = torch.cat(all_child_labels, dim=0).numpy()
    
    # Calculate AUC for each class
    parent_aucs = []
    for i in range(all_parent_labels.shape[1]):
        if len(np.unique(all_parent_labels[:, i])) > 1:
            auc = roc_auc_score(all_parent_labels[:, i], all_parent_preds[:, i])
            parent_aucs.append(auc)
    
    child_aucs = []
    child_auc_per_class = {}
    for i in range(all_child_labels.shape[1]):
        if len(np.unique(all_child_labels[:, i])) > 1:
            auc = roc_auc_score(all_child_labels[:, i], all_child_preds[:, i])
            child_aucs.append(auc)
            child_auc_per_class[CHILD_LABELS[i]] = float(auc)
    
    parent_auc = np.mean(parent_aucs) if parent_aucs else 0.0
    child_auc = np.mean(child_aucs) if child_aucs else 0.0
    mean_auc = (parent_auc + child_auc) / 2

    if return_arrays:
        return mean_auc, parent_auc, child_auc, all_parent_preds, all_parent_labels, all_child_preds, all_child_labels, child_auc_per_class
    return mean_auc, parent_auc, child_auc


def compute_optimal_thresholds(y_true, y_prob, label_names, method='youden'):
    thresholds = {}
    for i, name in enumerate(label_names):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        if len(np.unique(yt)) < 2:
            thresholds[name] = 0.5
            continue
        if method == 'f1':
            cand = np.linspace(0.05, 0.95, 19)
            best_t, best_s = 0.5, -1
            for t in cand:
                s = f1_score(yt, (yp >= t).astype(int), zero_division=0)
                if s > best_s:
                    best_s, best_t = s, t
            thresholds[name] = float(best_t)
        else:
            fpr, tpr, thr = roc_curve(yt, yp)
            j = tpr - fpr
            thresholds[name] = float(thr[np.argmax(j)])
    return thresholds


def fit_temperature_per_class(y_true, y_prob, label_names):
    """Fit per-class temperature scaling on validation probabilities via grid search."""
    temps = {}
    eps = 1e-6
    y_prob = np.clip(y_prob, eps, 1 - eps)
    logits = np.log(y_prob / (1 - y_prob))
    candidates = np.array([0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0], dtype=np.float32)

    for i, name in enumerate(label_names):
        yt = y_true[:, i].astype(np.float32)
        if len(np.unique(yt)) < 2:
            temps[name] = 1.0
            continue

        z = logits[:, i]
        best_t, best_nll = 1.0, float('inf')
        for t in candidates:
            p = 1.0 / (1.0 + np.exp(-z / t))
            p = np.clip(p, eps, 1 - eps)
            nll = -np.mean(yt * np.log(p) + (1 - yt) * np.log(1 - p))
            if nll < best_nll:
                best_nll, best_t = nll, float(t)
        temps[name] = best_t
    return temps


def apply_temperature_per_class(y_prob, label_names, temperatures):
    eps = 1e-6
    y_prob = np.clip(y_prob, eps, 1 - eps)
    logits = np.log(y_prob / (1 - y_prob))
    out = np.zeros_like(y_prob)
    for i, name in enumerate(label_names):
        t = float(temperatures.get(name, 1.0))
        out[:, i] = 1.0 / (1.0 + np.exp(-logits[:, i] / max(t, 1e-6)))
    return np.clip(out, eps, 1 - eps)


def audit_label_noise(val_df, y_true, y_prob, child_auc_per_class, output_dir, top_k_classes=3, top_n=50):
    if len(child_auc_per_class) == 0:
        return
    sorted_cls = sorted(child_auc_per_class.items(), key=lambda x: x[1])[:top_k_classes]
    records = []
    for cls_name, auc in sorted_cls:
        ci = CHILD_LABELS.index(cls_name)
        yt = y_true[:, ci]
        yp = y_prob[:, ci]
        fp_idx = np.where((yt == 0) & (yp >= 0.8))[0]
        fn_idx = np.where((yt == 1) & (yp <= 0.2))[0]
        fp_sorted = fp_idx[np.argsort(-yp[fp_idx])][:top_n]
        fn_sorted = fn_idx[np.argsort(yp[fn_idx])][:top_n]
        for idx in fp_sorted:
            records.append({'type': 'FP', 'class': cls_name, 'class_auc': float(auc), 'row_idx': int(idx), 'score': float(yp[idx])})
        for idx in fn_sorted:
            records.append({'type': 'FN', 'class': cls_name, 'class_auc': float(auc), 'row_idx': int(idx), 'score': float(yp[idx])})

    if records:
        out = pd.DataFrame(records)
        # Attach path if present
        if 'Path' in val_df.columns:
            out['Path'] = out['row_idx'].map(lambda i: val_df.iloc[i]['Path'])
        out_path = Path(output_dir) / 'label_noise_audit.csv'
        out.to_csv(out_path, index=False)
        print(f"Saved label-noise audit: {out_path}")


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Backward-compatible root. Used for both csv and images unless csv_dir/img_dir are provided.')
    parser.add_argument('--csv_dir', type=str, default=None,
                        help='Directory containing train/valid CSV files (defaults to data_dir).')
    parser.add_argument('--img_dir', type=str, default=None,
                        help='CheXpert image root containing batch folders (defaults to data_dir).')
    parser.add_argument('--train_csv', type=str, default='train_hierarchical.csv')
    parser.add_argument('--val_csv', type=str, default='valid_hierarchical.csv')
    parser.add_argument('--backbone', type=str, default='maxvit_large_tf_512',
                        choices=list(BACKBONES.keys()))
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Accumulate gradients over N batches before optimizer step.')
    parser.add_argument('--output_dir', type=str, default='./ensemble_models')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--multi_gpu', action='store_true', help='Use DataParallel across selected GPUs')
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='Comma-separated GPU ids for DataParallel (e.g. 0,1)')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)

    # Robust toggles (support both enable and disable from CLI)
    parser.add_argument('--use_mixup', dest='use_mixup', action='store_true')
    parser.add_argument('--no_mixup', dest='use_mixup', action='store_false')
    parser.add_argument('--use_cutmix', dest='use_cutmix', action='store_true')
    parser.add_argument('--no_cutmix', dest='use_cutmix', action='store_false')
    parser.add_argument('--use_tta', dest='use_tta', action='store_true')
    parser.add_argument('--no_tta', dest='use_tta', action='store_false')
    parser.set_defaults(use_mixup=True, use_cutmix=False, use_tta=False)

    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--cutmix_alpha', type=float, default=0.5)
    parser.add_argument('--label_smoothing', type=float, default=0.02)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'cosine_restarts'])

    parser.add_argument('--use_pos_weight', dest='use_pos_weight', action='store_true')
    parser.add_argument('--no_pos_weight', dest='use_pos_weight', action='store_false')

    parser.add_argument('--two_stage', dest='two_stage', action='store_true')
    parser.add_argument('--no_two_stage', dest='two_stage', action='store_false')
    parser.add_argument('--head_warmup_epochs', type=int, default=2,
                        help='Epochs to train heads only (backbone frozen) before full fine-tuning.')
    parser.add_argument('--finetune_lr', type=float, default=1.5e-6,
                        help='LR after unfreezing backbone in stage 2.')

    parser.add_argument('--late_stage_mixup_off_epoch', type=int, default=5,
                        help='Disable MixUp from this epoch onward (0 = never disable).')

    parser.add_argument('--class_aware_sampling', dest='class_aware_sampling', action='store_true')
    parser.add_argument('--no_class_aware_sampling', dest='class_aware_sampling', action='store_false')
    parser.add_argument('--rare_threshold', type=float, default=0.05,
                        help='Classes with positive rate below this threshold are treated as rare.')
    parser.add_argument('--sampler_boost', type=float, default=1.5,
                        help='Oversampling strength for rare positive samples.')

    parser.add_argument('--child_loss', type=str, default='focal', choices=['bce', 'focal'])
    parser.add_argument('--focal_gamma', type=float, default=1.5)
    parser.add_argument('--focal_alpha', type=float, default=0.2)
    parser.add_argument('--parent_aucm_bce_mix', type=float, default=0.0,
                        help='0..1 mix for parent AUCM+BCE (0 disables AUCM).')
    parser.add_argument('--child_focal_bce_mix', type=float, default=0.7,
                        help='0..1 mix for child Focal+BCE (1=focal only).')
    parser.add_argument('--parent_loss_weight', type=float, default=1.0,
                        help='Weight for parent loss in total loss.')
    parser.add_argument('--child_loss_weight', type=float, default=2.0,
                        help='Weight for child loss in total loss.')

    parser.add_argument('--ema', dest='use_ema', action='store_true')
    parser.add_argument('--no_ema', dest='use_ema', action='store_false')
    parser.add_argument('--ema_decay', type=float, default=0.9997)

    parser.set_defaults(use_pos_weight=True, two_stage=True, use_ema=True, class_aware_sampling=True)
    parser.add_argument('--resume_checkpoint', type=str, default='',
                        help='Path to checkpoint (.pth) to resume from. Can be absolute or relative to output_dir/backbone.')
    parser.add_argument('--no_resume_optimizer', action='store_true',
                        help='Resume model/EMA weights but rebuild optimizer/scheduler from CLI LR settings.')
    parser.add_argument('--early_stop_patience', type=int, default=7)
    parser.add_argument('--plateau_patience', type=int, default=4,
                        help='Reduce LR if no improvement for this many epochs.')
    parser.add_argument('--plateau_lr_factor', type=float, default=0.3,
                        help='LR multiply factor on plateau (e.g. 0.3 => ~3.3x down).')
    parser.add_argument('--min_lr', type=float, default=1e-7)
    parser.add_argument('--threshold_method', type=str, default='youden', choices=['youden', 'f1'])
    parser.add_argument('--uncertain_policy', type=str, default='zeros', choices=['zeros', 'ones'],
                        help='How to map uncertain labels (-1): zeros=U-Zeros, ones=U-Ones.')
    parser.add_argument('--save_thresholds', action='store_true',
                        help='Save per-class optimal thresholds from val ROC/F1 each epoch.')
    parser.add_argument('--label_noise_audit', action='store_true',
                        help='Audit top FP/FN samples for weakest classes and save CSV.')
    parser.add_argument('--hierarchy_penalty_weight', type=float, default=0.15,
                        help='Weight for hierarchy consistency penalty (child prob > parent prob).')
    parser.add_argument('--calibrate_thresholds', dest='calibrate_thresholds', action='store_true',
                        help='Fit per-class temperature scaling on val probs before threshold search.')
    parser.add_argument('--no_calibrate_thresholds', dest='calibrate_thresholds', action='store_false')
    parser.set_defaults(calibrate_thresholds=True)
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Device / Multi-GPU
    use_multi_gpu = False
    device_ids = [args.gpu]

    if torch.cuda.is_available() and args.multi_gpu:
        parsed_ids = [int(x.strip()) for x in args.gpu_ids.split(',') if x.strip()]
        valid_ids = [gid for gid in parsed_ids if gid < torch.cuda.device_count()]
        if len(valid_ids) >= 2:
            use_multi_gpu = True
            device_ids = valid_ids
        else:
            print(f"⚠️ multi_gpu requested but valid gpu_ids={valid_ids}; falling back to single GPU")

    device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')
    if use_multi_gpu:
        print(f"Using device: DataParallel on GPUs {device_ids}")
    else:
        print(f"Using device: {device}")
    print(f"Backbone: {args.backbone}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.backbone
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get image size from backbone config
    img_size = BACKBONES[args.backbone]['img_size']
    
    # Resolve data roots
    csv_dir = Path(args.csv_dir) if args.csv_dir else Path(args.data_dir)
    img_dir = Path(args.img_dir) if args.img_dir else Path(args.data_dir)
    train_csv_path = Path(args.train_csv)
    val_csv_path = Path(args.val_csv)
    if not train_csv_path.is_absolute():
        train_csv_path = csv_dir / train_csv_path
    if not val_csv_path.is_absolute():
        val_csv_path = csv_dir / val_csv_path

    print(f"CSV dir: {csv_dir}")
    print(f"Image dir: {img_dir}")
    print(f"Train CSV: {train_csv_path}")
    print(f"Val CSV: {val_csv_path}")

    # Create datasets
    train_transform = AdvancedAugmentation(img_size=img_size, mode='train')
    val_transform = AdvancedAugmentation(img_size=img_size, mode='val')
    
    train_dataset = CheXpertHierarchicalDataset(
        csv_path=train_csv_path,
        img_dir=img_dir,
        transform=train_transform,
        mode='train',
        uncertain_policy=args.uncertain_policy
    )
    
    val_dataset = CheXpertHierarchicalDataset(
        csv_path=val_csv_path,
        img_dir=img_dir,
        transform=val_transform,
        mode='val',
        uncertain_policy=args.uncertain_policy
    )
    
    train_sampler = None
    if args.class_aware_sampling:
        sample_weights, child_pos_rates, rare_child_cols = build_class_aware_sample_weights(
            train_dataset.df,
            CHILD_LABELS,
            rare_threshold=args.rare_threshold,
            boost=args.sampler_boost,
            uncertain_policy=args.uncertain_policy
        )
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True
        )
        print(
            f"Using class-aware sampling | rare_threshold={args.rare_threshold:.3f}, "
            f"boost={args.sampler_boost:.2f}, rare_child_labels={len(rare_child_cols)}"
        )
        if rare_child_cols:
            print("Rare child labels: " + ", ".join(rare_child_cols))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    model = HierarchicalCheXpertModel(
        backbone_name=args.backbone,
        num_parents=len(PARENT_LABELS),
        num_children=len(CHILD_LABELS)
    ).to(device)

    if use_multi_gpu:
        print(f"Wrapping model with DataParallel ({len(device_ids)} GPUs)")
        model = nn.DataParallel(model, device_ids=device_ids)
    
    # Losses (imbalance-aware + optional mixed objectives)
    parent_pos_weight = build_pos_weights(train_dataset.df, PARENT_LABELS, device, uncertain_policy=args.uncertain_policy) if args.use_pos_weight else None
    child_pos_weight = build_pos_weights(train_dataset.df, CHILD_LABELS, device, uncertain_policy=args.uncertain_policy) if args.use_pos_weight else None

    parent_bce = nn.BCEWithLogitsLoss(pos_weight=parent_pos_weight) if parent_pos_weight is not None else nn.BCEWithLogitsLoss()
    if args.parent_aucm_bce_mix > 0:
        parent_aucm = AUCMLoss()
        parent_criterion = MixedLoss(parent_aucm, parent_bce, ratio=args.parent_aucm_bce_mix)
        print(f"Using parent AUCM+BCE mixed loss | aucm_ratio={args.parent_aucm_bce_mix:.2f}")
    else:
        parent_criterion = parent_bce
        print("Using parent BCEWithLogitsLoss")

    if args.child_loss == 'focal':
        child_focal = FocalBCEWithLogitsLoss(
            gamma=args.focal_gamma,
            alpha=args.focal_alpha,
            pos_weight=child_pos_weight
        )
        child_bce = nn.BCEWithLogitsLoss(pos_weight=child_pos_weight) if child_pos_weight is not None else nn.BCEWithLogitsLoss()
        if args.child_focal_bce_mix < 1.0:
            child_criterion = MixedLoss(child_focal, child_bce, ratio=args.child_focal_bce_mix)
            print(f"Using child Focal+BCE mixed loss | focal_ratio={args.child_focal_bce_mix:.2f}")
        else:
            child_criterion = child_focal
            print(f"Using focal child loss | gamma={args.focal_gamma}, alpha={args.focal_alpha}")
    else:
        child_criterion = nn.BCEWithLogitsLoss(pos_weight=child_pos_weight) if child_pos_weight is not None else nn.BCEWithLogitsLoss()
        print("Using child BCEWithLogitsLoss")

    # Optional EMA model (evaluated instead of raw weights)
    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None
    if ema is not None:
        print(f"Using EMA with decay={args.ema_decay}")

    # Optional checkpoint resume
    resume_ckpt = None
    resume_epoch = 0
    start_epoch = 1
    best_auc = 0.0
    best_parent_thr = {k: 0.5 for k in PARENT_LABELS}
    best_child_thr = {k: 0.5 for k in CHILD_LABELS}
    best_parent_temp = {k: 1.0 for k in PARENT_LABELS}
    best_child_temp = {k: 1.0 for k in CHILD_LABELS}

    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if not resume_path.is_absolute():
            resume_path = output_dir / resume_path
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        print(f"Loading resume checkpoint: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        resume_epoch = int(resume_ckpt.get('epoch', 0))
        start_epoch = resume_epoch + 1
        best_auc = float(resume_ckpt.get('best_auc', 0.0))
        best_parent_thr = resume_ckpt.get('best_parent_thresholds', best_parent_thr)
        best_child_thr = resume_ckpt.get('best_child_thresholds', best_child_thr)
        best_parent_temp = resume_ckpt.get('best_parent_temperatures', best_parent_temp)
        best_child_temp = resume_ckpt.get('best_child_temperatures', best_child_temp)

        raw_state = resume_ckpt.get('raw_model_state_dict')
        model_state = resume_ckpt.get('model_state_dict')

        if raw_state is not None:
            unwrap_model(model).load_state_dict(raw_state, strict=False)
            print("Loaded raw_model_state_dict into model")
        elif model_state is not None:
            unwrap_model(model).load_state_dict(model_state, strict=False)
            print("Loaded model_state_dict into model")
        else:
            raise KeyError("Checkpoint missing model_state_dict/raw_model_state_dict")

        if ema is not None:
            ema_state = resume_ckpt.get('ema_state_dict')
            if ema_state is not None:
                ema.ema.load_state_dict(ema_state, strict=False)
                print("Loaded ema_state_dict")
            elif model_state is not None:
                ema.ema.load_state_dict(model_state, strict=False)
                print("EMA state missing; initialized EMA from model_state_dict")

        print(f"Resuming from epoch {resume_epoch}; next epoch will be {start_epoch}")

    # Two-stage fine-tuning setup
    stage1_epochs = min(args.head_warmup_epochs, args.epochs) if args.two_stage else 0
    stage2_lr = args.finetune_lr if args.finetune_lr > 0 else args.lr * 0.33

    if stage1_epochs > 0:
        if resume_epoch >= stage1_epochs:
            set_backbone_trainable(model, True)
            optimizer = build_optimizer(model, lr=stage2_lr, head_only=False)
            scheduler = build_scheduler(optimizer, args.scheduler, num_epochs=max(1, args.epochs - stage1_epochs))
            print(f"Stage 2 (resumed): full unfreeze, lr={stage2_lr:.2e}")
        else:
            set_backbone_trainable(model, False)
            optimizer = build_optimizer(model, lr=args.lr, head_only=True)
            scheduler = build_scheduler(optimizer, args.scheduler, num_epochs=stage1_epochs)
            print(f"Stage 1: head-only warmup for {stage1_epochs} epoch(s), lr={args.lr:.2e}")
    else:
        set_backbone_trainable(model, True)
        optimizer = build_optimizer(model, lr=stage2_lr, head_only=False)
        scheduler = build_scheduler(optimizer, args.scheduler, num_epochs=args.epochs)
        print(f"Single-stage full fine-tuning, lr={stage2_lr:.2e}")

    if resume_ckpt is not None and resume_ckpt.get('optimizer_state_dict') is not None and not args.no_resume_optimizer:
        try:
            optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
            print("Loaded optimizer_state_dict")
        except Exception as e:
            print(f"⚠️ Could not load optimizer_state_dict: {e}")
    elif resume_ckpt is not None and args.no_resume_optimizer:
        print("Skipped optimizer_state_dict; using freshly built optimizer/scheduler from CLI LR settings")

    # Training loop
    patience_counter = 0

    log_file = output_dir / f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

    if start_epoch > args.epochs:
        print(f"Resume epoch ({resume_epoch}) is already >= total epochs ({args.epochs}); nothing to train.")

    for epoch in range(start_epoch, args.epochs + 1):
        # Transition to stage 2: unfreeze backbone + lower LR
        if stage1_epochs > 0 and epoch == stage1_epochs + 1:
            set_backbone_trainable(model, True)
            optimizer = build_optimizer(model, lr=stage2_lr, head_only=False)
            scheduler = build_scheduler(optimizer, args.scheduler, num_epochs=max(1, args.epochs - stage1_epochs))
            print(f"\n>>> Stage 2 starts at epoch {epoch}: full unfreeze, lr={stage2_lr:.2e}")

        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Late-stage regularization control
        epoch_use_mixup = args.use_mixup
        if args.late_stage_mixup_off_epoch > 0 and epoch >= args.late_stage_mixup_off_epoch:
            epoch_use_mixup = False
            if epoch == args.late_stage_mixup_off_epoch:
                print(f">>> Late stage starts at epoch {epoch}: MixUp disabled")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch,
            parent_criterion=parent_criterion,
            child_criterion=child_criterion,
            use_mixup=epoch_use_mixup,
            use_cutmix=args.use_cutmix,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            label_smoothing=args.label_smoothing,
            ema=ema,
            hierarchy_penalty_weight=args.hierarchy_penalty_weight,
            grad_accum_steps=args.grad_accum_steps,
            parent_loss_weight=args.parent_loss_weight,
            child_loss_weight=args.child_loss_weight
        )

        # Validate (EMA weights if enabled)
        eval_model = ema.ema if ema is not None else model
        val_out = validate(
            eval_model, val_loader, device, use_tta=args.use_tta,
            return_arrays=(args.save_thresholds or args.label_noise_audit)
        )
        if args.save_thresholds or args.label_noise_audit:
            mean_auc, parent_auc, child_auc, p_pred, p_true, c_pred, c_true, child_auc_map = val_out
        else:
            mean_auc, parent_auc, child_auc = val_out

        # Step scheduler
        scheduler.step()
        
        # Log
        log_msg = (
            f"Epoch {epoch}: "
            f"Train Loss: {train_loss:.4f} | "
            f"Val AUC: {mean_auc:.4f} | "
            f"Parent AUC: {parent_auc:.4f} | "
            f"Child AUC: {child_auc:.4f}"
        )
        print(f"\n{log_msg}")
        
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')

        if args.save_thresholds:
            if args.calibrate_thresholds:
                parent_temp = fit_temperature_per_class(p_true, p_pred, PARENT_LABELS)
                child_temp = fit_temperature_per_class(c_true, c_pred, CHILD_LABELS)
                p_pred_cal = apply_temperature_per_class(p_pred, PARENT_LABELS, parent_temp)
                c_pred_cal = apply_temperature_per_class(c_pred, CHILD_LABELS, child_temp)
            else:
                parent_temp = {k: 1.0 for k in PARENT_LABELS}
                child_temp = {k: 1.0 for k in CHILD_LABELS}
                p_pred_cal = p_pred
                c_pred_cal = c_pred

            child_thr = compute_optimal_thresholds(c_true, c_pred_cal, CHILD_LABELS, method=args.threshold_method)
            parent_thr = compute_optimal_thresholds(p_true, p_pred_cal, PARENT_LABELS, method=args.threshold_method)
            thr_path = output_dir / f'thresholds_epoch_{epoch:03d}.json'
            with open(thr_path, 'w') as tf:
                json.dump({
                    'epoch': epoch,
                    'method': args.threshold_method,
                    'calibrated': bool(args.calibrate_thresholds),
                    'parent_temperatures': parent_temp,
                    'child_temperatures': child_temp,
                    'parent_thresholds': parent_thr,
                    'child_thresholds': child_thr
                }, tf, indent=2)
            print(f"Saved thresholds: {thr_path}")

        if args.label_noise_audit:
            audit_label_noise(val_dataset.df.reset_index(drop=True), c_true, c_pred, child_auc_map, output_dir)


        # Always save latest checkpoint for robust resume
        raw_model_to_save = unwrap_model(model)
        model_to_save = ema.ema if ema is not None else raw_model_to_save
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'raw_model_state_dict': raw_model_to_save.state_dict(),
            'ema_state_dict': ema.ema.state_dict() if ema is not None else None,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_auc': best_auc,
            'last_auc': mean_auc,
            'parent_auc': parent_auc,
            'child_auc': child_auc,
            'use_ema': ema is not None,
            'best_parent_thresholds': best_parent_thr,
            'best_child_thresholds': best_child_thr,
            'best_parent_temperatures': best_parent_temp,
            'best_child_temperatures': best_child_temp,
        }, output_dir / 'latest_model.pth')

        # Save best model
        if mean_auc > best_auc:
            best_auc = mean_auc
            patience_counter = 0

            if args.save_thresholds:
                best_parent_thr = parent_thr
                best_child_thr = child_thr
                best_parent_temp = parent_temp
                best_child_temp = child_temp

            model_to_save = ema.ema if ema is not None else unwrap_model(model)
            raw_model_to_save = unwrap_model(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'raw_model_state_dict': raw_model_to_save.state_dict(),
                'ema_state_dict': ema.ema.state_dict() if ema is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'parent_auc': parent_auc,
                'child_auc': child_auc,
                'use_ema': ema is not None,
                'best_parent_thresholds': best_parent_thr,
                'best_child_thresholds': best_child_thr,
                'best_parent_temperatures': best_parent_temp,
                'best_child_temperatures': best_child_temp,
            }, output_dir / 'best_model.pth')

            print(f"✓ New best model saved! AUC: {best_auc:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.early_stop_patience}")
            if patience_counter >= args.plateau_patience:
                for pg in optimizer.param_groups:
                    old_lr = pg['lr']
                    new_lr = max(args.min_lr, old_lr * args.plateau_lr_factor)
                    pg['lr'] = new_lr
                print(f"Plateau LR step applied (factor={args.plateau_lr_factor}): {old_lr:.2e} -> {new_lr:.2e}")
        
        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
