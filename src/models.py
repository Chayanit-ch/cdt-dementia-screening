import torch
import torch.nn as nn
import timm
from torchvision import models


class CDTModel(nn.Module):
    """Vision-only model"""

    def __init__(self, backbone_name="vit",
                 num_domains=5, num_classes=3):
        super().__init__()
        if backbone_name == "vit":
            self.backbone = timm.create_model(
                "vit_base_patch16_224",
                pretrained=True, num_classes=0
            )
            self.feature_dim = 768
        else:
            backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1
            )
            self.feature_dim = 2048
            self.backbone = nn.Sequential(
                *list(backbone.children())[:-1]
            )
        self.backbone_name = backbone_name
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(), nn.Dropout(0.3)
        )
        self.heads_reg = nn.ModuleList([
            nn.Linear(512, 1)
            for _ in range(num_domains)
        ])
        self.heads_cls = nn.ModuleList([
            nn.Linear(512, num_classes)
            for _ in range(num_domains)
        ])

    def forward(self, x, mode="reg"):
        if self.backbone_name == "vit":
            feat = self.backbone(x)
            feat = feat.unsqueeze(-1).unsqueeze(-1)
        else:
            feat = self.backbone(x)
        feat = self.shared(feat)
        if mode == "reg":
            return [h(feat).squeeze(-1)
                    for h in self.heads_reg]
        return [h(feat) for h in self.heads_cls]


class CDTHybridModel(nn.Module):
    """Hybrid model: Vision + VQA features"""

    def __init__(self, backbone_name="vit",
                 vqa_dim=12, num_domains=5):
        super().__init__()
        if backbone_name == "vit":
            self.backbone = timm.create_model(
                "vit_base_patch16_224",
                pretrained=True, num_classes=0
            )
            vis_dim = 768
        else:
            backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1
            )
            self.backbone = nn.Sequential(
                *list(backbone.children())[:-1]
            )
            vis_dim = 2048
        self.backbone_name = backbone_name
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.vis_shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(vis_dim, 512),
            nn.ReLU(), nn.Dropout(0.3)
        )
        self.vqa_encoder = nn.Sequential(
            nn.Linear(vqa_dim, 64),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(512+32, 256),
            nn.ReLU(), nn.Dropout(0.3)
        )
        self.heads_reg = nn.ModuleList([
            nn.Linear(256, 1)
            for _ in range(num_domains)
        ])
        self.heads_cls = nn.ModuleList([
            nn.Linear(256, 3)
            for _ in range(num_domains)
        ])

    def forward(self, img, vqa_feat, mode="reg"):
        if self.backbone_name == "vit":
            vis = self.backbone(img)
            vis = vis.unsqueeze(-1).unsqueeze(-1)
        else:
            vis = self.backbone(img)
        vis   = self.vis_shared(vis)
        vqa   = self.vqa_encoder(vqa_feat)
        fused = self.fusion(
            torch.cat([vis, vqa], dim=1)
        )
        if mode == "reg":
            return [h(fused).squeeze(-1)
                    for h in self.heads_reg]
        return [h(fused) for h in self.heads_cls]
