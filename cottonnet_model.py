"""
CottonNet Model Architecture Definition
Based on EfficientNet-B0 backbone with Squeeze-and-Excitation (SE) channel attention,
Dual Pooling (AdaptiveAvgPool + AdaptiveMaxPool), and regularized classification head.
Trained on SAR-CLD-2024 Cotton Leaf Dataset via Knowledge Distillation from DenseNet121.
"""

import torch
import torch.nn as nn
import timm

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CottonNet(nn.Module):
    """
    Lightweight CNN for cotton leaf disease classification.
    
    Attributes:
        num_classes: Number of disease categories (default: 7)
        dropout_1: First dropout rate in classification head (default: 0.4)
        dropout_2: Second dropout rate in classification head (default: 0.3)
        freeze_blocks: Number of early EfficientNet blocks frozen during training (default: 4)
    """
    CLASS_NAMES = [
        "Bacterial Blight",
        "Curl Virus",
        "Healthy Leaf",
        "Herbicide Growth Damage",
        "Leaf Hopper Jassids",
        "Leaf Redding",
        "Leaf Variegation"
    ]

    def __init__(self, num_classes=7, dropout_1=0.4, dropout_2=0.3, freeze_blocks=4):
        super().__init__()
        
        # EfficientNet-B0 backbone without default classifier
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        self.feature_dim = self.backbone.num_features  # 1280
        
        # Freeze early blocks
        self._freeze_backbone(freeze_blocks)
        
        # Channel Attention
        self.se_attention = SEBlock(self.feature_dim, reduction=16)
        
        # Dual Pooling (Average + Max)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Classification Head (1280 * 2 = 2560 input features)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_1),
            nn.Linear(self.feature_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_2),
            nn.Linear(256, num_classes)
        )
        
        self._init_classifier()
    
    def _freeze_backbone(self, freeze_blocks):
        """Freeze stem and first N blocks."""
        if hasattr(self.backbone, 'conv_stem'):
            for param in self.backbone.conv_stem.parameters():
                param.requires_grad = False
        if hasattr(self.backbone, 'bn1'):
            for param in self.backbone.bn1.parameters():
                param.requires_grad = False
        
        if hasattr(self.backbone, 'blocks'):
            for i, block in enumerate(self.backbone.blocks):
                if i < freeze_blocks:
                    for param in block.parameters():
                        param.requires_grad = False
    
    def _init_classifier(self):
        """Xavier initialization for classifier layers."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Extract features from backbone: [B, 1280, 7, 7] for 224x224 input
        features = self.backbone.forward_features(x)
        
        # Channel Attention
        features = self.se_attention(features)
        
        # Dual Pooling & Concatenation
        avg_out = self.avg_pool(features).flatten(1)
        max_out = self.max_pool(features).flatten(1)
        pooled = torch.cat([avg_out, max_out], dim=1)
        
        # Classification
        out = self.classifier(pooled)
        return out

    def get_gradcam_target_layer(self):
        """Returns target layer for GradCAM feature attribution."""
        return [self.backbone.blocks[-1]]
    
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

if __name__ == "__main__":
    model = CottonNet(num_classes=7)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    total_p, trainable_p = model.count_parameters()
    print(f"CottonNet initialized successfully!")
    print(f"Output shape: {out.shape}")
    print(f"Total parameters: {total_p:,} ({total_p/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_p:,} ({trainable_p/1e6:.2f}M)")
