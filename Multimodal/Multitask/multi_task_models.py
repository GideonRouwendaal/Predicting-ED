import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights


class ClassToken(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, hidden_dim))
    def forward(self, x):
        B = x.size(0)
        return self.token.expand(B, -1, -1)

class MLP(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, num_heads, dropout_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.drop1 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_dim, dropout_rate)
    def forward(self, x):
        x_attn, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop1(x_attn)
        x_mlp = self.mlp(self.norm2(x))
        return x + x_mlp


class ResNetViTBackbone(nn.Module):
    def __init__(self, cf):
        super().__init__()
        # Base ResNet18
        self.cf = cf

        if cf['pretrained']:
            resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)
        # if cf.num_channels == 1:
        original_conv = resnet.conv1
        # Create new conv layer with 1 input channel
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )

        # Average weights over the RGB input channels
        with torch.no_grad():
            new_conv.weight[:] = original_conv.weight.mean(dim=1, keepdim=True)
        resnet.conv1 = new_conv

        self.backbone = resnet
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove avgpool & fc

        # Patch Embedding
        # input_dim_patch_proj = 512 if cf.ResNetSize == 18 else 2048
        input_dim_patch_proj = 512
        self.patch_proj = nn.Conv2d(input_dim_patch_proj, cf["hidden_dim"], kernel_size=cf["patch_size"], padding="same")
        self.bn = nn.BatchNorm2d(cf["hidden_dim"])

        # Positional Embedding
        # self.pos_embed = nn.Embedding(cf.num_patches, cf.hidden_dim)
        self.pos_embed = None

        self.feat_dim = cf["hidden_dim"]  # Output feature dimension after patch projection

        # Class token
        self.cls_token = ClassToken(cf["hidden_dim"])

        # Transformer layers
        self.transformers = nn.ModuleList([
            TransformerEncoder(
                hidden_dim=cf["hidden_dim"],
                mlp_dim=cf["mlp_dim"],
                num_heads=cf["num_heads"],
                dropout_rate=cf["dropout_rate"]
            ) for _ in range(cf["num_layers"])
        ])

        self.norm = nn.LayerNorm(cf["hidden_dim"])
        self.head = nn.Linear(cf["hidden_dim"], cf["num_classes"])

        self.final_dropout = nn.Dropout(cf["dropout_rate"])


    def forward(self, x):
        B = x.size(0)
        x = self.backbone(x)  # (B, 2048, 4, 4) for 128x128 input
        x = self.patch_proj(x)  # (B, hidden_dim, 4, 4)
        x = self.bn(x)

        # Flatten patches
        x = x.flatten(2).transpose(1, 2)  # (B, 16, hidden_dim)
        if self.pos_embed is None:
            self.pos_embed = nn.Embedding(x.size(1), self.feat_dim).to(x.device)
        # Add position embedding
        # pos = torch.arange(self.cf.num_patches, device=x.device)
        pos = torch.arange(x.size(1), device=x.device)
        pos_embed = self.pos_embed(pos).unsqueeze(0).expand(B, -1, -1)  # (B, num_patches, hidden_dim)
        x = x + pos_embed

        # Add class token
        cls_token = self.cls_token(x)  # (B, 1, hidden_dim)
        x = torch.cat([cls_token, x], dim=1)  # (B, num_patches + 1, hidden_dim)
        for layer in self.transformers:
            x = layer(x)

        x = self.norm(x)
        x = x[:, 0]  # class token
        
        return x

class ResNetBackbone(nn.Module):
    def __init__(self, cf):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if cf['pretrained'] else None)
        orig = resnet.conv1
        conv1 = nn.Conv2d(1, orig.out_channels, orig.kernel_size, orig.stride, orig.padding, bias=orig.bias is not None)
        with torch.no_grad(): conv1.weight[:] = orig.weight.mean(1, keepdim=True)
        resnet.conv1 = conv1
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feat_dim = resnet.fc.in_features
    def forward(self, x):
        x = self.backbone(x)
        return x.view(x.size(0), -1)

class ViTBackbone(nn.Module):
    def __init__(self, cf):
        super().__init__()
        # Load pretrained ViT and strip off its classification head
        vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if cf['pretrained'] else None)
        # replace classification head with identity to yield features
        hidden_dim = vit.heads.head.in_features
        vit.heads.head = nn.Identity()
        self.vit = vit
        self.feat_dim = hidden_dim

    def forward(self, x):
        # replicate single-channel if needed
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        # returns feature vector of shape (B, hidden_dim)
        return self.vit(x)

class MultiTaskModel(nn.Module):
    def __init__(self, backbone, feat_dim, cf):
        super().__init__()
        self.backbone = backbone
        self.cls_head = nn.Linear(feat_dim, cf['num_classes'])
        self.reg_head = nn.Linear(feat_dim, 1)
    def forward(self, x):
        feats = self.backbone(x)
        logits = self.cls_head(feats)
        age_pred = self.reg_head(feats).squeeze(1)
        return logits, age_pred

def build_model(cf):
    mt = cf['model_type']
    if mt == 'resnet':
        backbone = ResNetBackbone(cf)
    elif mt == 'vit':
        backbone = ViTBackbone(cf)
    elif mt == 'hybrid_rvit':
        backbone = ResNetViTBackbone(cf)
    else:
        raise ValueError(f"Unknown model_type '{mt}'")
    return MultiTaskModel(backbone, backbone.feat_dim, cf)
