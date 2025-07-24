import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights
from torchvision import models


class ClassToken(nn.Module):
    def __init__(self, hidden_dim):
        super(ClassToken, self).__init__()
        self.token = nn.Parameter(torch.randn(1, 1, hidden_dim))  # (1, 1, hidden_dim)

    def forward(self, x):
        B = x.size(0)
        return self.token.expand(B, -1, -1)  # (B, 1, hidden_dim)


class MLP(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, dropout_rate):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout1(self.act(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, num_heads, dropout_rate):
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_dim, dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x_attn = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.dropout1(x_attn)
        x_mlp = self.mlp(self.norm2(x))
        x = x + self.dropout2(x_mlp)
        return x


class ResNetViT(nn.Module):
    def __init__(self, cf):
        super(ResNetViT, self).__init__()
        self.cf = cf


        if cf.pretrained:
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

        input_dim_patch_proj = 512
        self.patch_proj = nn.Conv2d(input_dim_patch_proj, cf.hidden_dim, kernel_size=cf.patch_size, padding="same")
        self.bn = nn.BatchNorm2d(cf.hidden_dim)

        # Positional Embedding
        self.pos_embed = nn.Embedding(289, cf.hidden_dim)
        # self.pos_embed = None

        # Class token
        self.cls_token = ClassToken(cf.hidden_dim)

        # Transformer layers
        self.transformers = nn.ModuleList([
            TransformerEncoder(
                hidden_dim=cf.hidden_dim,
                mlp_dim=cf.mlp_dim,
                num_heads=cf.num_heads,
                dropout_rate=cf.dropout_rate
            ) for _ in range(cf.num_layers)
        ])

        self.norm = nn.LayerNorm(cf.hidden_dim)
        self.head = nn.Linear(cf.hidden_dim, cf.num_classes)

        self.final_dropout = nn.Dropout(cf.dropout_rate)


    def forward(self, x):
        B = x.size(0)
        x = self.backbone(x)  # (B, 2048, 4, 4) for 128x128 input
        x = self.patch_proj(x)  # (B, hidden_dim, 4, 4)
        x = self.bn(x)

        # Flatten patches
        x = x.flatten(2).transpose(1, 2)  # (B, 16, hidden_dim)
        if self.pos_embed is None:
            self.pos_embed = nn.Embedding(x.size(1), self.cf.hidden_dim).to(x.device)
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
        x = self.final_dropout(x)  # (B, hidden_dim)
        logits = self.head(x)


        return logits


    def forward_features(self, x):
        B = x.size(0)
        x = self.backbone(x)  # (B, 2048, 4, 4) for 128x128 input
        x = self.patch_proj(x)  # (B, hidden_dim, 4, 4)
        x = self.bn(x)

        # Flatten patches
        x = x.flatten(2).transpose(1, 2)  # (B, 16, hidden_dim)
        if self.pos_embed is None:
            self.pos_embed = nn.Embedding(x.size(1), self.cf.hidden_dim).to(x.device)
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


class ViTClassifier(nn.Module):
    def __init__(self, cf):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if cf.pretrained == False else None
        num_classes = cf.num_classes
        dropout_rate = cf.dropout_rate

        self.vit = models.vit_b_16(weights=weights)
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.vit(x)


class ResNetClassifier(nn.Module):
    def __init__(self, cf):
        super().__init__()
        pretrained = cf.pretrained
        num_classes = cf.num_classes
        dropout_rate = cf.dropout_rate
        resnet = models.resnet18(pretrained=pretrained)
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

        # Replace the original conv1 layer
        resnet.conv1 = new_conv

        resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(resnet.fc.in_features, num_classes)
        )

        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)

    def forward_features(self, x):
        # Extract features before the final classification layer
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        return x


def build_model(cf):
    if cf.model_type == "resnet":
        model = ResNetClassifier(cf)
    elif cf.model_type == "vit":
        model = ViTClassifier(cf)
    elif cf.model_type == "hybrid_rvit":
        model = ResNetViT(cf)
    else:
        raise ValueError("Invalid model type. Choose 'resnet', 'vit', or 'hybrid_rvit'.")

    return model