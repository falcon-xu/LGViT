import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath

from configuration_swin import SwinConfig
from models.deit_highway import DeiTConfig, DeiTLayer

class SwinHighway(nn.Module):
    def __init__(self, config: SwinConfig, stage: int) -> None:
        super(SwinHighway, self).__init__()

        self.num_features = int(config.embed_dim * 2 ** (stage - 1))

        self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.num_features, config.num_labels) if config.num_labels > 0 else nn.Identity()

    def forward(self, encoder_outputs):
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output.transpose(1, 2))
        pooled_output = torch.flatten(pooled_output, 1)
        logits = self.classifier(pooled_output)

        return logits, pooled_output


class ViT_EE_Highway(nn.Module):
    def __init__(self, config: SwinConfig, stage: int) -> None:
        super(ViT_EE_Highway, self).__init__()
        deit_config = DeiTConfig()
        deit_config.num_attention_heads = 8
        deit_config.hidden_size = 512
        self.layer = DeiTLayer(deit_config)
        self.classifier = SwinHighway(config, 3)

    def forward(self, encoder_outputs):
        sequence_output = encoder_outputs[0]
        layer_output = self.layer(sequence_output)
        logits, pooled_output = self.classifier(layer_output)

        return logits, pooled_output

# Local perception head
class highway_conv1_1(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
        )

        self.proj = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.proj_act = nn.GELU()
        self.proj_bn = nn.BatchNorm2d(hidden_features, eps=1e-5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_features, eps=1e-5),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x0 = x
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj(x)
        x = self.proj_act(x)
        x = self.proj_bn(x)
        x = self.conv2(x)
        x = x.flatten(2).permute(0, 2, 1) + x0
        x = self.drop(x)
        return x

# Local perception head
class highway_conv2_1(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_features, eps=1e-5),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x0 = x
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = x.flatten(2).permute(0, 2, 1) + x0
        x = self.drop(x)
        return x

# Global aggregation head
class GlobalSparseAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.upsample = nn.Upsample(scale_factor=sr_ratio, mode='nearest')
        self.sr = sr_ratio
        if self.sr > 1:
            self.sampler = nn.AvgPool2d(1, sr_ratio)
            # kernel_size = sr_ratio
            # self.LocalProp= nn.ConvTranspose2d(dim, dim, kernel_size, stride=sr_ratio, groups=dim)
            # self.norm = nn.LayerNorm(dim)
        else:
            self.sampler = nn.Identity()
            self.upsample = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        if self.sr > 1.:
            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.sampler(x)
            x = x.flatten(2).transpose(1, 2)

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        # if self.sr > 1:
        #     x = x.permute(0, 2, 1).reshape(B, C, int(H/self.sr), int(W/self.sr))
        #     x = self.LocalProp(x)
        #     x = x.reshape(B, C, -1).permute(0, 2, 1)
        #     x = self.norm(x)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


highway_classes = {
    "conv1_1": highway_conv1_1,
    "conv2_1": highway_conv2_1,
    "attention": GlobalSparseAttn,
}


class SwinHighway_v2(nn.Module):
    def __init__(self, config: SwinConfig, stage: int, highway_type) -> None:
        super(SwinHighway_v2, self).__init__()

        self.config = config
        self.highway_type = highway_type
        self.num_features = int(config.embed_dim * 2 ** (stage - 1))

        if "attention" in highway_type:
            sr_ratio = eval(highway_type[-1])
            self.mlp = GlobalSparseAttn(self.num_features, sr_ratio=sr_ratio)
        else:
            self.mlp = highway_classes[highway_type](self.num_features)
        self.classifier = SwinHighway(config, stage)

    def forward(self, encoder_outputs):
        sequence_output = encoder_outputs[0]
        mlp_output = self.mlp(sequence_output)
        logits, pooled_output = self.classifier((mlp_output,))

        return logits, pooled_output    
