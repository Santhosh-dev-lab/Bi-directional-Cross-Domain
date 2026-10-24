import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SemanticTokenizer(nn.Module):
    """
    Spatial-Spectral Semantic Tokenizer (Default in BiDA).
    Uses 3D Conv + 2D Conv followed by spatial-spectral attention projection
    to extract L semantic tokens from hyperspectral patches.
    """
    def __init__(self, in_channels=1, n_bands=48, dim=64, num_tokens=4, patch_size=13):
        super(SemanticTokenizer, self).__init__()
        self.L = num_tokens
        self.cT = dim
        self.token_len = num_tokens

        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8 * n_bands, out_channels=dim,
                      kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )

        self.conv_a = nn.Conv2d(dim, self.token_len, kernel_size=1,
                                padding=0, bias=False)

    def forward(self, x):
        # x: (B, 1, n_bands, H, W) or (B, n_bands, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)

        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens


class SpatialPatchTokenizer(nn.Module):
    """
    Standard Spatial Patch Tokenizer (ViT-style).
    Flattens 2D spatial patches across all spectral channels and linearly projects
    each patch to the embedding dimension.
    """
    def __init__(self, in_channels=1, n_bands=48, dim=64, num_tokens=4, patch_size=13):
        super(SpatialPatchTokenizer, self).__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.grid_h = int(round(num_tokens ** 0.5))
        self.grid_w = num_tokens // self.grid_h
        if self.grid_h * self.grid_w != num_tokens:
            self.grid_h = num_tokens
            self.grid_w = 1

        self.proj = nn.Sequential(
            nn.Conv2d(n_bands, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.grid_h, self.grid_w))
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(1)
        # x is (B, n_bands, H, W)
        feat = self.proj(x)  # (B, dim, grid_h, grid_w)
        tokens = rearrange(feat, 'b c h w -> b (h w) c')  # (B, L, dim)
        tokens = self.mlp(tokens)
        return tokens


class BandGroupingTokenizer(nn.Module):
    """
    Band-Grouping Spectral Tokenizer.
    Splits continuous spectral channels into contiguous sub-band groups and extracts
    spatial-spectral token representations for each group.
    """
    def __init__(self, in_channels=1, n_bands=48, dim=64, num_tokens=4, patch_size=13):
        super(BandGroupingTokenizer, self).__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.n_bands = n_bands

        # Calculate contiguous sub-band splits
        band_splits = []
        base_size = n_bands // num_tokens
        rem = n_bands % num_tokens
        start = 0
        for i in range(num_tokens):
            size = base_size + (1 if i < rem else 0)
            end = start + size
            band_splits.append((start, end))
            start = end
        self.band_splits = band_splits

        # Encoder for each sub-band group
        self.group_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(end - start, dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim)
            ) for start, end in band_splits
        ])

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(1)
        # x is (B, n_bands, H, W)
        tokens = []
        for i, (start, end) in enumerate(self.band_splits):
            group_x = x[:, start:end, :, :]
            tok = self.group_encoders[i](group_x)  # (B, dim)
            tokens.append(tok.unsqueeze(1))
        tokens = torch.cat(tokens, dim=1)  # (B, num_tokens, dim)
        return tokens


class Cube3DTokenizer(nn.Module):
    """
    3D-Cube Tokenizer.
    Extracts spatial-spectral sub-cubes via strided 3D convolutions.
    """
    def __init__(self, in_channels=1, n_bands=48, dim=64, num_tokens=4, patch_size=13):
        super(Cube3DTokenizer, self).__init__()
        self.num_tokens = num_tokens
        self.dim = dim

        self.conv3d_stage1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=(5, 3, 3), stride=(2, 2, 2), padding=(2, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.conv3d_stage2 = nn.Sequential(
            nn.Conv3d(16, dim, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool3d((num_tokens, 1, 1))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        # x: (B, 1, n_bands, H, W)
        feat = self.conv3d_stage1(x)
        feat = self.conv3d_stage2(feat)
        feat = self.pool(feat)  # (B, dim, num_tokens, 1, 1)
        feat = feat.squeeze(-1).squeeze(-1)  # (B, dim, num_tokens)
        tokens = rearrange(feat, 'b c l -> b l c')  # (B, num_tokens, dim)
        tokens = self.norm(tokens)
        return tokens


def get_tokenizer(tokenizer_type, in_channels=1, n_bands=48, dim=64, num_tokens=4, patch_size=13):
    """
    Factory function to instantiate tokenizer by type.
    Supported types: 'semantic', 'patch', 'band_grouping', '3d_cube'
    """
    tokenizer_type = tokenizer_type.lower()
    if tokenizer_type == 'semantic':
        return SemanticTokenizer(in_channels=in_channels, n_bands=n_bands, dim=dim,
                                 num_tokens=num_tokens, patch_size=patch_size)
    elif tokenizer_type == 'patch':
        return SpatialPatchTokenizer(in_channels=in_channels, n_bands=n_bands, dim=dim,
                                    num_tokens=num_tokens, patch_size=patch_size)
    elif tokenizer_type == 'band_grouping':
        return BandGroupingTokenizer(in_channels=in_channels, n_bands=n_bands, dim=dim,
                                    num_tokens=num_tokens, patch_size=patch_size)
    elif tokenizer_type in ('3d_cube', '3dcube', 'cube'):
        return Cube3DTokenizer(in_channels=in_channels, n_bands=n_bands, dim=dim,
                               num_tokens=num_tokens, patch_size=patch_size)
    else:
        raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}. "
                         f"Choose from ['semantic', 'patch', 'band_grouping', '3d_cube']")
