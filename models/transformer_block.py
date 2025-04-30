import torch
import torch.nn as nn

class TextureEnhancementTransformer(nn.Module):
    def __init__(self, in_channels, embed_dim=128, num_heads=4, num_layers=1):
        super().__init__()
        self.embedding = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, embed_dim // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 8, embed_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.project = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.embedding(x)
        x_seq = x.flatten(2).permute(2, 0, 1)
        x_seq = self.transformer(x_seq)
        x = x_seq.permute(1, 2, 0).view(B, -1, H, W)
        x = x * self.channel_attn(x)
        return self.project(x)
