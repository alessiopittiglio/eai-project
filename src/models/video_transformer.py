import torch
import torch.nn as nn


class PatchEmbedding3D(nn.Module):
    """
    Converts a video into a sequence of patch embeddings using 3D convolution (tubelets).
    """
    def __init__(self,
                 in_channels: int = 3,
                 embed_dim: int = 768,
                 tubelet_size: tuple = (2, 16, 16),  # (T, P, P)
                 dropout: float = 0.1):
        super(PatchEmbedding3D, self).__init__()
        t, p, _ = tubelet_size
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(t, p, p),
            stride=(t, p, p)
        )
        # number of patches = (T/t) * (H/p) * (W/p)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = None  # will be initialized in forward when input size known
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        # Apply tubelet embedding
        patches = self.proj(x)  # [B, E, T', H', W']
        E, T_p, H_p, W_p = patches.shape[1:]
        num_patches = T_p * H_p * W_p
        # Flatten spatial-temporal dimensions
        patches = patches.flatten(2).transpose(1, 2)  # [B, N, E]
        # Initialize positional embeddings if necessary
        if self.pos_embed is None or self.pos_embed.shape[1] != num_patches + 1:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, E))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # Expand class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, E]
        x = torch.cat((cls_tokens, patches), dim=1)  # [B, N+1, E]
        x = x + self.pos_embed
        return self.dropout(x)


class VideoTransformer(nn.Module):
    """
    Vision Transformer for video deepfake detection.
    """
    def __init__(self,
                 image_size: tuple = (224, 224),
                 num_frames: int = 16,
                 in_channels: int = 3,
                 patch_size: tuple = (2, 16, 16),
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 num_classes: int = 2):
        super(VideoTransformer, self).__init__()
        self.patch_embed = PatchEmbedding3D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            tubelet_size=(patch_size[0], patch_size[1], patch_size[2]),
            dropout=dropout
        )
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor):
        # x: [B, C, T, H, W]
        x = self.patch_embed(x)         # [B, N+1, E]
        # Transformer expects [S, B, E]
        x = x.transpose(0, 1)           # [N+1, B, E]
        x = self.encoder(x)             # [N+1, B, E]
        cls_token_final = x[0]          # [B, E]
        x = self.norm(cls_token_final)  # [B, E]
        logits = self.head(x)           # [B, num_classes]
        return logits


# Example usage
if __name__ == "__main__":
    model = VideoTransformer(
        image_size=(224, 224),
        num_frames=16,
        in_channels=3,
        patch_size=(2, 16, 16),
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        num_classes=2
    )
    # Dummy video batch: batch of 2 videos, 3 channels, 16 frames, 224x224
    dummy_video = torch.randn(2, 3, 16, 224, 224)
    logits = model(dummy_video)
    print("Output logits shape:", logits.shape)  # Expect [2, 2]
