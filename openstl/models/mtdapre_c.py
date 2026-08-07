import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_
from openstl.modules import TAttention, PreNorm, LDAttention, SwiGLU, TGDFN, SGDFN


class TGatedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, TAttention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                PreNorm(dim, TGDFN(dim, mlp_dim, drop=dropout)),
                DropPath(drop_path) if drop_path > 0. else nn.Identity(),
                DropPath(drop_path) if drop_path > 0. else nn.Identity()
            ]))

    def forward(self, x):
        for attn, ff, drop_path1, drop_path2 in self.layers:
            x = x + drop_path1(attn(x))
            x = x + drop_path2(ff(x))
        return self.norm(x)


class SGatedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, LDAttention(dim, heads=heads, dropout=attn_dropout)),
                PreNorm(dim, SwiGLU(dim, mlp_dim, drop=dropout)),
                DropPath(drop_path) if drop_path > 0. else nn.Identity(),
                DropPath(drop_path) if drop_path > 0. else nn.Identity()
            ]))

    def forward(self, x):
        for attn, ff, drop_path1, drop_path2 in self.layers:
            x = x + drop_path1(attn(x))
            x = x + drop_path2(ff(x))
        return self.norm(x)


class MTDAPreLayer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super(MTDAPreLayer, self).__init__()

        self.ts_temporal_transformer = TGatedTransformer(dim, depth, heads, dim_head,
                                                         mlp_dim, dropout, attn_dropout, drop_path)
        self.ts_space_transformer = SGatedTransformer(dim, depth, heads, dim_head,
                                                      mlp_dim, dropout, attn_dropout, drop_path)

    def forward(self, x):
        # x: [B, T, H', W', D]
        # ts-t branch
        x_ts = self.ts_temporal_transformer(x)
        # ts-s branch
        x_ts = self.ts_space_transformer(x_ts)
        return x_ts


def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')


class MTDAPre_Model(nn.Module):
    def __init__(self, model_config, **kwargs):
        super().__init__()
        self.image_height = model_config['height']
        self.image_width = model_config['width']
        self.patch_size = model_config['patch_size']
        self.num_patches_h = self.image_height // self.patch_size
        self.num_patches_w = self.image_width // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.num_frames_in = model_config['pre_seq']
        self.dim = model_config['dim']
        self.num_channels = model_config['num_channels']
        self.out_channels = model_config['out_channels']
        self.num_classes = self.out_channels
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.attn_dropout = model_config['attn_dropout']
        self.drop_path = model_config['drop_path']
        self.scale_dim = model_config['scale_dim']
        self.Ndepth = model_config['Ndepth']  # Ensure this is defined
        self.depth = model_config['depth']  # Ensure this is defined
        assert self.image_height % self.patch_size == 0, 'Image height must be divisible by the patch size.'
        assert self.image_width % self.patch_size == 0, 'Image width must be divisible by the patch size.'
        self.patch_dim = self.num_channels * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t h w (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_dim, self.dim),
        )

        self.pos_embedding = nn.Parameter(sinusoidal_embedding(self.num_frames_in * self.num_patches, self.dim),
                                          requires_grad=False).view(1, self.num_frames_in, self.num_patches_h, self.num_patches_w, self.dim)

        self.blocks = nn.ModuleList([
            MTDAPreLayer(self.dim, self.depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout,
                          self.attn_dropout, self.drop_path)
            for i in range(self.Ndepth)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.out_channels * self.patch_size ** 2)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # skip offset
        if getattr(m, '_no_reinit', False):
            return
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, T, C, H, W = x.shape

        # Patch Embedding   [B, T, C, H, W] -> [B, T, H', W', D]
        x = self.to_patch_embedding(x)

        # Position Embedding  [1, T, H', W', D]
        x += self.pos_embedding.to(x.device)

        # MTDAPre Encoder [B, T, H', W', D]
        for blk in self.blocks:
            x = blk(x)

        # MLP head  [B, T, H', W', D] -> [B, T, H', W', C*p*p]
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.view(B, T, self.num_patches_h, self.num_patches_w, self.out_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T, self.out_channels, H, W)

        return x