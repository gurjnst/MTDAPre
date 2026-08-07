import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_
from openstl.modules import TAttention, PreNorm, LDAttention, SwiGLU, TGDFN, SGDFN


class AuxVariateEmbedding(nn.Module):
    def __init__(self, pre_seq, patch_size, dim):
        super().__init__()
        self.patch_dim = pre_seq * patch_size ** 2
        self.to_variate_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b c h w (t p1 p2)', p1=patch_size, p2=patch_size),
            nn.Linear(self.patch_dim, dim)
        )

    def forward(self, x):
        return self.to_variate_embedding(x)


class AuxSpatialEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
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


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=16, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, glb, aux):
        B, _, H, W, D = glb.shape

        q = self.norm_q(glb)
        aux = self.norm_kv(aux)

        q = rearrange(q, 'b n h w d -> (b h w) n d')
        aux = rearrange(aux, 'b c h w d -> (b h w) c d')

        q = self.to_q(q)
        k = self.to_k(aux)
        v = self.to_v(aux)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        attn = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h 1 d -> b (h d)')
        out = self.to_out(out)
        out = rearrange(out, '(b h w) d -> b 1 h w d', b=B, h=H, w=W)

        return out


class TGatedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, TAttention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                PreNorm(dim, TGDFN(dim, mlp_dim, drop=dropout)),
                DropPath(drop_path) if drop_path > 0. else nn.Identity(),
                DropPath(drop_path) if drop_path > 0. else nn.Identity(),
                DropPath(drop_path) if drop_path > 0. else nn.Identity()
            ]))

    def forward(self, x, glb, aux):
        x_all = torch.cat([x, glb], dim=1)
        for attn, cross, ff, drop_path1, drop_path2, drop_path3 in self.layers:
            x_all = x_all + drop_path1(attn(x_all))
            glb = x_all[:, -1:]
            glb = glb + drop_path2(cross(glb, aux))
            x_all = torch.cat([x_all[:, :-1], glb], dim=1)
            x_all = x_all + drop_path3(ff(x_all))
        x_all = self.norm(x_all)
        return x_all[:, :-1], x_all[:, -1:]


class SGatedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, LDAttention(dim, heads=heads, dropout=attn_dropout)),
                CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                PreNorm(dim, SwiGLU(dim, mlp_dim, drop=dropout)),
                DropPath(drop_path) if drop_path > 0. else nn.Identity(),
                DropPath(drop_path) if drop_path > 0. else nn.Identity(),
                DropPath(drop_path) if drop_path > 0. else nn.Identity()
            ]))

    def forward(self, x, glb, aux):
        x_all = torch.cat([x, glb], dim=1)
        for attn, cross, ff, drop_path1, drop_path2, drop_path3 in self.layers:
            x_all = x_all + drop_path1(attn(x_all))
            glb = x_all[:, -1:]
            glb = glb + drop_path2(cross(glb, aux))
            x_all = torch.cat([x_all[:, :-1], glb], dim=1)
            x_all = x_all + drop_path3(ff(x_all))
        x_all = self.norm(x_all)
        return x_all[:, :-1], x_all[:, -1:]


class MTDAPreLayer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super(MTDAPreLayer, self).__init__()

        self.ts_temporal_transformer = TGatedTransformer(dim, depth, heads, dim_head,
                                                                mlp_dim, dropout, attn_dropout, drop_path)
        self.ts_space_transformer = SGatedTransformer(dim, depth, heads, dim_head,
                                                             mlp_dim, dropout, attn_dropout, drop_path)

    def forward(self, x, glb, aux):
        x, glb = self.ts_temporal_transformer(x, glb, aux)
        x, glb = self.ts_space_transformer(x, glb, aux)
        return x, glb


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
        self.Ndepth = model_config['Ndepth']
        self.depth = model_config['depth']
        self.auxdepth = model_config.get('auxdepth', 4)
        assert self.image_height % self.patch_size == 0, 'Image height must be divisible by the patch size.'
        assert self.image_width % self.patch_size == 0, 'Image width must be divisible by the patch size.'
        assert self.num_channels > 1, 'CrossAttention requires at least one auxiliary variable.'

        self.patch_dim = self.patch_size ** 2
        self.to_main_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t h w (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_dim, self.dim, bias=False)
        )
        self.to_aux_embedding = AuxVariateEmbedding(self.num_frames_in, self.patch_size, self.dim)

        self.pos_embedding = nn.Parameter(sinusoidal_embedding(self.num_frames_in * self.num_patches, self.dim),
                                          requires_grad=False).view(1, self.num_frames_in,
                                                                    self.num_patches_h, self.num_patches_w, self.dim)
        self.spatial_pos_embedding = nn.Parameter(sinusoidal_embedding(self.num_patches, self.dim),
                                                  requires_grad=False).view(1, 1,
                                                                            self.num_patches_h, self.num_patches_w,
                                                                            self.dim)
        self.glb_token = nn.Parameter(torch.randn(1, 1, 1, 1, self.dim))

        self.aux_spatial_encoder = AuxSpatialEncoder(dim=self.dim, depth=self.auxdepth, heads=self.heads, mlp_dim=self.dim * self.scale_dim,
            dropout=self.dropout, attn_dropout=self.attn_dropout, drop_path=self.drop_path)

        self.blocks = nn.ModuleList([
            MTDAPreLayer(self.dim, self.depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout,
                           self.attn_dropout, self.drop_path)
            for _ in range(self.Ndepth)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm((self.num_frames_in + 1) * self.dim),
            nn.Linear((self.num_frames_in + 1) * self.dim,
                      self.num_frames_in * self.out_channels * self.patch_size ** 2)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
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
        assert T == self.num_frames_in

        pos = self.pos_embedding.to(x.device)
        s_pos = self.spatial_pos_embedding.to(x.device)

        x_main = self.to_main_patch_embedding(x[:, :, 0:1])
        x_main = x_main + pos

        x_aux = self.to_aux_embedding(x[:, :, 1:])
        x_aux = x_aux + s_pos.squeeze(1).unsqueeze(1)
        x_aux = self.aux_spatial_encoder(x_aux)

        glb = self.glb_token.repeat(B, 1, self.num_patches_h, self.num_patches_w, 1)
        glb = glb + s_pos

        for blk in self.blocks:
            x_main, glb = blk(x_main, glb, x_aux)

        x_all = torch.cat([x_main, glb], dim=1)
        x = rearrange(x_all, 'b t h w d -> b h w (t d)')
        x = self.head(x.reshape(-1, (self.num_frames_in + 1) * self.dim))
        x = x.view(B, self.num_patches_h, self.num_patches_w,
                   self.num_frames_in, self.out_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 4, 1, 5, 2, 6).reshape(B, self.num_frames_in, self.out_channels, H, W)

        return x
