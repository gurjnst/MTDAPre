from torch import nn, einsum
from einops import rearrange
from timm.models.layers import to_2tuple
import torch
import torch.nn.functional as F
import einops

try:
    from natten.functional import na2d
except ImportError as e:
    raise ImportError(
        "LocalDeformableAttention requires natten.functional.na2d."
    ) from e

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class LocalDeformableAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, kernel_size: int, dilation: int = 1, offset_range_factor=1.0, stride=1,
                no_off=False, is_causal: bool = False, proj_drop: float = 0.0,):
        super().__init__()
        n_head_channels = dim // num_heads
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.nc = n_head_channels * num_heads
        self.n_groups = num_heads
        self.n_group_channels = self.nc // self.n_groups
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = kernel_size
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.dilation = dilation
        self.is_causal = is_causal
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels,
                      kk, stride, pad_size, groups=self.n_group_channels),
            LayerNorm2d(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        nn.init.constant_(self.conv_offset[-1].weight, 0.)
        self.conv_offset[-1]._no_reinit = True
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        # Dual-scale offset for irregular structures such as radar data.
        # self.conv_offset = nn.ModuleDict({
        #     'branch3': nn.Sequential(
        #         nn.Conv2d(self.n_group_channels, self.n_group_channels, 3, stride, 1, groups=self.n_group_channels),
        #         LayerNorm2d(self.n_group_channels),
        #         nn.GELU()
        #     ),
        #     'branch7': nn.Sequential(
        #         nn.Conv2d(self.n_group_channels, self.n_group_channels, 7, stride, 3, groups=self.n_group_channels),
        #         LayerNorm2d(self.n_group_channels),
        #         nn.GELU()
        #     ),
        #     'out_conv': nn.Conv2d(self.n_group_channels * 2, 2, 1, 1, 0, bias=False)
        # })
        # nn.init.constant_(self.conv_offset['out_conv'].weight, 0.)
        # self.conv_offset['out_conv']._no_reinit = True

        self.proj_q = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

        self.rpe_table = nn.Conv2d(
            self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key,
                           dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key,
                           dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(
            B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(
            q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
        # br3, br7, out = self.conv_offset['branch3'], self.conv_offset['branch7'], self.conv_offset['out_conv']
        # offset = out(torch.cat([br3(q_off), br7(q_off)], dim=1)).contiguous()

        Hk, Wk = offset.size(2), offset.size(3)

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor(
                [1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)

        if self.no_off:
            x_sampled = F.avg_pool2d(
                x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(
                3) == Wk, f"Size is {x_sampled.size()}"
        else:
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups,
                                self.n_group_channels, H, W),
                grid=pos[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, H, W)

        residual_lepe = self.rpe_table(q)

        q = einops.rearrange(q, 'b (g c) h w -> b h w g c',
                             g=self.n_groups, b=B, c=self.n_group_channels, h=H, w=W)
        k = einops.rearrange(self.proj_k(x_sampled), 'b (g c) h w -> b h w g c',
                             g=self.n_groups, b=B, c=self.n_group_channels, h=H, w=W)
        v = einops.rearrange(self.proj_v(x_sampled), 'b (g c) h w -> b h w g c',
                             g=self.n_groups, b=B, c=self.n_group_channels, h=H, w=W)
        out = na2d(q, k, v, kernel_size=self.kernel_size, dilation=self.dilation, is_causal=self.is_causal, scale=self.scale)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        out = out + residual_lepe

        y = self.proj_drop(self.proj_out(out))

        return y


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        B, T, H, W, D = x.shape
        x = rearrange(x, 'b t h w d -> (b h w) t d')
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = rearrange(out, '(b h w) t d -> b t h w d', b=B, h=H, w=W)
        return out


class LDAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0., kernel_size=7, dilation=1, offset_range_factor=1.0, stride=1, no_off=False, is_causal=False):
        super().__init__()
        self.attn = LocalDeformableAttention(
            dim=dim,
            num_heads=heads,
            kernel_size=kernel_size,
            dilation=dilation,
            offset_range_factor=offset_range_factor,
            stride=stride,
            no_off=no_off,
            is_causal=is_causal,
            proj_drop=dropout,
        )

    def forward(self, x):
        # x: [B, T, H, W, D]
        b, t, h, w, c = x.shape
        x_2d = x.reshape(b * t, h, w, c).permute(0, 3, 1, 2).contiguous()
        out = self.attn(x_2d)
        out = out.permute(0, 2, 3, 1).contiguous().reshape(b, t, h, w, c)

        return out


class SwiGLU(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TGDFN(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features * 2, bias=bias[0])
        self.conv = nn.Conv1d(hidden_features * 2, hidden_features * 2, 3, 1, 1, groups=hidden_features * 2, bias=False)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        # x: [B, T, H, W, D]
        B, T, H, W, D = x.shape
        x = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D)
        x = self.fc1(x)
        x = self.conv(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x1, x2 = x.chunk(2, dim=-1)
        x = self.act(x1) * x2
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = x.reshape(B, H, W, T, D).permute(0, 3, 1, 2, 4).contiguous()
        return x


class SGDFN(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        self.fc1 = nn.Linear(in_features, hidden_features * 2, bias=bias[0])
        self.conv = nn.Conv2d(hidden_features * 2, hidden_features * 2, 3, 1, 1, groups=hidden_features * 2, bias=False)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        # x: [B, T, H, W, D]
        B, T, H, W, D = x.shape
        x = x.reshape(B * T, H, W, D)
        x = self.fc1(x)
        x = self.conv(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x1, x2 = x.chunk(2, dim=-1)
        x = self.act(x1) * x2
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = x.reshape(B, T, H, W, D)
        return x