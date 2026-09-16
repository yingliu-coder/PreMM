import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def _make_group_norm(channels: int, max_groups: int = 32):
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)

def _trunc_normal_(tensor, std=0.02):
    return nn.init.trunc_normal_(tensor, std=std)

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rnd = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        rnd.floor_()
        return x.div(keep) * rnd

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x):
        return x * self.gamma

class ResBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        expansion: int = 1,
        kernel_size: int = 3,
        padding_mode: str = 'zeros',
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden = out_channels * expansion
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size, padding=pad, padding_mode=padding_mode)
        self.norm1 = _make_group_norm(hidden)
        self.conv2 = nn.Conv2d(hidden, out_channels, kernel_size, padding=pad, padding_mode=padding_mode)
        self.norm2 = _make_group_norm(out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.gelu(h + self.skip(x))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int((2.0 / 3.0) * mlp_ratio * dim)
        hidden = int(math.ceil(hidden / 64.0) * 64)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, g = self.fc1(x).chunk(2, dim=-1)
        return self.drop(self.fc2(a * F.silu(g)))

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = float(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        n, l, d = x.shape
        qkv = self.qkv(x).reshape(n, l, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).reshape(n, l, d)
        return self.proj_drop(self.proj(y))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = float(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, qx, kvx):
        n, lq, d = qx.shape
        lk = kvx.shape[1]
        q = self.q(qx).reshape(n, lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(kvx).reshape(n, lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(kvx).reshape(n, lk, self.num_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).reshape(n, lq, d)
        return self.proj_drop(self.proj(y))

class ComplexTemporalCell(nn.Module):
    def __init__(self, t_in: int, t_out: int, channels: int, size_factor: float = 1.0):
        super().__init__()
        hidden_t = max(4, int(t_out * size_factor))
        self.t_in = t_in
        self.t_out = t_out
        self.channels = channels
        self.bias_time = nn.Sequential(
            nn.Linear(t_in, hidden_t),
            nn.SELU(True),
            nn.Linear(hidden_t, t_out),
        )
        scale = 0.02
        self.w1 = nn.Parameter(scale * torch.randn(2, t_in, hidden_t))
        self.b1 = nn.Parameter(scale * torch.randn(2, 1, 1, 1, hidden_t))
        self.w2 = nn.Parameter(scale * torch.randn(2, hidden_t, t_out))
        self.b2 = nn.Parameter(scale * torch.randn(2, 1, 1, 1, t_out))

    def forward(self, x):
        xp = x.permute(0, 2, 3, 4, 1)
        bias = self.bias_time(xp)
        xf = torch.fft.rfft2(xp.float(), dim=(2, 3), norm='ortho')

        r1 = torch.einsum('bchwt,to->bchwo', xf.real, self.w1[0]) \
             - torch.einsum('bchwt,to->bchwo', xf.imag, self.w1[1]) + self.b1[0]
        i1 = torch.einsum('bchwt,to->bchwo', xf.real, self.w1[1]) \
             + torch.einsum('bchwt,to->bchwo', xf.imag, self.w1[0]) + self.b1[1]
        r1, i1 = F.gelu(r1), F.gelu(i1)

        r2 = torch.einsum('bchwt,to->bchwo', r1, self.w2[0]) \
             - torch.einsum('bchwt,to->bchwo', i1, self.w2[1]) + self.b2[0]
        i2 = torch.einsum('bchwt,to->bchwo', r1, self.w2[1]) \
             + torch.einsum('bchwt,to->bchwo', i1, self.w2[0]) + self.b2[1]
        yc = torch.complex(r2, i2)
        y = torch.fft.irfft2(yc, s=(x.shape[-2], x.shape[-1]), dim=(2, 3), norm='ortho')
        y = y + bias.float()
        return y.permute(0, 4, 1, 2, 3).to(x.dtype)


class ComplexSpectralBlock(nn.Module):
    def __init__(self, t_in: int, t_out: int, channels: int):
        super().__init__()
        self.t_out = t_out
        self.channels = channels
        self.temporal = ComplexTemporalCell(t_in, t_out, channels, size_factor=1.0)
        tc = channels * t_out
        groups = min(32, tc)
        while tc % groups != 0:
            groups -= 1
        self.spatial = nn.Sequential(
            nn.Conv2d(tc, tc, 3, padding=1),
            nn.GroupNorm(groups, tc),
            nn.SiLU(),
            nn.Conv2d(tc, tc, 3, padding=1),
        )

    def forward(self, x):
        y = self.temporal(x)
        B, T, C, H, W = y.shape
        flat = y.reshape(B, T * C, H, W)
        flat = flat + self.spatial(flat)
        return flat.reshape(B, T, C, H, W)


class ComplexSpectralForecastPath(nn.Module):
    def __init__(self, t_in: int, t_out: int, input_dim: int, hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.t_in = t_in
        self.t_out = t_out
        self.encoder = nn.Sequential(
            ResBlock2D(input_dim, hidden_dim),
            ResBlock2D(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        )
        self.cells = nn.ModuleList([
            ComplexSpectralBlock(t_in if i == 0 else t_out, t_out, hidden_dim)
            for i in range(n_layers)
        ])
        self.decoder = nn.Sequential(
            ResBlock2D(hidden_dim),
            ResBlock2D(hidden_dim),
            nn.Conv2d(hidden_dim, input_dim, 1),
        )
        self.direct_time = nn.Sequential(
            nn.Linear(t_in, t_out * 2),
            nn.SELU(True),
            nn.Linear(t_out * 2, t_out),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        h = self.encoder(x.reshape(B * T, C, H, W)).reshape(B, T, -1, H, W)
        direct = self.direct_time(h.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        for cell in self.cells:
            h = cell(h)
        h = h + direct
        D = h.shape[2]
        out = self.decoder(h.reshape(B * self.t_out, D, H, W)).reshape(B, self.t_out, C, H, W)
        return out


class PhaseDynamicsPath(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], t_in: int, t_out: int, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        H, W = input_shape
        Wf = W // 2 + 1
        self.t_out = t_out
        self.input_dim = input_dim
        in_ch = t_in * input_dim + 2
        out_ch = t_out * input_dim
        self.base = nn.Conv2d(in_ch, out_ch, 1)
        self.branch0 = nn.Sequential(
            ResBlock2D(in_ch, hidden_dim, kernel_size=1),
            ResBlock2D(hidden_dim, kernel_size=1),
            nn.Conv2d(hidden_dim, out_ch, 1),
        )
        self.branch1 = nn.Sequential(
            ResBlock2D(in_ch, hidden_dim, kernel_size=1),
            ResBlock2D(hidden_dim, kernel_size=1),
            nn.Conv2d(hidden_dim, out_ch, 1),
        )
        self.branch2 = nn.Sequential(
            ResBlock2D(in_ch, hidden_dim, kernel_size=3, padding_mode='circular'),
            ResBlock2D(hidden_dim, kernel_size=3, padding_mode='circular'),
            nn.Conv2d(hidden_dim, out_ch, 1),
        )
        self.mix = nn.Conv2d(out_ch * 4, out_ch, 1)
        u = torch.fft.fftfreq(H)
        v = torch.fft.rfftfreq(W)
        uu, vv = torch.meshgrid(u, v, indexing='ij')
        self.register_buffer('uv', torch.stack([uu, vv], dim=0), persistent=False)

    def forward(self, x):
        B, T, C, H, W = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = torch.fft.rfft2(x.float(), dim=(-2, -1), norm='ortho')
            amp = torch.abs(xf)
            phase = torch.angle(xf) / math.pi
            phase_flat = phase.reshape(B, T * C, H, W // 2 + 1)
            puv = torch.cat([phase_flat, self.uv.to(x.device).unsqueeze(0).expand(B, -1, -1, -1)], dim=1)
            p0 = self.base(puv)
            p1 = p0 + self.branch0(puv)
            p2 = p0 * torch.tanh(self.branch1(puv))
            p3 = p0 * torch.tanh(self.branch2(puv))
            delta = self.mix(torch.cat([p0, p1, p2, p3], dim=1))
            delta = delta.reshape(B, self.t_out, C, H, W // 2 + 1)
            future_phase = phase[:, -1:] * math.pi + delta * math.pi
            last_amp = amp[:, -1:].expand(-1, self.t_out, -1, -1, -1)
            phase_fft = torch.polar(last_amp, future_phase)
            phase_image = torch.fft.irfft2(phase_fft, s=(H, W), dim=(-2, -1), norm='ortho')
        return phase_image.to(x.dtype), future_phase, amp


class AlphaInspiredSpectralPrior(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        t_in: int,
        t_out: int,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        spec_num: int = 20,
    ):
        super().__init__()
        self.t_out = t_out
        self.input_dim = input_dim
        self.content_path = ComplexSpectralForecastPath(t_in, t_out, input_dim, hidden_dim, n_layers)
        self.phase_path = PhaseDynamicsPath(input_shape, t_in, t_out, input_dim, hidden_dim)
        self.out_mixer = nn.Sequential(
            ResBlock2D(input_dim * 3, hidden_dim),
            ResBlock2D(hidden_dim),
            nn.Conv2d(hidden_dim, input_dim, 1),
        )
        H, W = input_shape
        mask = torch.zeros(H, W // 2 + 1)
        k = min(spec_num, H // 2, W // 2)
        mask[:k, :k] = 1.0
        mask[-k:, :k] = 1.0
        self.register_buffer('spec_mask', mask, persistent=False)

    def forward(self, x):
        content = self.content_path(x)
        phase_img, future_phase, input_amp = self.phase_path(x)
        B, T, C, H, W = content.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            cf = torch.fft.rfft2(content.float(), dim=(-2, -1), norm='ortho')
            amp = torch.abs(cf)
            alpha_fft = amp * self.spec_mask.view(1, 1, 1, H, W // 2 + 1) * torch.exp(1j * future_phase)
            alpha = torch.fft.irfft2(alpha_fft, s=(H, W), dim=(-2, -1), norm='ortho').to(x.dtype)
        mix = torch.cat([content, phase_img, alpha], dim=2).reshape(B * T, 3 * C, H, W)
        prior = self.out_mixer(mix).reshape(B, T, C, H, W)
        return prior, {
            'content_forecast': content,
            'phase_forecast': phase_img,
            'alpha_recomposition': alpha,
            'future_phase': future_phase,
            'input_amp': input_amp,
        }

class RadarPatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, dim: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.norm = _make_group_norm(dim)
        self.refine = ResBlock2D(dim)

    def forward(self, x):
        B, T, C, H, W = x.shape
        y = F.gelu(self.norm(self.proj(x.reshape(B * T, C, H, W))))
        y = self.refine(y)
        _, D, h, w = y.shape
        return y.reshape(B, T, D, h, w)


class LatentDownsample(nn.Module):
    def __init__(self, dim: int, stride: int = 2):
        super().__init__()
        self.stride = stride
        self.net = nn.Identity() if stride == 1 else nn.Sequential(
            _make_group_norm(dim),
            nn.Conv2d(dim, dim, 3, stride=stride, padding=1),
            nn.GELU(),
            ResBlock2D(dim),
        )

    def forward(self, x):
        if self.stride == 1:
            return x
        B, T, D, H, W = x.shape
        y = self.net(x.reshape(B * T, D, H, W))
        h, w = y.shape[-2:]
        return y.reshape(B, T, D, h, w)


class LatentUpsample(nn.Module):
    def __init__(self, dim: int, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.net = nn.Identity() if scale == 1 else nn.Sequential(
            nn.Conv2d(dim, dim * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.GELU(),
            ResBlock2D(dim),
        )

    def forward(self, x):
        if self.scale == 1:
            return x
        B, T, D, H, W = x.shape
        y = self.net(x.reshape(B * T, D, H, W))
        h, w = y.shape[-2:]
        return y.reshape(B, T, D, h, w)


class TrajectoryInitializer(nn.Module):
    def __init__(self, dim: int, t_out: int, max_step_disp: float = 2.0):
        super().__init__()
        self.t_out = t_out
        self.max_step_disp = float(max_step_disp)
        self.velocity = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 3, padding=1),
            _make_group_norm(dim),
            nn.GELU(),
            ResBlock2D(dim),
            nn.Conv2d(dim, 2, 3, padding=1),
        )
        self.step_offset = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1),
            _make_group_norm(dim),
            nn.GELU(),
            nn.Conv2d(dim, 2, 3, padding=1),
        )
        self.confidence = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.velocity[-1].weight)
        nn.init.zeros_(self.velocity[-1].bias)
        nn.init.zeros_(self.step_offset[-1].weight)
        nn.init.zeros_(self.step_offset[-1].bias)

    @staticmethod
    def _base_grid(B, H, W, device, dtype):
        yy = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing='ij')
        grid = torch.stack([gx, gy], dim=-1)
        return grid.unsqueeze(0).expand(B, -1, -1, -1)

    @staticmethod
    def _offset_to_grid(offset, H, W):
        dx = offset[:, 0] * (2.0 / max(W - 1, 1))
        dy = offset[:, 1] * (2.0 / max(H - 1, 1))
        return torch.stack([dx, dy], dim=-1)

    def forward(self, z, phase_future):
        B, Tin, D, H, W = z.shape
        last = z[:, -1]
        last_delta = z[:, -1] - z[:, -2]
        mean_delta = (z[:, 1:] - z[:, :-1]).mean(dim=1)
        phase0 = phase_future[:, 0]
        vel_in = torch.cat([last_delta, mean_delta, phase0], dim=1)
        velocity = self.max_step_disp * torch.tanh(self.velocity(vel_in))  # [B,2,H,W]

        base_grid = self._base_grid(B, H, W, z.device, z.dtype)
        warped = []
        offsets = []
        confs = []
        for k in range(self.t_out):
            pf = phase_future[:, k]
            residual = 0.5 * torch.tanh(self.step_offset(torch.cat([last_delta, pf], dim=1)))
            offset = float(k + 1) * velocity + residual
            grid = base_grid - self._offset_to_grid(offset, H, W)
            wk = F.grid_sample(last, grid, mode='bilinear', padding_mode='border', align_corners=True)
            conf = self.confidence(torch.cat([wk, pf], dim=1))
            warped.append(wk * conf + last * (1.0 - conf))
            offsets.append(offset)
            confs.append(conf)
        return (
            torch.stack(warped, dim=1),
            torch.stack(offsets, dim=1),
            torch.stack(confs, dim=1),
        )

class SpatiotemporalWindowMixer(nn.Module):
    def __init__(self, dim, num_heads, total_time, window_size, dropout, drop_path, layer_scale, shift=False):
        super().__init__()
        self.total_time = total_time
        self.window_size = int(window_size)
        self.shift = bool(shift)
        self.norm = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, dropout, dropout)
        self.scale = LayerScale(dim, layer_scale)
        self.drop_path = DropPath(drop_path)
        self.bridge = nn.Sequential(
            _make_group_norm(dim),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )

    def _partition(self, x):
        B, T, D, H, W = x.shape
        ws = min(self.window_size, H, W)
        shift = ws // 2 if self.shift and ws > 1 else 0
        if shift:
            x = torch.roll(x, shifts=(-shift, -shift), dims=(-2, -1))

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        y = x.permute(0, 1, 3, 4, 2)
        if pad_h or pad_w:
            y = F.pad(y, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = y.shape[2], y.shape[3]
        nh, nw = Hp // ws, Wp // ws
        windows = y.reshape(B, T, nh, ws, nw, ws, D).permute(0, 2, 4, 1, 3, 5, 6)
        windows = windows.reshape(B * nh * nw, T * ws * ws, D)
        meta = (B, T, D, H, W, ws, shift, pad_h, pad_w, Hp, Wp, nh, nw)
        return windows, meta

    @staticmethod
    def _reverse(windows, meta):
        B, T, D, H, W, ws, shift, pad_h, pad_w, Hp, Wp, nh, nw = meta
        y = windows.reshape(B, nh, nw, T, ws, ws, D).permute(0, 3, 1, 4, 2, 5, 6)
        y = y.reshape(B, T, Hp, Wp, D)[:, :, :H, :W]
        x = y.permute(0, 1, 4, 2, 3)
        if shift:
            x = torch.roll(x, shifts=(shift, shift), dims=(-2, -1))
        return x

    def forward(self, x):
        B, T, D, H, W = x.shape
        x2 = x.reshape(B * T, D, H, W)
        x = (x2 + self.bridge(x2)).reshape(B, T, D, H, W)
        windows, meta = self._partition(x)
        windows = windows + self.drop_path(self.scale(self.attn(self.norm(windows))))
        return self._reverse(windows, meta)


class DualStreamWindowCrossMixer(nn.Module):
    def __init__(self, dim, num_heads, total_time, window_size, dropout, drop_path, layer_scale, shift=False):
        super().__init__()
        self.window_size = int(window_size)
        self.shift = bool(shift)
        self.norm_m = nn.LayerNorm(dim)
        self.norm_i = nn.LayerNorm(dim)
        self.m_from_i = CrossAttention(dim, num_heads, dropout, dropout)
        self.i_from_m = CrossAttention(dim, num_heads, dropout, dropout)
        self.scale_m = LayerScale(dim, layer_scale)
        self.scale_i = LayerScale(dim, layer_scale)
        self.drop_path = DropPath(drop_path)

    def _partition(self, x):
        B, T, D, H, W = x.shape
        ws = min(self.window_size, H, W)
        shift = ws // 2 if self.shift and ws > 1 else 0
        if shift:
            x = torch.roll(x, shifts=(-shift, -shift), dims=(-2, -1))
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        y = x.permute(0, 1, 3, 4, 2)
        if pad_h or pad_w:
            y = F.pad(y, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = y.shape[2], y.shape[3]
        nh, nw = Hp // ws, Wp // ws
        w = y.reshape(B, T, nh, ws, nw, ws, D).permute(0, 2, 4, 1, 3, 5, 6)
        w = w.reshape(B * nh * nw, T * ws * ws, D)
        return w, (B, T, D, H, W, ws, shift, Hp, Wp, nh, nw)

    @staticmethod
    def _reverse(w, meta):
        B, T, D, H, W, ws, shift, Hp, Wp, nh, nw = meta
        y = w.reshape(B, nh, nw, T, ws, ws, D).permute(0, 3, 1, 4, 2, 5, 6)
        y = y.reshape(B, T, Hp, Wp, D)[:, :, :H, :W]
        x = y.permute(0, 1, 4, 2, 3)
        if shift:
            x = torch.roll(x, shifts=(shift, shift), dims=(-2, -1))
        return x

    def forward(self, motion, intensity):
        mw, meta = self._partition(motion)
        iw, _ = self._partition(intensity)
        mn, inn = self.norm_m(mw), self.norm_i(iw)
        dm = self.m_from_i(mn, inn)
        di = self.i_from_m(inn, mn)
        mw = mw + self.drop_path(self.scale_m(dm))
        iw = iw + self.drop_path(self.scale_i(di))
        return self._reverse(mw, meta), self._reverse(iw, meta)


class TokenFFN(nn.Module):
    def __init__(self, dim, mlp_ratio, dropout, drop_path, layer_scale):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = SwiGLU(dim, mlp_ratio, dropout)
        self.scale = LayerScale(dim, layer_scale)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        y = x.permute(0, 1, 3, 4, 2)
        y = y + self.drop_path(self.scale(self.ffn(self.norm(y))))
        return y.permute(0, 1, 4, 2, 3)


class LatentComplexTemporalMixer(nn.Module):
    def __init__(self, total_time: int, dim: int, drop_path: float = 0.0, layer_scale: float = 1e-4):
        super().__init__()
        hidden_t = max(16, total_time * 2)
        self.total_time = total_time
        self.w1 = nn.Parameter(0.02 * torch.randn(2, total_time, hidden_t))
        self.w2 = nn.Parameter(0.02 * torch.randn(2, hidden_t, total_time))
        self.b1 = nn.Parameter(torch.zeros(2, 1, 1, 1, hidden_t))
        self.b2 = nn.Parameter(torch.zeros(2, 1, 1, 1, total_time))
        self.scale = nn.Parameter(torch.full((1, 1, dim, 1, 1), layer_scale))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        B, T, D, H, W = x.shape
        dtype = x.dtype
        xp = x.permute(0, 2, 3, 4, 1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = torch.fft.rfft2(xp.float(), dim=(2, 3), norm='ortho')
            r1 = torch.einsum('bdhwt,to->bdhwo', xf.real, self.w1[0]) - torch.einsum('bdhwt,to->bdhwo', xf.imag, self.w1[1]) + self.b1[0]
            i1 = torch.einsum('bdhwt,to->bdhwo', xf.real, self.w1[1]) + torch.einsum('bdhwt,to->bdhwo', xf.imag, self.w1[0]) + self.b1[1]
            r1, i1 = F.gelu(r1), F.gelu(i1)
            r2 = torch.einsum('bdhwt,to->bdhwo', r1, self.w2[0]) - torch.einsum('bdhwt,to->bdhwo', i1, self.w2[1]) + self.b2[0]
            i2 = torch.einsum('bdhwt,to->bdhwo', r1, self.w2[1]) + torch.einsum('bdhwt,to->bdhwo', i1, self.w2[0]) + self.b2[1]
            delta = torch.fft.irfft2(torch.complex(r2, i2), s=(H, W), dim=(2, 3), norm='ortho')
            delta = delta.permute(0, 4, 1, 2, 3)
        return x + self.drop_path(delta.to(dtype) * self.scale)


class CoupledSTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        total_time: int,
        window_size: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: float,
        layer_scale: float,
        shift: bool,
        use_spectral: bool,
    ):
        super().__init__()
        self.motion_st = SpatiotemporalWindowMixer(dim, num_heads, total_time, window_size, dropout, drop_path, layer_scale, shift)
        self.intensity_st = SpatiotemporalWindowMixer(dim, num_heads, total_time, window_size, dropout, drop_path, layer_scale, shift)
        self.cross = DualStreamWindowCrossMixer(dim, num_heads, total_time, window_size, dropout, drop_path, layer_scale, shift)
        self.motion_spec = LatentComplexTemporalMixer(total_time, dim, drop_path, layer_scale) if use_spectral else nn.Identity()
        self.intensity_spec = LatentComplexTemporalMixer(total_time, dim, drop_path, layer_scale) if use_spectral else nn.Identity()
        self.motion_ffn = TokenFFN(dim, mlp_ratio, dropout, drop_path, layer_scale)
        self.intensity_ffn = TokenFFN(dim, mlp_ratio, dropout, drop_path, layer_scale)

    def forward(self, motion, intensity):
        motion = self.motion_st(motion)
        intensity = self.intensity_st(intensity)
        motion, intensity = self.cross(motion, intensity)
        motion = self.motion_spec(motion)
        intensity = self.intensity_spec(intensity)
        motion = self.motion_ffn(motion)
        intensity = self.intensity_ffn(intensity)
        return motion, intensity

class DualStreamInitializer(nn.Module):
    def __init__(self, dim: int, t_in: int, t_out: int):
        super().__init__()
        self.t_in = t_in
        self.t_out = t_out
        self.motion_context = nn.Sequential(ResBlock2D(dim), ResBlock2D(dim))
        self.stats = nn.Sequential(
            nn.Conv2d(dim * 5, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1),
        )
        self.intensity_context = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU(),
            ResBlock2D(dim),
        )
        self.future_motion = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1),
            nn.GELU(),
            ResBlock2D(dim),
        )
        self.future_intensity = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1),
            nn.GELU(),
            ResBlock2D(dim),
        )
        self.motion_type = nn.Parameter(torch.zeros(1, 1, dim, 1, 1))
        self.intensity_type = nn.Parameter(torch.zeros(1, 1, dim, 1, 1))
        self.time_embed = nn.Parameter(torch.zeros(1, t_in + t_out, dim, 1, 1))
        _trunc_normal_(self.motion_type)
        _trunc_normal_(self.intensity_type)
        _trunc_normal_(self.time_embed)

    def forward(self, z, trajectory, phase_prior, spectral_prior, content_prior):
        B, Tin, D, H, W = z.shape
        Tout = trajectory.shape[1]
        delta = torch.cat([torch.zeros_like(z[:, :1]), z[:, 1:] - z[:, :-1]], dim=1)
        mctx = self.motion_context(delta.reshape(B * Tin, D, H, W)).reshape(B, Tin, D, H, W)

        st = torch.cat([
            z.mean(1), z.std(1, unbiased=False), z.max(1).values, z[:, -1], z[:, -1] - z[:, 0]
        ], dim=1)
        st = self.stats(st).unsqueeze(1).expand(-1, Tin, -1, -1, -1)
        ictx = self.intensity_context(torch.cat([z, st], dim=2).reshape(B * Tin, 2 * D, H, W)).reshape(B, Tin, D, H, W)

        mf = self.future_motion(torch.cat([trajectory, phase_prior, spectral_prior], dim=2).reshape(B * Tout, 3 * D, H, W)).reshape(B, Tout, D, H, W)
        inf = self.future_intensity(torch.cat([content_prior, spectral_prior, trajectory], dim=2).reshape(B * Tout, 3 * D, H, W)).reshape(B, Tout, D, H, W)

        motion = torch.cat([mctx, mf], dim=1) + self.motion_type + self.time_embed[:, :Tin + Tout]
        intensity = torch.cat([ictx, inf], dim=1) + self.intensity_type + self.time_embed[:, :Tin + Tout]
        return motion, intensity


class HighResolutionFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 5, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, 5, 1),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(dim * 5, dim * 2, 1),
            nn.GELU(),
            ResBlock2D(dim * 2),
            nn.Conv2d(dim * 2, dim, 1),
            ResBlock2D(dim),
        )

    def forward(self, backbone, trajectory, prior, content, phase):
        B, T, D, H, W = backbone.shape
        tensors = [backbone, trajectory, prior, content, phase]
        joint = torch.cat(tensors, dim=2).reshape(B * T, 5 * D, H, W)
        g = torch.softmax(self.gate(joint), dim=1)
        flat = [t.reshape(B * T, D, H, W) for t in tensors]
        base = sum(g[:, i:i+1] * flat[i] for i in range(5))
        return (base + self.refine(joint)).reshape(B, T, D, H, W)


class ResidualDecoder(nn.Module):
    def __init__(self, dim: int, out_channels: int, patch_size: int):
        super().__init__()
        self.decoder = nn.Sequential(
            ResBlock2D(dim),
            ResBlock2D(dim),
            nn.Conv2d(dim, out_channels * patch_size * patch_size, 3, padding=1),
            nn.PixelShuffle(patch_size),
        )
        nn.init.zeros_(self.decoder[-2].weight)
        nn.init.zeros_(self.decoder[-2].bias)

    def forward(self, x):
        B, T, D, H, W = x.shape
        y = self.decoder(x.reshape(B * T, D, H, W))
        return y.reshape(B, T, y.shape[1], y.shape[-2], y.shape[-1])


class PriorResidualFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.tensor(0.10))
        hidden = max(32, channels * 16)
        self.refine = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 3, padding=1),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, prior, residual, last_frame):
        B, T, C, H, W = prior.shape
        base = prior
        pred = base + self.residual_scale * residual
        last = last_frame.expand(-1, T, -1, -1, -1)
        refine = self.refine(torch.cat([pred, prior, last], dim=2).reshape(B * T, 3 * C, H, W))
        return (pred.reshape(B * T, C, H, W) + refine).reshape(B, T, C, H, W)

class ForecastLoss(nn.Module):
    def __init__(
        self,
        spec_num: int = 20,
        phase_weight: float = 0.01,
        amp_weight: float = 0.01,
        prior_weight: float = 0.10,
        motion_weight: float = 0.03,
    ):
        super().__init__()
        self.spec_num = spec_num
        self.phase_weight = phase_weight
        self.amp_weight = amp_weight
        self.prior_weight = prior_weight
        self.motion_weight = motion_weight

    def forward(self, pred, target, aux=None):
        mse = F.mse_loss(pred, target)
        total = mse
        out = {'mse_loss': mse}

        with torch.autocast(device_type=pred.device.type, enabled=False):
            pf = torch.fft.rfft2(pred.float(), dim=(-2, -1), norm='ortho')
            tf = torch.fft.rfft2(target.float(), dim=(-2, -1), norm='ortho')
            pa, ta = torch.abs(pf), torch.abs(tf)
            amp_loss = F.smooth_l1_loss(torch.log1p(pa), torch.log1p(ta))
            H, Wf = ta.shape[-2:]
            k = min(self.spec_num, max(1, H // 2), max(1, Wf - 1))
            mask = torch.zeros(H, Wf, device=pred.device)
            mask[:k, :k] = 1.0
            mask[-k:, :k] = 1.0
            mask = mask.view(1, 1, 1, H, Wf)
            phase_err = 1.0 - torch.cos(torch.angle(pf) - torch.angle(tf))
            amp_imp = (ta / (ta.mean(dim=(-2, -1), keepdim=True) + 1e-6)).clamp(max=5.0)
            phase_map = amp_imp * mask
            phase_loss = (phase_err * phase_map).sum() / (phase_map.sum() + 1e-6)

        amp_loss = amp_loss.to(mse.dtype)
        phase_loss = phase_loss.to(mse.dtype)
        total = total + self.amp_weight * amp_loss + self.phase_weight * phase_loss
        out['amp_loss'] = self.amp_weight * amp_loss
        out['phase_loss'] = self.phase_weight * phase_loss

        if pred.shape[1] > 1:
            motion_loss = F.smooth_l1_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])
            total = total + self.motion_weight * motion_loss
            out['motion_loss'] = self.motion_weight * motion_loss

        if aux is not None and 'spectral_prior' in aux:
            prior_loss = F.mse_loss(aux['spectral_prior'], target)
            total = total + self.prior_weight * prior_loss
            out['prior_loss'] = self.prior_weight * prior_loss

        out['total_loss'] = total
        return out

@dataclass
class ScaleConfig:
    hidden_dim: int
    depth: int
    num_heads: int
    patch_size: int = 4
    backbone_stride: int = 2
    window_size: int = 4
    mlp_ratio: float = 4.0
    spectral_every: int = 3
    prior_hidden_dim: int = 64
    prior_layers: int = 3

SCALE_CONFIGS: Dict[str, ScaleConfig] = {
    's120': ScaleConfig(hidden_dim=192, depth=10, num_heads=6, patch_size=4, backbone_stride=2, window_size=4, spectral_every=2, prior_hidden_dim=64, prior_layers=3),
    's200': ScaleConfig(hidden_dim=288, depth=18, num_heads=9, patch_size=4, backbone_stride=2, window_size=4, spectral_every=3, prior_hidden_dim=64, prior_layers=3),
    's400': ScaleConfig(hidden_dim=448, depth=24, num_heads=14, patch_size=4, backbone_stride=2, window_size=4, spectral_every=3, prior_hidden_dim=64, prior_layers=3),
    'forecast': ScaleConfig(hidden_dim=192, depth=10, num_heads=6, patch_size=4, backbone_stride=2, window_size=4, spectral_every=2, prior_hidden_dim=64, prior_layers=3),
}


class ScalableRadarBranchV4(nn.Module):
    def __init__(
        self,
        pre_seq_length: int,
        aft_seq_length: int,
        input_shape: Tuple[int, int] = (128, 128),
        input_dim: int = 1,
        hidden_dim: int = 192,
        depth: int = 10,
        num_heads: int = 6,
        patch_size: int = 4,
        backbone_stride: int = 2,
        window_size: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.10,
        layer_scale: float = 1e-3,
        spectral_every: int = 2,
        prior_hidden_dim: int = 64,
        prior_layers: int = 3,
        spec_num: int = 20,
        use_checkpoint: bool = False,
        phase_weight: float = 0.01,
        amp_weight: float = 0.01,
        prior_weight: float = 0.10,
        motion_weight: float = 0.03,
        max_step_disp: float = 2.0,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        assert input_shape[0] % (patch_size * backbone_stride) == 0
        assert input_shape[1] % (patch_size * backbone_stride) == 0
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.backbone_stride = backbone_stride
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        total_time = pre_seq_length + aft_seq_length

        self.spectral_prior = AlphaInspiredSpectralPrior(
            input_shape=input_shape,
            t_in=pre_seq_length,
            t_out=aft_seq_length,
            input_dim=input_dim,
            hidden_dim=prior_hidden_dim,
            n_layers=prior_layers,
            spec_num=spec_num,
        )

        self.input_embed = RadarPatchEmbedding(input_dim, hidden_dim, patch_size)
        self.prior_embed = RadarPatchEmbedding(input_dim, hidden_dim, patch_size)
        self.content_embed = RadarPatchEmbedding(input_dim, hidden_dim, patch_size)
        self.phase_embed = RadarPatchEmbedding(input_dim, hidden_dim, patch_size)

        self.trajectory = TrajectoryInitializer(hidden_dim, aft_seq_length, max_step_disp=max_step_disp)

        self.z_down = LatentDownsample(hidden_dim, backbone_stride)
        self.prior_down = LatentDownsample(hidden_dim, backbone_stride)
        self.content_down = LatentDownsample(hidden_dim, backbone_stride)
        self.phase_down = LatentDownsample(hidden_dim, backbone_stride)
        self.traj_down = LatentDownsample(hidden_dim, backbone_stride)

        self.initializer = DualStreamInitializer(hidden_dim, pre_seq_length, aft_seq_length)

        dpr = torch.linspace(0.0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            CoupledSTBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                total_time=total_time,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[i],
                layer_scale=layer_scale,
                shift=(i % 2 == 1),
                use_spectral=(spectral_every > 0 and (i + 1) % spectral_every == 0),
            )
            for i in range(depth)
        ])
        self.motion_norm = nn.LayerNorm(hidden_dim)
        self.intensity_norm = nn.LayerNorm(hidden_dim)

        self.low_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, 1),
            nn.GELU(),
            ResBlock2D(hidden_dim * 2),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
        )
        self.up = LatentUpsample(hidden_dim, backbone_stride)
        self.high_fusion = HighResolutionFusion(hidden_dim)
        self.decoder = ResidualDecoder(hidden_dim, input_dim, patch_size)
        self.output_fusion = PriorResidualFusion(input_dim)

        self.criterion = ForecastLoss(
            spec_num=spec_num,
            phase_weight=phase_weight,
            amp_weight=amp_weight,
            prior_weight=prior_weight,
            motion_weight=motion_weight,
        )

        self.apply(self._init_weights)
        nn.init.normal_(self.decoder.decoder[-2].weight, std=1e-3)
        nn.init.zeros_(self.decoder.decoder[-2].bias)
        nn.init.zeros_(self.output_fusion.refine[-1].weight)
        nn.init.zeros_(self.output_fusion.refine[-1].bias)
        nn.init.normal_(self.trajectory.velocity[-1].weight, std=1e-3)
        nn.init.zeros_(self.trajectory.velocity[-1].bias)
        nn.init.normal_(self.trajectory.step_offset[-1].weight, std=1e-3)
        nn.init.zeros_(self.trajectory.step_offset[-1].bias)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def _channel_norm(x, norm):
        return norm(x.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)

    def _run_block(self, block, motion, intensity):
        if self.use_checkpoint and self.training:
            return checkpoint(block, motion, intensity, use_reentrant=False)
        return block(motion, intensity)

    def forward(self, x, return_features: bool = False):
        B, Tin, C, H, W = x.shape
        if Tin != self.pre_seq_length:
            raise ValueError(f'Expected Tin={self.pre_seq_length}, got {Tin}')

        prior, sp_aux = self.spectral_prior(x)
        content = sp_aux['content_forecast']
        phase_img = sp_aux['phase_forecast']

        z_high = self.input_embed(x)
        prior_high = self.prior_embed(prior)
        content_high = self.content_embed(content)
        phase_high = self.phase_embed(phase_img)

        trajectory_high, offsets, traj_conf = self.trajectory(z_high, phase_high)

        z = self.z_down(z_high)
        prior_low = self.prior_down(prior_high)
        content_low = self.content_down(content_high)
        phase_low = self.phase_down(phase_high)
        traj_low = self.traj_down(trajectory_high)

        motion, intensity = self.initializer(
            z, traj_low, phase_low, prior_low, content_low
        )
        for block in self.blocks:
            motion, intensity = self._run_block(block, motion, intensity)

        motion = self._channel_norm(motion, self.motion_norm)
        intensity = self._channel_norm(intensity, self.intensity_norm)
        mf = motion[:, -self.aft_seq_length:]
        inf = intensity[:, -self.aft_seq_length:]

        low_joint = torch.cat([mf, inf, traj_low, prior_low], dim=2)
        low = self.low_fusion(
            low_joint.reshape(B * self.aft_seq_length, 4 * self.hidden_dim, low_joint.shape[-2], low_joint.shape[-1])
        ).reshape(B, self.aft_seq_length, self.hidden_dim, low_joint.shape[-2], low_joint.shape[-1])

        up = self.up(low)
        high = self.high_fusion(up, trajectory_high, prior_high, content_high, phase_high)

        residual = self.decoder(high)
        pred = self.output_fusion(prior, residual, x[:, -1:])

        if return_features:
            aux = {
                'spectral_prior': prior,
                'content_forecast': content,
                'phase_forecast': phase_img,
                'trajectory_features': trajectory_high,
                'trajectory_offsets': offsets,
                'trajectory_confidence': traj_conf,
                'motion_features': mf,
                'intensity_features': inf,
                'radar_features_low': low,
                'radar_features': high,
            }
            aux.update({k: v for k, v in sp_aux.items() if k not in aux})
            return pred, aux
        return pred

    def compute_loss(self, pred, target, aux=None):
        return self.criterion(pred, target, aux)

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        if compute_loss:
            pred, aux = self(frames_in, return_features=True)
            if frames_gt is None:
                raise ValueError('frames_gt is required when compute_loss=True')
            return pred, self.compute_loss(pred, frames_gt, aux)
        return self(frames_in), None

    def num_parameters(self, trainable_only: bool = False):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

def get_model(
    img_channels=1,
    T_in=5,
    T_out=20,
    input_shape=(128, 128),
    scale='forecast',
    dim=None,
    n_layers=None,
    patch_size=None,
    backbone_stride=None,
    num_heads=None,
    window_size=None,
    mlp_ratio=None,
    spectral_every=None,
    prior_hidden_dim=None,
    prior_layers=None,
    dropout=0.0,
    drop_path_rate=0.10,
    layer_scale=1e-3,
    use_checkpoint=False,
    spec_num=20,
    pha_weight=0.01,
    amp_weight=0.01,
    prior_weight=0.10,
    motion_weight=0.03,
    max_step_disp=2.0,
    **kwargs,
):
    if scale not in SCALE_CONFIGS:
        raise ValueError(f'Unknown scale={scale}. Available: {list(SCALE_CONFIGS)}')
    cfg = SCALE_CONFIGS[scale]
    return ScalableRadarBranchV4(
        pre_seq_length=T_in,
        aft_seq_length=T_out,
        input_shape=input_shape,
        input_dim=img_channels,
        hidden_dim=dim if dim is not None else cfg.hidden_dim,
        depth=n_layers if n_layers is not None else cfg.depth,
        num_heads=num_heads if num_heads is not None else cfg.num_heads,
        patch_size=patch_size if patch_size is not None else cfg.patch_size,
        backbone_stride=backbone_stride if backbone_stride is not None else cfg.backbone_stride,
        window_size=window_size if window_size is not None else cfg.window_size,
        mlp_ratio=mlp_ratio if mlp_ratio is not None else cfg.mlp_ratio,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        layer_scale=layer_scale,
        spectral_every=spectral_every if spectral_every is not None else cfg.spectral_every,
        prior_hidden_dim=prior_hidden_dim if prior_hidden_dim is not None else cfg.prior_hidden_dim,
        prior_layers=prior_layers if prior_layers is not None else cfg.prior_layers,
        spec_num=spec_num,
        use_checkpoint=use_checkpoint,
        phase_weight=pha_weight,
        amp_weight=amp_weight,
        prior_weight=prior_weight,
        motion_weight=motion_weight,
        max_step_disp=max_step_disp,
    )


if __name__ == '__main__':
    model = get_model(scale='s120', input_shape=(128, 128), T_in=5, T_out=20)
    print(f's120: {model.num_parameters()/1e6:.2f} M parameters')
