
# model.py — Modular VC models: CycleGAN(G/D) + ContentDecoder(FiLM) + ProsodyConverter
# - Generator/Discriminator: (B, 36, T) -> (B, 36, T), PatchGAN (optional multi-scale)
# - ContentDecoder: (content + logf0 [+vuv] [+style]) -> normalized MCEP
#     * style: you can concat [speaker_embed] (+ optional [age_embed]) outside and pass here
#     * stem norm is GN/none to preserve speaker cues better than IN
# - ProsodyConverter: low-dim prosody sequence (e.g., logf0[, energy]) A<->B (mini CycleGAN)
# - Speaker encoders/embedders + simple cosine speaker loss util
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ModelConfig",
    "Generator",
    "Discriminator",
    "DecoderCfg",
    "ContentDecoder",
    "ProsodyCfg",
    "ProsodyGenerator",
    "ProsodyDiscriminator",
    "ProsodyConverter",
    "SpkEncCfg",
    "SpeakerEncoder",
    "SpeakerIDEmbedding",
    "speaker_cosine_loss",
]

# =========================
# Config
# =========================
@dataclass
class ModelConfig:
    in_dim: int = 36
    base_channels: int = 128
    n_resblocks: int = 6
    kernel_size: int = 7
    res_kernel_size: int = 3
    norm: str = "in"               # "in"|"bn"|"gn"|"none"
    num_groups: int = 8
    dropout: float = 0.0
    activation: str = "glu"        # used by blocks (GLU gating)
    spectral_norm: bool = False
    use_sigmoid_D: bool = True
    multiscale_D: bool = False
    weight_init: str = "kaiming"   # "kaiming"|"xavier"|"none"

# =========================
# Helpers
# =========================
def get_norm_1d(c: int, kind: str, num_groups: int = 8) -> nn.Module:
    kind = (kind or "none").lower()
    if kind == "in":
        return nn.InstanceNorm1d(c, affine=True)
    if kind == "bn":
        return nn.BatchNorm1d(c)
    if kind == "gn":
        g = min(num_groups, c)
        return nn.GroupNorm(g, c)
    return nn.Identity()

def maybe_sn(module: nn.Module, use_sn: bool) -> nn.Module:
    return nn.utils.spectral_norm(module) if use_sn else module

def weight_init_fn(m: nn.Module, how: str = "kaiming"):
    if how == "none":
        return
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        if how == "xavier":
            nn.init.xavier_uniform_(m.weight)
        else:
            nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.InstanceNorm1d, nn.GroupNorm)):
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.ones_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)

# =========================
# Core Blocks
# =========================
class GatedConv1d(nn.Module):
    """Conv ⊗ sigmoid(Conv) gating (GLU)"""
    def __init__(self, in_ch, out_ch, k, s=1, p=0, spectral_norm=False):
        super().__init__()
        self.conv = maybe_sn(nn.Conv1d(in_ch, out_ch, k, s, p), spectral_norm)
        self.gate = maybe_sn(nn.Conv1d(in_ch, out_ch, k, s, p), spectral_norm)
    def forward(self, x):
        return self.conv(x) * torch.sigmoid(self.gate(x))

class ConvBlock1d(nn.Module):
    """[Conv or GatedConv] -> Norm -> Act -> Dropout"""
    def __init__(self, in_ch, out_ch, k, s=1, p=0,
                 norm="in", act="lrelu", dropout=0.0,
                 spectral_norm=False, gated=False, num_groups=8):
        super().__init__()
        if gated:
            self.conv = GatedConv1d(in_ch, out_ch, k, s, p, spectral_norm=spectral_norm)
            self.post = nn.Identity()
        else:
            self.conv = maybe_sn(nn.Conv1d(in_ch, out_ch, k, s, p), spectral_norm)
            self.post = get_norm_1d(out_ch, norm, num_groups=num_groups)
        self.act = self._get_act(act)
        self.do = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    @staticmethod
    def _get_act(name: str) -> nn.Module:
        name = (name or "lrelu").lower()
        if name == "relu": return nn.ReLU(inplace=True)
        if name == "lrelu": return nn.LeakyReLU(0.2, inplace=True)
        if name == "tanh": return nn.Tanh()
        if name == "glu": return nn.Identity()  # gated conv already applies sigmoid
        return nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.post(x)
        x = self.act(x)
        x = self.do(x)
        return x

class ResBlock1d(nn.Module):
    """
    Residual block (Generator-compatible signature).
    Uses ConvBlock1d twice (+ skip). Supports GLU/gated path.
    """
    def __init__(self, ch: int, k: int = 3, norm: str = "in",
                 act: str = "lrelu", dropout: float = 0.0,
                 spectral_norm: bool = False, gated: bool = False, num_groups: int = 8):
        super().__init__()
        p = k // 2
        self.b1 = ConvBlock1d(
            ch, ch, k, 1, p,
            norm=norm, act=act, dropout=dropout,
            spectral_norm=spectral_norm, gated=gated, num_groups=num_groups
        )
        self.b2 = ConvBlock1d(
            ch, ch, k, 1, p,
            norm=norm, act=act, dropout=dropout,
            spectral_norm=spectral_norm, gated=gated, num_groups=num_groups
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.b2(self.b1(x))

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation"""
    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, num_features)
        self.beta = nn.Linear(cond_dim, num_features)
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        g = self.gamma(z).unsqueeze(-1)
        b = self.beta(z).unsqueeze(-1)
        return x * (1.0 + g) + b

class ResFiLMBlock1d(nn.Module):
    """FiLM residual: two convs with per-channel FiLM from a style vector."""
    def __init__(self, ch: int, style_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm1d(ch, affine=False)  # keep affine off to not override FiLM
        self.norm2 = nn.InstanceNorm1d(ch, affine=False)
        self.film = nn.Sequential(
            nn.Linear(style_dim, ch * 4),
            nn.SiLU(),
            nn.Linear(ch * 4, ch * 4)
        )
    def forward(self, x, style):  # style: (B, S)
        B, C, L = x.shape
        g1, b1, g2, b2 = self.film(style).chunk(4, dim=1)   # (B,C) ×4
        g1 = g1.view(B, C, 1); b1 = b1.view(B, C, 1)
        g2 = g2.view(B, C, 1); b2 = b2.view(B, C, 1)
        h = self.conv1(x); h = self.norm1(h); h = (1.0 + g1) * h + b1; h = F.relu(h)
        h = self.conv2(h); h = self.norm2(h); h = (1.0 + g2) * h + b2
        h = x + h
        return F.relu(h)

# =========================
# CycleGAN-VC Generator / Discriminator
# =========================
class Generator(nn.Module):
    """
    API-compatible CycleGAN-VC generator (B, 36, T) -> (B, 36, T).
    """
    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg.base_channels
        k = self.cfg.kernel_size
        p = k // 2

        self.encoder = nn.Sequential(
            ConvBlock1d(self.cfg.in_dim, c, k, 1, p,
                        norm=self.cfg.norm, act="glu", dropout=self.cfg.dropout,
                        spectral_norm=self.cfg.spectral_norm, gated=True, num_groups=self.cfg.num_groups),
            get_norm_1d(c, self.cfg.norm, num_groups=self.cfg.num_groups),
        )

        self.res_blocks = nn.Sequential(
            *[
                ResBlock1d(c, k=self.cfg.res_kernel_size, norm=self.cfg.norm,
                           act="glu", dropout=self.cfg.dropout,
                           spectral_norm=self.cfg.spectral_norm, gated=True, num_groups=self.cfg.num_groups)
                for _ in range(self.cfg.n_resblocks)
            ]
        )

        self.decoder = nn.Sequential(
            maybe_sn(nn.Conv1d(c, self.cfg.in_dim, k, 1, p), self.cfg.spectral_norm),
            nn.Tanh(),
        )

        self.apply(lambda m: weight_init_fn(m, self.cfg.weight_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.res_blocks(h)
        y = self.decoder(h)
        return y

class PatchDiscriminator1D(nn.Module):
    """
    PatchGAN 1D discriminator. Returns score map (B,1,T').
    """
    def __init__(self,
                 in_dim: int = 36,
                 channels: List[int] = [128, 256],
                 ks: int = 5,
                 stride: int = 2,
                 spectral_norm: bool = False,
                 use_sigmoid: bool = True):
        super().__init__()
        p = ks // 2
        layers: List[nn.Module] = []
        last_c = in_dim
        for c in channels:
            layers += [
                maybe_sn(nn.Conv1d(last_c, c, ks, stride, p), spectral_norm),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            last_c = c
        layers += [maybe_sn(nn.Conv1d(last_c, 1, 3, 1, 1), spectral_norm)]
        if use_sigmoid:
            layers += [nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        self.apply(lambda m: weight_init_fn(m, "kaiming"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MultiScaleDiscriminator(nn.Module):
    """Two-scale PatchGAN: stride2 & stride4 receptive fields. Sum outputs."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.d1 = PatchDiscriminator1D(
            in_dim=cfg.in_dim, channels=[128, 256], ks=5, stride=2,
            spectral_norm=cfg.spectral_norm, use_sigmoid=cfg.use_sigmoid_D
        )
        self.d2 = PatchDiscriminator1D(
            in_dim=cfg.in_dim, channels=[128, 256], ks=7, stride=4,
            spectral_norm=cfg.spectral_norm, use_sigmoid=cfg.use_sigmoid_D
        )
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        s1 = self.d1(x)
        s2 = self.d2(x)
        return s1 + F.interpolate(s2, size=s1.shape[-1], mode="nearest")  # align & sum

class Discriminator(nn.Module):
    """Wrapper discriminator to keep API compatibility."""
    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        if self.cfg.multiscale_D:
            self.D = MultiScaleDiscriminator(self.cfg)
        else:
            self.D = PatchDiscriminator1D(
                in_dim=self.cfg.in_dim, channels=[128, 256], ks=5, stride=2,
                spectral_norm=self.cfg.spectral_norm, use_sigmoid=self.cfg.use_sigmoid_D
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.D(x)

# =========================
# Speaker encoders / embedders
# =========================
@dataclass
class SpkEncCfg:
    in_dim: int = 36
    channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    out_dim: int = 128
    use_l2_norm: bool = True

class SpeakerEncoder(nn.Module):
    """
    Light conv encoder with statistical pooling (mean,std).
    Input:  (B, in_dim=36, T)
    Output: (B, out_dim)
    """
    def __init__(self, cfg: SpkEncCfg = SpkEncCfg()):
        super().__init__()
        layers = []
        last_c = cfg.in_dim
        for c in cfg.channels:
            layers.append(nn.Conv1d(last_c, c, kernel_size=3, stride=2, padding=1))
            layers.append(nn.InstanceNorm1d(c, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            last_c = c
        self.convs = nn.Sequential(*layers)
        self.proj = nn.Linear(last_c * 2, cfg.out_dim)
        self.cfg = cfg
        self.apply(lambda m: weight_init_fn(m, "kaiming"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.convs(x)                     # (B,C,T')
        mean, std = h.mean(dim=2), h.std(dim=2)
        stat_pooled = torch.cat([mean, std], dim=1)
        emb = self.proj(stat_pooled)          # (B, out_dim)
        if self.cfg.use_l2_norm:
            emb = F.normalize(emb, dim=1)
        return emb

class SpeakerIDEmbedding(nn.Module):
    """Simple trainable embedding table for speaker IDs."""
    def __init__(self, num_speakers: int, dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_speakers, dim)
        nn.init.normal_(self.embed.weight, std=0.02)
    def forward(self, spk_ids: torch.LongTensor) -> torch.Tensor:
        return self.embed(spk_ids)

def speaker_cosine_loss(emb_pred: torch.Tensor, emb_ref: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """
    1 - cosine similarity (optional margin). Both inputs are (B, E).
    Use with λ_spk in train loop to preserve speaker identity.
    """
    emb_pred = F.normalize(emb_pred, dim=1)
    emb_ref  = F.normalize(emb_ref,  dim=1)
    cos = (emb_pred * emb_ref).sum(dim=1)
    loss = 1.0 - cos
    if margin > 0:
        loss = torch.clamp(loss - margin, min=0.0)
    return loss.mean()

# =========================
# Content-fixed Decoder (FiLM style)
# =========================
@dataclass
class DecoderCfg:
    # input dims
    content_dim: int                    # content channels
    pitch_dim: int = 1                  # logf0 channels
    vuv_dim: int = 0                    # voicing flag channels (0/1)
    # network
    out_dim: int = 36
    channels: int = 256
    n_resblocks: int = 8
    # style
    style_dim: int = 0                  # 0 -> no style/FiLM
    use_film: bool = False
    # speaker-preserving stem norm (avoid IN here)
    stem_norm: str = "gn"               # "gn"|"none"|"bn"|"in"
    stem_groups: int = 8

class ContentDecoder(nn.Module):
    """
    Inputs:
      content: (B, Cc, L)
      pitch:   (B,  1, L)   # normalized logf0
      vuv:     (B,  1, L) or None
      style:   (B,  S) or None   # concat([speaker_embed], [age_embed]) outside if needed
    Output:
      y_mcep:  (B, out_dim=36, L)
    """
    def __init__(self, cfg: DecoderCfg):
        super().__init__()
        self.cfg = cfg
        self.use_film = (cfg.use_film and cfg.style_dim > 0)

        in_ch = cfg.content_dim + cfg.pitch_dim + (cfg.vuv_dim if cfg.vuv_dim > 0 else 0)
        ch = cfg.channels

        # stem: prefer GN or no norm to preserve speaker cues
        stem_norm = get_norm_1d(ch, cfg.stem_norm, num_groups=cfg.stem_groups)
        self.inp = nn.Sequential(
            nn.Conv1d(in_ch, ch, kernel_size=5, padding=2),
            stem_norm,
            nn.SiLU(),
        )

        # FiLM or plain residual stack
        blocks: List[nn.Module] = []
        if self.use_film:
            for _ in range(cfg.n_resblocks):
                blocks.append(ResFiLMBlock1d(ch, cfg.style_dim))
        else:
            for _ in range(cfg.n_resblocks):
                blocks.append(ResBlock1d(ch, k=3, norm="gn", act="lrelu", gated=False, num_groups=cfg.stem_groups))
        self.blocks = nn.ModuleList(blocks)

        self.out = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(ch, cfg.out_dim, kernel_size=1),
        )

        self.apply(lambda m: weight_init_fn(m, "kaiming"))

    def forward(self, content, pitch, vuv=None, style=None):
        xs = [content, pitch]
        if vuv is not None and self.cfg.vuv_dim > 0:
            xs.append(vuv)
        x = torch.cat(xs, dim=1)                 # (B,in_ch,L)

        h = self.inp(x)
        if self.use_film:
            assert style is not None, "use_film=True but style=None"
            for blk in self.blocks:
                h = blk(h, style)
        else:
            for blk in self.blocks:
                h = blk(h)

        y = self.out(h)
        return y

# =========================
# Prosody Converter (low-dim CycleGAN for logf0/energy)
# =========================
@dataclass
class ProsodyCfg:
    in_dim: int = 1                 # 1: logf0 only, 2: logf0+energy
    channels: int = 64
    n_resblocks: int = 6
    ks: int = 7
    res_ks: int = 3
    norm: str = "in"
    spectral_norm: bool = False
    use_sigmoid_D: bool = True
    weight_init: str = "kaiming"

class ProsodyGenerator(nn.Module):
    """(B, D, T) -> (B, D, T)"""
    def __init__(self, cfg: ProsodyCfg = ProsodyCfg()):
        super().__init__()
        c = cfg.channels
        p = cfg.ks // 2
        self.enc = nn.Sequential(
            ConvBlock1d(cfg.in_dim, c, cfg.ks, 1, p,
                        norm=cfg.norm, act="lrelu", dropout=0.0,
                        spectral_norm=cfg.spectral_norm, gated=False),
            get_norm_1d(c, cfg.norm),
        )
        self.trunk = nn.Sequential(
            *[ResBlock1d(c, k=cfg.res_ks, norm=cfg.norm, act="lrelu",
                         dropout=0.0, spectral_norm=cfg.spectral_norm, gated=False)
              for _ in range(cfg.n_resblocks)]
        )
        self.dec = nn.Sequential(
            maybe_sn(nn.Conv1d(c, cfg.in_dim, cfg.ks, 1, p), cfg.spectral_norm),
            nn.Tanh()
        )
        self.apply(lambda m: weight_init_fn(m, cfg.weight_init))

    def forward(self, x):
        h = self.enc(x)
        h = self.trunk(h)
        y = self.dec(h)
        return y

class ProsodyDiscriminator(nn.Module):
    """PatchGAN for low-dim sequences"""
    def __init__(self, cfg: ProsodyCfg = ProsodyCfg()):
        super().__init__()
        self.net = PatchDiscriminator1D(
            in_dim=cfg.in_dim, channels=[64, 128], ks=5, stride=2,
            spectral_norm=cfg.spectral_norm, use_sigmoid=cfg.use_sigmoid_D
        )
        self.apply(lambda m: weight_init_fn(m, cfg.weight_init))

    def forward(self, x):
        return self.net(x)

class ProsodyConverter(nn.Module):
    """
    A wrapper holding A2B/B2A generators & discriminators for prosody.
    Training loop is in train.py.
    """
    def __init__(self, cfg: ProsodyCfg = ProsodyCfg()):
        super().__init__()
        self.G_A2B = ProsodyGenerator(cfg)
        self.G_B2A = ProsodyGenerator(cfg)
        self.D_A = ProsodyDiscriminator(cfg)
        self.D_B = ProsodyDiscriminator(cfg)

# =========================
# Self-test
# =========================
if __name__ == "__main__":
    B, T = 2, 256

    # 1) CycleGAN-VC G/D
    G = Generator()
    D = Discriminator()
    x = torch.randn(B, 36, T)
    y = G(x)
    s = D(y)
    print("[CycleGAN] x:", x.shape, "y:", y.shape, "D(y):", s.shape)

    # 2) Speaker encoders / embedders
    spk_enc = SpeakerEncoder(SpkEncCfg())
    spk_emb_from_mcep = spk_enc(x)  # pretend x is mcep
    print("[SpkEnc] emb:", spk_emb_from_mcep.shape)

    spk_table = SpeakerIDEmbedding(num_speakers=10, dim=128)
    spk_ids = torch.tensor([1, 3])
    spk_emb_from_id = spk_table(spk_ids)
    print("[SpkID] emb:", spk_emb_from_id.shape)

    # 3) ContentDecoder (speaker-first; style = spk_emb [+ age_emb(optional)])
    style_dim = 128  # speaker-only for now
    dec_cfg = DecoderCfg(content_dim=768, pitch_dim=1, vuv_dim=1,
                         out_dim=36, channels=256, n_resblocks=6,
                         style_dim=style_dim, use_film=True,
                         stem_norm="gn", stem_groups=8)
    DEC = ContentDecoder(dec_cfg)
    content = torch.randn(B, 768, T)
    logf0   = torch.randn(B,   1, T)
    vuv     = torch.randint(0, 2, (B, 1, T)).float()
    style   = spk_emb_from_id  # or torch.cat([spk_emb, age_emb], dim=1)
    y_mcep  = DEC(content, logf0, vuv=vuv, style=style)
    print("[Decoder] out:", y_mcep.shape)

    # 4) ProsodyConverter (logf0 only)
    pcfg = ProsodyCfg(in_dim=1, channels=64, n_resblocks=3)
    P = ProsodyConverter(pcfg)
    f0A = torch.randn(B, 1, T)
    f0B = torch.randn(B, 1, T)
    fb = P.G_A2B(f0A)
    fa = P.G_B2A(f0B)
    sdA = P.D_A(fa)
    sdB = P.D_B(fb)
    print("[Prosody] fb:", fb.shape, "fa:", fa.shape, "sdA:", sdA.shape, "sdB:", sdB.shape)


