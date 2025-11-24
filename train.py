'''
# train.py — Recon + Speaker Consistency + Age Constraint (KD with Keras Teacher)
# - Generator: ContentDecoder(content, logf0[, vuv], style) -> MCEP(36,L)
# - Loss: L1 + Δ/Δ² + SpeakerCos + Grouped-CE(청년/중장년) + KD(teacher 18-way)
# - TF teacher(.h5/.keras/SavedModel dir) robust load (TF on CPU)
# - AgeHead: GPU+FP32 고정, 호출부는 AMP 끄고 실행 → dtype/device 에러 방지

import os
import math
import time
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ====== 프로젝트 모듈 ======
from data_preprocess import CacheIO
from model import ContentDecoder, DecoderCfg, SpeakerEncoder, SpkEncCfg

# ====== TensorFlow/Keras (teacher) ======
import tensorflow as tf
# TF는 CPU만 사용 (CUDA 충돌 방지)
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

# ---------------- Utils ----------------
def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def now(): return time.strftime("%H:%M:%S")

def infinite_loader(loader: DataLoader) -> Iterator:
    while True:
        for b in loader:
            yield b

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ---------------- Dataset ----------------
class ReconDataset(Dataset):
    """
    Returns: (content_norm[Cc,L], logf0_norm[1,L], vuv[1,L], target_mcep[36,L], domain_id[int])
    Cache keys (from data_preprocess.py):
      coded_sps_A_norm/B_norm        : List[(36,T)]
      content_emb_A/B                : List[(T,Cc)]
      logf0_seq_A/B                  : List[(T,1)]  (<=0 => unvoiced)
      vuv_A/B                        : List[(T,1)]  (optional)
      content_normalization          : {"mean":(Cc,1), "std":(Cc,1)}
      logf0_global_normalization     : {"mean":float, "std":float}
    """
    def __init__(self, cache, n_frames=192, use_vuv=True):
        self.n_frames = int(n_frames); self.use_vuv = use_vuv

        A = cache["coded_sps_A_norm"]; B = cache["coded_sps_B_norm"]
        self.targets = A + B
        self.dom_ids = [0]*len(A) + [1]*len(B)

        self.contents = cache["content_emb_A"] + cache["content_emb_B"]
        self.logf0s   = cache["logf0_seq_A"] + cache["logf0_seq_B"]

        self.vuvs = None
        if use_vuv and ("vuv_A" in cache and "vuv_B" in cache):
            self.vuvs = cache["vuv_A"] + cache["vuv_B"]

        self.cmean = cache["content_normalization"]["mean"].astype(np.float32)
        self.cstd  = cache["content_normalization"]["std"].astype(np.float32)
        self.lmean = float(cache["logf0_global_normalization"]["mean"])
        self.lstd  = float(cache["logf0_global_normalization"]["std"])

        print(f"[Dataset] utters={len(self.targets)} (A:{len(A)}, B:{len(B)}) | seg_len={self.n_frames}")

    def __len__(self): return len(self.targets)

    @staticmethod
    def _padT(arr: np.ndarray, target_T: int, time_axis: int) -> np.ndarray:
        T = arr.shape[time_axis]
        need = target_T - T + 1
        L = max(need//2, 0); R = max(need-L, 0)
        pw = [(0,0)]*arr.ndim; pw[time_axis] = (L,R)
        return np.pad(arr, pw, mode="edge")

    def __getitem__(self, idx: int):
        y   = self.targets[idx]         # (36,T)
        c   = self.contents[idx]        # (T,Cc)
        lf0 = self.logf0s[idx]          # (T,1)
        v   = self.vuvs[idx] if self.vuvs is not None else None
        d   = self.dom_ids[idx]

        T = y.shape[1]
        if T <= self.n_frames:
            y   = self._padT(y, self.n_frames, 1)
            c   = self._padT(c, self.n_frames, 0)
            lf0 = self._padT(lf0, self.n_frames, 0)
            if v is not None: v = self._padT(v, self.n_frames, 0)
            T = y.shape[1]

        s = np.random.randint(0, T - self.n_frames); e = s + self.n_frames
        y   = y[:, s:e]            # (36,L)
        c   = c[s:e, :]            # (L,Cc)
        lf0 = lf0[s:e, :]          # (L,1)
        v   = v[s:e, :] if self.vuvs is not None else np.zeros((self.n_frames,1), np.float32)

        # content 정규화
        c = (c.T - self.cmean)/self.cstd  # (Cc,L)

        # logf0 글로벌 정규화 (무성=0)
        lf0n = np.zeros_like(lf0)
        nz = lf0[:,0] > 0
        if np.any(nz): lf0n[nz] = (lf0[nz] - self.lmean)/self.lstd

        return (torch.from_numpy(c.astype(np.float32)),
                torch.from_numpy(lf0n.T.astype(np.float32)),
                torch.from_numpy(v.T.astype(np.float32)),
                torch.from_numpy(y.astype(np.float32)),
                torch.tensor(d, dtype=torch.long))

# ---------------- EMA ----------------
class EMA:
    def __init__(self, model: nn.Module, decay=0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k,v in model.state_dict().items()}
        self.backup = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0-self.decay)

    def store(self, model: nn.Module):
        self.backup = {k: v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model: nn.Module):
        if self.backup is not None:
            model.load_state_dict(self.backup, strict=False)
            self.backup = None

# ---------------- Age Head (student, 1D conv) ----------------
class AgeHeadPT(nn.Module):
    def __init__(self, in_dim=36, channels=128, n_blocks=2, n_classes=18):
        super().__init__()
        layers = [nn.Conv1d(in_dim, channels, 5, padding=2), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(n_blocks-1):
            layers += [nn.Conv1d(channels, channels, 5, padding=2), nn.LeakyReLU(0.2, inplace=True)]
        self.feat = nn.Sequential(*layers)
        self.cls  = nn.Linear(channels, n_classes)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, mcep):  # mcep: (B,36,L)
        h = self.feat(mcep)               # (B,C,L)
        g = h.mean(dim=2)                 # GAP
        return self.cls(g)                # (B,n_classes)

# ---------------- KD & Grouped-CE ----------------
def kd_loss(student_logits: torch.Tensor, teacher_probs: torch.Tensor, T: float = 2.0) -> torch.Tensor:
    log_p = F.log_softmax(student_logits / T, dim=1)
    q = teacher_probs / T
    q = q / q.sum(dim=1, keepdim=True)
    return F.kl_div(log_p, q, reduction="batchmean") * (T*T)

def grouped_ce_loss(logits: torch.Tensor, d: torch.Tensor, groups: Dict[int, list]) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)  # (B,C)
    losses = []
    for i in range(logits.size(0)):
        gi = int(d[i].item())
        idxs = groups[gi]
        g = logp[i, idxs]
        losses.append( -(torch.logsumexp(g, dim=0) - math.log(len(idxs))) )
    return torch.stack(losses).mean()

# ---------------- Keras Teacher (robust loader) ----------------
from pathlib import Path
import zipfile
def robust_load_keras_model(path_like: str):
    p = Path(path_like)
    if not p.is_absolute():
        p = (Path(__file__).parent / p).resolve()
    path_posix = p.as_posix()
    # SavedModel dir
    if p.is_dir():
        return tf.keras.models.load_model(path_posix)
    # .keras zip
    if zipfile.is_zipfile(p):
        return tf.keras.models.load_model(path_posix)
    # HDF5(.h5 or .keras but actually h5)
    try:
        import h5py
        with h5py.File(p, "r"): pass
        return tf.keras.models.load_model(path_posix, compile=False)
    except Exception:
        pass
    size = os.path.getsize(p) if p.exists() else -1
    raise ValueError(f"[robust_load_keras_model] Unsupported or broken file: '{p}' (size={size}).")

@torch.no_grad()
def build_age_vector_111(mcep: torch.Tensor, lf0: torch.Tensor) -> np.ndarray:
    """
    Teacher 입력이 (B, 111) (Dense MLP)라고 가정 — 통계 기반 111차 벡터 생성
    - MCEP: mean/std/skew (36*3=108)
    - F0  : voiced mean/std + voiced ratio (=3)
    * torch.nanmean/nanstd 미사용(구버전 호환). 무성 구간은 마스크로 제외.
    """
    with torch.amp.autocast(mcep.device.type if mcep.is_cuda else "cpu", enabled=False):
        m = mcep.float()  # (B,36,L)
        f = lf0.float()   # (B,1,L), unvoiced=0

        # MCEP 통계
        m_mean = m.mean(dim=2)                        # (B,36)
        m_std  = m.std(dim=2).clamp_min(1e-8)         # (B,36)
        m_center = m - m_mean[..., None]
        m_skew = (m_center.pow(3).mean(dim=2) / (m_std.pow(3))).clamp(-1e6, 1e6)

        # F0 통계 (무성 마스크)
        voiced = (f != 0).float()                     # (B,1,L)
        denom = voiced.sum(dim=(1,2), keepdim=True).clamp_min(1e-6)
        f_mean = ((f * voiced).sum(dim=(1,2), keepdim=True) / denom)
        f_var  = (((f - f_mean)**2 * voiced).sum(dim=(1,2), keepdim=True) / denom)
        f_std  = torch.sqrt(f_var + 1e-8)
        v_rate = voiced.mean(dim=(1,2), keepdim=True)

        f_mean = f_mean.squeeze(-1)  # (B,1)
        f_std  = f_std.squeeze(-1)   # (B,1)
        v_rate = v_rate.squeeze(-1)  # (B,1)

        feats = torch.cat([m_mean, m_std, m_skew, f_mean, f_std, v_rate], dim=1)  # (B,111)
        feats = feats.clamp(-1e6, 1e6)
        return feats.cpu().numpy().astype(np.float32)

class KerasTeacher:
    def __init__(self, keras_path: str):
        p = Path(keras_path)
        if not p.is_absolute():
            p = (Path(__file__).parent / p).resolve()
        print("[Teacher] using path :", p)
        print("[Teacher] exists(os):", os.path.exists(p))
        try:
            print("[Teacher] zip(.keras)?", zipfile.is_zipfile(p))
            print("[Teacher] size(bytes):", os.path.getsize(p) if p.exists() else -1)
        except Exception:
            pass
        self.model = robust_load_keras_model(p)
        self.in_dim = int(self.model.input_shape[-1])
        self.n_classes = int(self.model.output_shape[-1])
        print(f"[Teacher] loaded OK | in_dim={self.in_dim} | classes={self.n_classes}")

    @torch.no_grad()
    def predict_probs(self, mcep: torch.Tensor, lf0: torch.Tensor) -> torch.Tensor:
        feats = build_age_vector_111(mcep, lf0)            # (B,111)
        probs = self.model(feats, training=False).numpy()  # (B, n_classes)
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return torch.from_numpy(probs).float()

# ---------------- Config ----------------
@dataclass
class ReconConfig:
    # 경로/런타임
    cache_dir: str = "./cache"
    out_dir: str = "checkpoints_recon"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337

    # 학습
    epochs: int = 1200
    steps_per_epoch: int = 500
    batch_size: int = 4
    n_frames: int = 192
    num_workers: int = 0
    pin_memory: bool = True

    # 옵티마이저/AMP
    lr: float = 2e-4
    betas: Tuple[float, float] = (0.9, 0.98)
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    amp: bool = True

    # 손실 가중치 (부가 손실 유지)
    lambda_delta: float = 0.5
    lambda_delta2: float = 0.25
    lambda_spk: float = 1.0
    lambda_age_ce: float = 0.5
    lambda_kd: float = 0.5
    kd_T: float = 2.0
    kd_every: int = 1

    # 모델
    content_dim: Optional[int] = None
    use_vuv: bool = True
    channels: int = 256
    n_resblocks: int = 8
    spk_style_dim: int = 128
    spk_encoder_trainable: bool = False  # 기본 동결(훈련/추론 정합)

    # Teacher
    keras_model_path: str = "age_model/best_model.h5"

    # 18클래스 → 도메인(0/1) 매핑
    age_class_groups: Optional[Dict[int, list]] = None

    # EMA/저장/로그
    ema_decay: float = 0.999
    save_every_steps: int = 5000
    log_every: int = 50

    # === (NEW) 스타일 프로토타입 저장 설정 ===
    save_style_prototypes: bool = True      # 도메인 평균 스타일 저장할지
    style_proto_refs: int = 16              # 평균 벡터 만들 때 샘플 수
    style_proto_seglen: int = 192           # 각 샘플 크롭 길이 (n_frames와 맞춤)
    style_prototypes_path: str = "checkpoints_recon/style_prototypes.pth"

    # === (NEW) 디버그/모니터링 ===
    log_yhat_rms: bool = True               # y_hat RMS 로깅 (붕괴 탐지)


# ---------------- Trainer ----------------
class Trainer:
    def __init__(self, cfg: ReconConfig):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        ensure_dir(cfg.out_dir)

        # ----- data -----
        self.cache = CacheIO.load_preprocessed_data(cfg.cache_dir)
        if cfg.content_dim is None:
            cfg.content_dim = int(self.cache["content_normalization"]["mean"].shape[0])
        self.dataset = ReconDataset(self.cache, n_frames=cfg.n_frames, use_vuv=cfg.use_vuv)
        self.loader = DataLoader(self.dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                                 num_workers=cfg.num_workers,
                                 pin_memory=(cfg.pin_memory and self.device.type=="cuda"))
        self.iter = infinite_loader(self.loader)

        # ----- teacher 먼저 로드 (클래스 수 맞추기) -----
        self.teacher = KerasTeacher(cfg.keras_model_path)

        # ----- models -----
        # SpeakerEncoder: 기본 동결(훈련/추론 일치). 학습 원하면 cfg.spk_encoder_trainable=True
        self.spk_enc = SpeakerEncoder(SpkEncCfg(in_dim=36, out_dim=cfg.spk_style_dim)).to(self.device)
        if not cfg.spk_encoder_trainable:
            for p in self.spk_enc.parameters():
                p.requires_grad_(False)
            self.spk_enc.eval()

        dec_cfg = DecoderCfg(
            content_dim=cfg.content_dim, pitch_dim=1,
            vuv_dim=(1 if cfg.use_vuv else 0),
            out_dim=36, channels=cfg.channels, n_resblocks=cfg.n_resblocks,
            style_dim=cfg.spk_style_dim, use_film=True
        )
        self.net = ContentDecoder(dec_cfg).to(self.device)

        # AgeHead — teacher n_classes로 동기화
        self.age_head = AgeHeadPT(in_dim=36, channels=128, n_blocks=2,
                                  n_classes=self.teacher.n_classes).to(self.device).float()

        # ----- opt/amp/ema -----
        params = list(self.net.parameters()) + list(self.age_head.parameters())
        if cfg.spk_encoder_trainable:
            params += list(self.spk_enc.parameters())
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)

        try:
            from torch.amp import autocast as _autocast, GradScaler as _GradScaler
            self.autocast = lambda: _autocast(self.device.type, enabled=cfg.amp)
            self.scaler = _GradScaler(enabled=cfg.amp)
        except Exception:
            self.autocast = lambda: torch.cuda.amp.autocast(enabled=cfg.amp)
            self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

        self.ema = EMA(self.net, decay=cfg.ema_decay)

        # ----- loss -----
        self.l1 = nn.L1Loss()

        print(f"[Init] device={self.device} | Decoder={count_params(self.net):,} | "
              f"AgeHead={count_params(self.age_head):,} | content_dim={cfg.content_dim} | "
              f"spk_trainable={cfg.spk_encoder_trainable}")

    # helpers
    def _delta(self, x):  return x[:,:,1:] - x[:,:,:-1]
    def _delta2(self, x): return self._delta(self._delta(x))

    def _speaker_cosine(self, y_hat, y_ref):
        # spk encoder는 FP32/NO-AMP로
        with torch.amp.autocast(self.device.type, enabled=False):
            e_ref = self.spk_enc(y_ref.float())
            e_hat = self.spk_enc(y_hat.float())
            e_ref = F.normalize(e_ref, dim=1); e_hat = F.normalize(e_hat, dim=1)
            return (1.0 - (e_ref*e_hat).sum(dim=1)).mean()

    def _save_ckpt(self, tag="step"):
        torch.save(self.net.state_dict(), os.path.join(self.cfg.out_dir, f"content_decoder_{tag}.pth"))
        torch.save(self.age_head.state_dict(), os.path.join(self.cfg.out_dir, f"age_head_{tag}.pth"))
        if self.cfg.spk_encoder_trainable:
            torch.save(self.spk_enc.state_dict(), os.path.join(self.cfg.out_dir, f"spk_encoder_{tag}.pth"))

    @torch.no_grad()
    def _style_mean(self, pool: list, n_refs: int, seg_len: int) -> torch.Tensor:
        """
        pool: list of np.ndarray (36, T) — self.cache["coded_sps_{A,B}_norm"]
        반환: (1, E) 평균 스타일 벡터
        """
        if len(pool) == 0:
            raise RuntimeError("style_mean: empty pool")

        idxs = np.linspace(0, len(pool)-1, min(n_refs, len(pool)), dtype=int)
        vecs = []
        for i in idxs:
            mcep = np.asarray(pool[i], dtype=np.float32)  # (36, T)
            T = mcep.shape[1]
            if T < seg_len:
                need = seg_len - T + 1
                L = max(need//2, 0); R = max(need-L, 0)
                mcep = np.pad(mcep, ((0,0),(L,R)), mode="edge")
                T = mcep.shape[1]
            s = (T - seg_len)//2
            e = s + seg_len
            seg = torch.from_numpy(mcep[:, s:e]).unsqueeze(0).to(self.device)  # (1,36,L)
            v = self.spk_enc(seg)   # (1,E)
            vecs.append(v)
        V = torch.cat(vecs, dim=0)       # (N,E)
        return V.mean(dim=0, keepdim=True).float()  # (1,E)

    @torch.no_grad()
    def save_domain_style_prototypes(self):
        """
        도메인 A/B 평균 스타일을 파일로 저장 (test.py가 바로 사용 가능)
        """
        if not self.cfg.save_style_prototypes:
            return
        A = self.cache["coded_sps_A_norm"]
        B = self.cache["coded_sps_B_norm"]
        n = int(self.cfg.style_proto_refs)
        L = int(self.cfg.style_proto_seglen)

        wA = self._style_mean(A, n_refs=n, seg_len=L).cpu()
        wB = self._style_mean(B, n_refs=n, seg_len=L).cpu()
        out = self.cfg.style_prototypes_path
        ensure_dir(os.path.dirname(out))
        torch.save({"A": wA, "B": wB}, out)
        print(f"[{now()}] 💾 Saved style prototypes → {out} | ||A||={float(wA.norm()):.4f}, ||B||={float(wB.norm()):.4f}")

    def train(self):
        gstep = 0
        for epoch in range(1, self.cfg.epochs+1):
            mean_loss = 0.0
            for step in range(self.cfg.steps_per_epoch):
                gstep += 1
                c, l, v, y, d = next(self.iter)  # (B,Cc,L),(B,1,L),(B,1,L),(B,36,L),(B,)
                c=c.to(self.device, non_blocking=True)
                l=l.to(self.device, non_blocking=True)
                v=v.to(self.device, non_blocking=True)
                y=y.to(self.device, non_blocking=True)
                d=d.to(self.device, non_blocking=True)

                with self.autocast():
                    # style from GT target (학습과 추론 정합의 핵심)
                    with torch.set_grad_enabled(self.cfg.spk_encoder_trainable):
                        spk = self.spk_enc(y)                     # (B,E)
                    y_hat = self.net(c, l, vuv=v, style=spk)      # (B,36,L)

                    # recon
                    L1  = self.l1(y_hat, y)
                    D1  = self.l1(self._delta(y_hat),  self._delta(y))
                    D2  = self.l1(self._delta2(y_hat), self._delta2(y))

                # speaker consistency (FP32/NO-AMP)
                Lspk = self._speaker_cosine(y_hat, y)

                # ---- AgeHead (FP32, NO-AMP) ----
                with torch.amp.autocast(self.device.type, enabled=False):
                    mcep_fake = y_hat.float().to(self.device, non_blocking=True)
                    mcep_real = y.float().to(self.device, non_blocking=True)

                    if epoch == 1 and step == 0:
                        p0 = next(self.age_head.parameters())
                        print("[CHK:init] age_head device:", p0.device, "dtype:", p0.dtype)

                    logit_fake = self.age_head(mcep_fake)   # (B, n_classes)
                    logit_real = self.age_head(mcep_real)   # (B, n_classes)

                # KD (18↔18)
                Lkd = torch.tensor(0.0, device=self.device)
                if (gstep % self.cfg.kd_every) == 0:
                    with torch.no_grad():
                        tprob_fake = self.teacher.predict_probs(y_hat, l).to(self.device)  # (B,18)
                        tprob_real = self.teacher.predict_probs(y,     l).to(self.device)
                    Lkd_fake = kd_loss(logit_fake, tprob_fake, self.cfg.kd_T)
                    Lkd_real = kd_loss(logit_real, tprob_real, self.cfg.kd_T)
                    Lkd = 0.5*(Lkd_fake + Lkd_real)

                # Domain constraint (grouped CE)
                if self.cfg.age_class_groups:
                    Lce = grouped_ce_loss(logit_fake, d, self.cfg.age_class_groups)
                else:
                    Lce = torch.tensor(0.0, device=self.device)

                loss = (L1 + self.cfg.lambda_delta*D1 + self.cfg.lambda_delta2*D2
                        + self.cfg.lambda_spk*Lspk
                        + self.cfg.lambda_age_ce*Lce
                        + self.cfg.lambda_kd*Lkd)

                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.opt); self.scaler.update()

                self.ema.update(self.net)
                mean_loss += float(loss.detach().cpu())

                # ---- 모니터링: y_hat RMS / spk RMS ----
                if self.cfg.log_yhat_rms and (gstep % self.cfg.log_every) == 0:
                    with torch.no_grad():
                        yhat_rms = y_hat.detach().float().pow(2).mean().sqrt().item()
                        spk_rms  = spk.detach().float().pow(2).mean().sqrt().item()
                    print(f"[{now()}][e{epoch:03d} s{gstep:07d}] "
                          f"L1={L1:.3f} d1={D1:.3f} d2={D2:.3f} spk={Lspk:.3f} "
                          f"ageCE={Lce:.3f} kd={Lkd:.3f} | tot={loss:.3f} "
                          f"| yhat_rms={yhat_rms:.4f} spk_rms={spk_rms:.4f}")
                elif (gstep % self.cfg.log_every) == 0:
                    print(f"[{now()}][e{epoch:03d} s{gstep:07d}] "
                          f"L1={L1:.3f} d1={D1:.3f} d2={D2:.3f} spk={Lspk:.3f} "
                          f"ageCE={Lce:.3f} kd={Lkd:.3f} | tot={loss:.3f}")

                if (gstep % self.cfg.save_every_steps) == 0:
                    self.ema.store(self.net); self._save_ckpt(f"step{gstep}"); self.ema.restore(self.net)

            print(f"[{now()}] Epoch {epoch} / {self.cfg.epochs} | mean_loss={mean_loss/self.cfg.steps_per_epoch:.4f}")

        self.ema.store(self.net); self._save_ckpt("final"); print(f"[{now()}] Saved final."); self.ema.restore(self.net)

        # === (NEW) 도메인 평균 스타일 프로토타입 저장 (test.py에서 즉시 활용 가능) ===
        if self.cfg.save_style_prototypes:
            with torch.no_grad():
                self.save_domain_style_prototypes()


# ---------------- main ----------------
if __name__ == "__main__":
    set_seed(1337)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # === 방향성 제약 매핑: 청년(20·30대) vs 중장년(40·50·60대) ===
    # 가정: 18클래스가 10·20·30·40·50·60대를 각 3구간(초/중/후)으로 구성.
    # 인덱스 예시: 10대(0..2), 20대(3..5), 30대(6..8), 40대(9..11), 50대(12..14), 60대(15..17)
    # → 청년(0)=[3..8], 중장년(1)=[9..17]
    AGE_GROUPS = {
        0: list(range(3, 9)),     # 20·30대
        1: list(range(9, 18)),    # 40·50·60대
    }
    print("[AGE_GROUPS] youth(0):", AGE_GROUPS[0], "| middle(1):", AGE_GROUPS[1])

    cfg = ReconConfig(
        device=DEVICE,
        cache_dir="./cache",
        out_dir="checkpoints_recon",
        epochs=500, steps_per_epoch=500, batch_size=4, n_frames=192,
        lr=5e-5, betas=(0.9,0.98), grad_clip=1.0, amp=True,
        lambda_delta=0.5, lambda_delta2=0.25, lambda_spk=1.0,
        lambda_age_ce=0.5, lambda_kd=0.5, kd_T=2.0, kd_every=1,
        keras_model_path="age_model/best_model.h5",
        save_every_steps=5000, log_every=50,
        channels=256, n_resblocks=8, spk_style_dim=128,
        spk_encoder_trainable=False, use_vuv=True,
        age_class_groups=AGE_GROUPS,
    )

    print(f"[{now()}] RUN | device={DEVICE}")
    Trainer(cfg).train()
'''

'''
#순환일관성 포함 - 나이제약,gan 학습 안돰
# ============================ train.py (FULL, FIXED) ============================
# CycleGAN for unpaired VC with:
#  - LSGAN + Cycle L1 + Identity L1 + Δ/Δ² temporal smoothness
#  - Speaker consistency (cosine via SpeakerEncoder; FP32/NO-AMP)
#  - Age constraint (Grouped-CE: youth vs middle/old) via PT head
#  - Knowledge Distillation from Keras teacher (18-way age) with temperature
#  - Fake buffer, EMA for Generators, AMP, linear LR decay, resume
#
# 요구:
#   - model.py: ModelConfig, Generator, Discriminator, SpeakerEncoder, SpkEncCfg
#   - ./cache/coded_sps_A_norm.pickle, coded_sps_B_norm.pickle  (list of (D,T) np.float32)
#   - (선택) ./cache/logf0_seq_A.pickle, logf0_seq_B.pickle     (list of (T,1) np.float32, 무성=0)
#   - age_model/best_model.h5 (또는 SavedModel/.keras)
#
# 실행:
#   python train.py

import os
import time
import math
import random
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ===== 프로젝트 모듈 (네 model.py 기준) =====
from model import ModelConfig, Generator, Discriminator, SpeakerEncoder, SpkEncCfg

# ===== TensorFlow/Keras (teacher for KD) =====
import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


# ---------------- Utilities ----------------
def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def now(): return time.strftime("%H:%M:%S")
def ensure_dir(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------- Dataset (Unpaired AB) ----------------
class UnpairedABDataset(Dataset):
    """
    비병렬 CycleGAN 학습용. A/B에서 독립적으로 랜덤 크롭.
    파일 형식:
      ./cache/coded_sps_A_norm.pickle : list[np.ndarray (D,T)]
      ./cache/coded_sps_B_norm.pickle : list[np.ndarray (D,T)]
      (선택) ./cache/logf0_seq_A.pickle / logf0_seq_B.pickle : list[np.ndarray (T,1)] (무성=0)
    """
    def __init__(self, cache_dir="./cache", n_frames=192, require_f0=True):
        self.cache_dir = cache_dir
        self.L = int(n_frames)
        self.A = self._load_pickle("coded_sps_A_norm.pickle")
        self.B = self._load_pickle("coded_sps_B_norm.pickle")
        self.D = int(self.A[0].shape[0])
        assert self.D == self.B[0].shape[0], "A/B MCEP 차원 불일치"

        self.has_f0 = False
        self.f0A, self.f0B = None, None
        if require_f0:
            fa = os.path.join(cache_dir, "logf0_seq_A.pickle")
            fb = os.path.join(cache_dir, "logf0_seq_B.pickle")
            if os.path.exists(fa) and os.path.exists(fb):
                self.f0A = self._load_pickle("logf0_seq_A.pickle")
                self.f0B = self._load_pickle("logf0_seq_B.pickle")
                self.has_f0 = True

        print(f"[Dataset] A={len(self.A)} B={len(self.B)} | dim={self.D} | seg={self.L} | f0={self.has_f0}")

    def _load_pickle(self, name):
        with open(os.path.join(self.cache_dir, name), "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _padT(arr: np.ndarray, target_T: int) -> np.ndarray:
        if arr is None:
            return None
        T = arr.shape[1] if arr.ndim == 2 else arr.shape[0]
        if T >= target_T: return arr
        need = target_T - T + 1
        L = max(need // 2, 0); R = max(need - L, 0)
        if arr.ndim == 2:  # (D,T)
            return np.pad(arr, ((0, 0), (L, R)), mode="edge")
        else:              # (T,1)
            return np.pad(arr, ((L, R), (0, 0)), mode="edge")

    def _crop_pair(self, y: np.ndarray, f0: Optional[np.ndarray]):
        # y: (D,T), f0: (T,1) or None
        T = y.shape[1]
        if T <= self.L:
            y = self._padT(y, self.L); T = y.shape[1]
            if f0 is not None:
                f0 = self._padT(f0, self.L)
        s = np.random.randint(0, T - self.L); e = s + self.L
        y_out = torch.from_numpy(y[:, s:e].astype(np.float32))  # (D,L)
        if f0 is None:
            return y_out, None
        else:
            f0_out = torch.from_numpy(f0[s:e, :].astype(np.float32)).t()  # (1,L)
            return y_out, f0_out

    def __len__(self):
        return min(len(self.A), len(self.B))

    def __getitem__(self, idx):
        ia = np.random.randint(0, len(self.A))
        ib = np.random.randint(0, len(self.B))
        yA = self.A[ia]; yB = self.B[ib]
        f0A = self.f0A[ia] if (self.has_f0 and ia < len(self.f0A)) else None
        f0B = self.f0B[ib] if (self.has_f0 and ib < len(self.f0B)) else None

        yA, lf0A = self._crop_pair(yA, f0A)  # (D,L), (1,L) or None
        yB, lf0B = self._crop_pair(yB, f0B)
        if lf0A is None: lf0A = torch.zeros(1, yA.shape[1], dtype=torch.float32)
        if lf0B is None: lf0B = torch.zeros(1, yB.shape[1], dtype=torch.float32)
        return yA, lf0A, yB, lf0B


# ---------------- Fake Buffer ----------------
class FakePool1D:
    def __init__(self, pool_size=50):
        self.pool_size = int(pool_size)
        self.data = []

    @torch.no_grad()
    def query(self, x):  # x: (B,D,L)
        out = []
        for i in range(x.size(0)):
            xi = x[i:i+1]
            if len(self.data) < self.pool_size:
                self.data.append(xi.detach().cpu())
                out.append(xi)
            else:
                if random.random() > 0.5:
                    j = random.randint(0, self.pool_size - 1)
                    tmp = self.data[j].clone()
                    self.data[j] = xi.detach().cpu()
                    out.append(tmp.to(x.device))
                else:
                    out.append(xi)
        return torch.cat(out, dim=0)


# ---------------- EMA for Generators ----------------
class EMA:
    def __init__(self, model: nn.Module, decay=0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.back = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def store(self, model: nn.Module):
        self.back = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model: nn.Module):
        if self.back is not None:
            model.load_state_dict(self.back, strict=False)
            self.back = None


# ---------------- Age/KD heads ----------------
class AgeHeadPT(nn.Module):
    def __init__(self, in_dim=36, channels=128, n_blocks=2, n_classes=18):
        super().__init__()
        layers = [nn.Conv1d(in_dim, channels, 5, padding=2), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(n_blocks - 1):
            layers += [nn.Conv1d(channels, channels, 5, padding=2), nn.LeakyReLU(0.2, inplace=True)]
        self.feat = nn.Sequential(*layers)
        self.cls = nn.Linear(channels, n_classes)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, mcep):  # (B,D,L)
        h = self.feat(mcep)
        g = h.mean(dim=2)
        return self.cls(g)  # (B,C)


class KerasTeacher:
    def __init__(self, path: str):
        self.model = self._robust_load(path)
        self.in_dim = int(self.model.input_shape[-1])
        self.n_class = int(self.model.output_shape[-1])
        print(f"[Teacher] loaded | in_dim={self.in_dim} | n_classes={self.n_class}")

    @staticmethod
    def _robust_load(p: str):
        from pathlib import Path
        import zipfile, h5py
        P = Path(p)
        if not P.is_absolute():
            P = (Path(__file__).parent / P).resolve()
        if P.is_dir():
            return tf.keras.models.load_model(P.as_posix())
        if zipfile.is_zipfile(P):
            return tf.keras.models.load_model(P.as_posix())
        with h5py.File(P, "r"): pass
        return tf.keras.models.load_model(P.as_posix(), compile=False)

    @torch.no_grad()
    def probs(self, feat_np: np.ndarray) -> torch.Tensor:
        pr = self.model(feat_np, training=False).numpy()
        pr = np.clip(pr, 1e-8, 1.0)
        pr = pr / pr.sum(axis=1, keepdims=True)
        return torch.from_numpy(pr.astype(np.float32))


def kd_loss(student_logits: torch.Tensor, teacher_probs: torch.Tensor, T: float = 2.0) -> torch.Tensor:
    log_p = F.log_softmax(student_logits / T, dim=1)
    q = teacher_probs / T
    q = q / q.sum(dim=1, keepdim=True)
    return F.kl_div(log_p, q, reduction="batchmean") * (T * T)

def grouped_ce_loss(logits: torch.Tensor, d: torch.Tensor, groups: Dict[int, list]) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)  # (B,C)
    losses = []
    for i in range(logits.size(0)):
        gi = int(d[i].item())
        idxs = groups[gi]
        g = logp[i, idxs]
        losses.append(-(torch.logsumexp(g, dim=0) - math.log(len(idxs))))
    return torch.stack(losses).mean()

@torch.no_grad()
def build_age_stats(mcep: torch.Tensor, lf0: torch.Tensor) -> torch.Tensor:
    """
    [m_mean, m_std, m_skew, f0_mean(voiced), f0_std(voiced), voiced_ratio]
    크기: 3*D + 3
    mcep: (B,D,L), lf0:(B,1,L) (무성=0)
    """
    m = mcep.float()
    f = lf0.float()

    m_mean = m.mean(dim=2)                        # (B,D)
    m_std  = m.std(dim=2).clamp_min(1e-8)         # (B,D)
    m_center = m - m_mean[..., None]
    m_skew = (m_center.pow(3).mean(dim=2) / (m_std.pow(3))).clamp(-1e6, 1e6)

    voiced = (f != 0).float()
    denom = voiced.sum(dim=(1,2), keepdim=True).clamp_min(1e-6)
    f_mean = ((f * voiced).sum(dim=(1,2), keepdim=True) / denom).squeeze(-1)  # (B,1)
    f_var  = (((f - f_mean)**2 * voiced).sum(dim=(1,2), keepdim=True) / denom).squeeze(-1)
    f_std  = torch.sqrt(f_var + 1e-8)
    v_rate = voiced.mean(dim=(1,2), keepdim=True).squeeze(-1)

    feat = torch.cat([m_mean, m_std, m_skew, f_mean, f_std, v_rate], dim=1)  # (B, 3D+3)
    return feat

def adapt_dim(feat: torch.Tensor, target_dim: int) -> torch.Tensor:
    B, D = feat.shape
    if D == target_dim: return feat
    if D < target_dim:
        pad = feat.new_zeros(B, target_dim - D)
        return torch.cat([feat, pad], dim=1)
    return feat[:, :target_dim]


# ---------------- Config ----------------
@dataclass
class TrainCfg:
    cache_dir: str = "./cache"
    out_dir: str = "checkpoints_cycle_full"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337

    # data
    n_frames: int = 192
    batch_size: int = 4
    num_workers: int = 0
    pin_memory: bool = True
    use_f0_for_kd: bool = True

    # model
    base_channels: int = 128
    n_resblocks: int = 6

    # opt
    lr: float = 2e-4
    betas: Tuple[float, float] = (0.5, 0.999)
    epochs: int = 300
    lr_decay_from: int = 150
    amp: bool = True
    grad_clip: float = 5.0
    ema_decay: float = 0.999

    # loss weights
    lambda_cyc: float = 10.0
    lambda_id:  float = 5.0
    lambda_delta: float = 0.5
    lambda_delta2: float = 0.25
    lambda_spk: float = 1.0
    lambda_age_ce: float = 0.5
    lambda_kd: float = 0.5
    kd_T: float = 2.0
    kd_every: int = 1

    # teacher
    keras_model_path: str = "age_model/best_model.h5"

    # age groups (domain A=0: 청년, domain B=1: 중장년)
    age_class_groups: Optional[Dict[int, list]] = None

    # logging / ckpt
    log_every: int = 50
    save_every: int = 5000
    resume: Optional[str] = None  # ckpt path


# ---------------- Trainer ----------------
class Trainer:
    def __init__(self, cfg: TrainCfg):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        ensure_dir(cfg.out_dir)

        # data
        self.ds = UnpairedABDataset(cfg.cache_dir, n_frames=cfg.n_frames, require_f0=cfg.use_f0_for_kd)
        self.loader = DataLoader(
            self.ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
            num_workers=cfg.num_workers, pin_memory=(cfg.pin_memory and self.device.type == "cuda")
        )

        Dm = self.ds.D
        # models
        gcfg = ModelConfig()
        gcfg.in_dim = Dm
        gcfg.base_channels = cfg.base_channels
        gcfg.n_resblocks = cfg.n_resblocks
        gcfg.multiscale_D = False

        self.G_A2B = Generator(gcfg).to(self.device)
        self.G_B2A = Generator(gcfg).to(self.device)
        self.D_A   = Discriminator(gcfg).to(self.device)
        self.D_B   = Discriminator(gcfg).to(self.device)

        # EMA (Generators)
        self.ema_A2B = EMA(self.G_A2B, decay=cfg.ema_decay)
        self.ema_B2A = EMA(self.G_B2A, decay=cfg.ema_decay)

        # Speaker encoder (동결)
        spk_cfg = SpkEncCfg(in_dim=Dm, out_dim=128)
        self.spk = SpeakerEncoder(spk_cfg).to(self.device)
        for p in self.spk.parameters(): p.requires_grad_(False)
        self.spk.eval()

        # Age head (student) + Keras teacher
        self.teacher = KerasTeacher(cfg.keras_model_path)
        self.age_head = AgeHeadPT(in_dim=Dm, channels=128, n_blocks=2, n_classes=self.teacher.n_class).to(self.device).float()

        # opt / sched
        self.opt_G = torch.optim.Adam(
            list(self.G_A2B.parameters()) + list(self.G_B2A.parameters()) + list(self.age_head.parameters()),
            lr=cfg.lr, betas=cfg.betas
        )
        self.opt_D = torch.optim.Adam(
            list(self.D_A.parameters()) + list(self.D_B.parameters()),
            lr=cfg.lr, betas=cfg.betas
        )

        def lr_lambda(epoch):
            if epoch < cfg.lr_decay_from: return 1.0
            t = epoch - cfg.lr_decay_from
            T = max(cfg.epochs - cfg.lr_decay_from, 1)
            return max(0.0, 1.0 - t / T)
        self.sch_G = torch.optim.lr_scheduler.LambdaLR(self.opt_G, lr_lambda=lr_lambda)
        self.sch_D = torch.optim.lr_scheduler.LambdaLR(self.opt_D, lr_lambda=lr_lambda)

        # AMP
        try:
            from torch.amp import autocast as _autocast, GradScaler as _GradScaler
            self.autocast = lambda: _autocast(self.device.type, enabled=self.cfg.amp)
            self.scaler_G = _GradScaler(enabled=self.cfg.amp)
            self.scaler_D = _GradScaler(enabled=self.cfg.amp)
        except Exception:
            self.autocast = lambda: torch.cuda.amp.autocast(enabled=self.cfg.amp)
            self.scaler_G = torch.cuda.amp.GradScaler(enabled=self.cfg.amp)
            self.scaler_D = torch.cuda.amp.GradScaler(enabled=self.cfg.amp)

        # loss & pools
        self.l1 = nn.L1Loss()
        self.pool_A = FakePool1D(50)
        self.pool_B = FakePool1D(50)

        print(f"[Init] G_A2B={count_params(self.G_A2B):,} | G_B2A={count_params(self.G_B2A):,} "
              f"| D_A={count_params(self.D_A):,} | D_B={count_params(self.D_B):,} "
              f"| AgeHead={count_params(self.age_head):,} | MCEP_dim={Dm}")

        # resume
        if cfg.resume and os.path.exists(cfg.resume):
            self._load_ckpt(cfg.resume)
            print(f"[{now()}] Resumed from {cfg.resume}")

    # ---- Helpers ----
    def _delta(self, x):  return x[:, :, 1:] - x[:, :, :-1]
    def _delta2(self, x): return self._delta(self._delta(x))

    def _spk_cos(self, a, b):
        with torch.amp.autocast(self.device.type, enabled=False):
            ea = F.normalize(self.spk(a.float()), dim=1)
            eb = F.normalize(self.spk(b.float()), dim=1)
            return (1.0 - (ea * eb).sum(dim=1)).mean()

    # LSGAN
    def d_loss(self, D, real, fake):
        pr = D(real)
        pf = D(fake.detach())
        return 0.5 * (torch.mean((pr - 1.0) ** 2) + torch.mean(pf ** 2))

    def g_loss(self, D, fake):
        pf = D(fake)
        return 0.5 * torch.mean((pf - 1.0) ** 2)

    # ckpt
    def _save_ckpt(self, tag):
        path = os.path.join(self.cfg.out_dir, f"ckpt_{tag}.pth")
        torch.save({
            "G_A2B": self.G_A2B.state_dict(),
            "G_B2A": self.G_B2A.state_dict(),
            "D_A":   self.D_A.state_dict(),
            "D_B":   self.D_B.state_dict(),
            "AgeHead": self.age_head.state_dict(),
            "opt_G": self.opt_G.state_dict(),
            "opt_D": self.opt_D.state_dict(),
            "sch_G": self.sch_G.state_dict(),
            "sch_D": self.sch_D.state_dict(),
        }, path)
        print(f"[{now()}] 💾 Saved: {path}")

    def _load_ckpt(self, path):
        ck = torch.load(path, map_location=self.device)
        self.G_A2B.load_state_dict(ck["G_A2B"], strict=False)
        self.G_B2A.load_state_dict(ck["G_B2A"], strict=False)
        self.D_A.load_state_dict(ck["D_A"], strict=False)
        self.D_B.load_state_dict(ck["D_B"], strict=False)
        self.age_head.load_state_dict(ck["AgeHead"], strict=False)
        self.opt_G.load_state_dict(ck["opt_G"])
        self.opt_D.load_state_dict(ck["opt_D"])
        if "sch_G" in ck: self.sch_G.load_state_dict(ck["sch_G"])
        if "sch_D" in ck: self.sch_D.load_state_dict(ck["sch_D"])

    # ---- Train ----
    def train(self):
        gstep = 0
        for epoch in range(1, self.cfg.epochs + 1):
            for yA, lf0A, yB, lf0B in self.loader:
                gstep += 1
                yA = yA.to(self.device, non_blocking=True)      # (B,D,L)
                yB = yB.to(self.device, non_blocking=True)
                lf0A = lf0A.to(self.device, non_blocking=True)  # (B,1,L)
                lf0B = lf0B.to(self.device, non_blocking=True)

                # ======== G step ========
                self.opt_G.zero_grad(set_to_none=True)
                with self.autocast():
                    # forward
                    fake_B = self.G_A2B(yA)      # A->B
                    rec_A  = self.G_B2A(fake_B)  # A->B->A
                    fake_A = self.G_B2A(yB)      # B->A
                    rec_B  = self.G_A2B(fake_A)  # B->A->B

                    # identity
                    idt_A = self.G_B2A(yA)
                    idt_B = self.G_A2B(yB)

                    # GAN for G
                    g_gan = self.g_loss(self.D_B, fake_B) + self.g_loss(self.D_A, fake_A)

                    # Cycle L1
                    g_cyc = self.l1(rec_A, yA) + self.l1(rec_B, yB)

                    # Identity L1
                    g_id  = self.l1(idt_A, yA) + self.l1(idt_B, yB)

                    # Δ/Δ² smoothness (변수명 충돌 방지: sm1/sm2)
                    sm1 = self.l1(self._delta(rec_A),  self._delta(yA)) + \
                          self.l1(self._delta(rec_B),  self._delta(yB))
                    sm2 = self.l1(self._delta2(rec_A), self._delta2(yA)) + \
                          self.l1(self._delta2(rec_B), self._delta2(yB))

                # Speaker consistency (FP32, NO-AMP)
                Lspk = self._spk_cos(yA, fake_B) + self._spk_cos(yB, fake_A)

                # Age + KD (FP32, NO-AMP)
                with torch.amp.autocast(self.device.type, enabled=False):
                    logit_fakeB = self.age_head(fake_B.float())
                    logit_fakeA = self.age_head(fake_A.float())

                    # 도메인 라벨 (변수명 충돌 방지: domA/domB)
                    domB = torch.full((yA.size(0),), 1, dtype=torch.long, device=self.device)  # fakeB -> 중장년(1)
                    domA = torch.full((yB.size(0),), 0, dtype=torch.long, device=self.device)  # fakeA -> 청년(0)

                    if self.cfg.age_class_groups:
                        Lce = grouped_ce_loss(logit_fakeB, domB, self.cfg.age_class_groups) + \
                              grouped_ce_loss(logit_fakeA, domA, self.cfg.age_class_groups)
                    else:
                        Lce = torch.tensor(0.0, device=self.device)

                    # KD 준비: teacher feats
                    feat_fakeB = build_age_stats(fake_B, lf0A)   # (B,3D+3)
                    feat_fakeA = build_age_stats(fake_A, lf0B)
                    feat_realA = build_age_stats(yA,   lf0A)
                    feat_realB = build_age_stats(yB,   lf0B)

                    tdim = self.teacher.in_dim
                    f_fakeB = adapt_dim(feat_fakeB, tdim).cpu().numpy()
                    f_fakeA = adapt_dim(feat_fakeA, tdim).cpu().numpy()
                    f_realA = adapt_dim(feat_realA, tdim).cpu().numpy()
                    f_realB = adapt_dim(feat_realB, tdim).cpu().numpy()

                    with torch.no_grad():
                        tprob_fakeB = self.teacher.probs(f_fakeB).to(self.device)
                        tprob_fakeA = self.teacher.probs(f_fakeA).to(self.device)
                        tprob_realA = self.teacher.probs(f_realA).to(self.device)
                        tprob_realB = self.teacher.probs(f_realB).to(self.device)

                    Lkd = 0.25 * (
                        kd_loss(logit_fakeB, tprob_fakeB, self.cfg.kd_T) +
                        kd_loss(logit_fakeA, tprob_fakeA, self.cfg.kd_T) +
                        kd_loss(self.age_head(yA.float()), tprob_realA, self.cfg.kd_T) +
                        kd_loss(self.age_head(yB.float()), tprob_realB, self.cfg.kd_T)
                    )

                G_total = (g_gan + self.cfg.lambda_cyc * g_cyc + self.cfg.lambda_id * g_id +
                           self.cfg.lambda_delta * sm1 + self.cfg.lambda_delta2 * sm2 +
                           self.cfg.lambda_spk * Lspk + self.cfg.lambda_age_ce * Lce + self.cfg.lambda_kd * Lkd)

                self.scaler_G.scale(G_total).backward()
                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    self.scaler_G.unscale_(self.opt_G)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.G_A2B.parameters()) + list(self.G_B2A.parameters()) + list(self.age_head.parameters()),
                        self.cfg.grad_clip
                    )
                self.scaler_G.step(self.opt_G); self.scaler_G.update()

                # EMA update
                self.ema_A2B.update(self.G_A2B)
                self.ema_B2A.update(self.G_B2A)

                # ======== D step ========
                with torch.no_grad():
                    fake_A_pool = self.pool_A.query(fake_A)
                    fake_B_pool = self.pool_B.query(fake_B)

                self.opt_D.zero_grad(set_to_none=True)
                with self.autocast():
                    dA_loss = self.d_loss(self.D_A, yA, fake_A_pool)
                    dB_loss = self.d_loss(self.D_B, yB, fake_B_pool)
                    D_total = dA_loss + dB_loss
                self.scaler_D.scale(D_total).backward()
                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    self.scaler_D.unscale_(self.opt_D)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.D_A.parameters()) + list(self.D_B.parameters()),
                        self.cfg.grad_clip
                    )
                self.scaler_D.step(self.opt_D); self.scaler_D.update()

                # ---- log ----
                if gstep % self.cfg.log_every == 0:
                    with torch.no_grad():
                        rms_fake = 0.5 * (fake_A.float().pow(2).mean().sqrt() + fake_B.float().pow(2).mean().sqrt()).item()
                    print(f"[{now()}][e{epoch:03d} s{gstep:07d}] "
                          f"G: gan={float(g_gan):.3f} cyc={float(g_cyc):.3f} id={float(g_id):.3f} "
                          f"sm1={float(sm1):.3f} sm2={float(sm2):.3f} "
                          f"| spk={float(Lspk):.3f} ageCE={float(Lce):.3f} kd={float(Lkd):.3f} "
                          f"| D={float(D_total):.3f} | rms={rms_fake:.4f}")

                if gstep % self.cfg.save_every == 0:
                    # EMA 가중치로 안전 저장
                    self.ema_A2B.store(self.G_A2B); self.ema_B2A.store(self.G_B2A)
                    self._save_ckpt(f"step{gstep}")
                    self.ema_A2B.restore(self.G_A2B); self.ema_B2A.restore(self.G_B2A)

            # epoch end
            self.sch_G.step(); self.sch_D.step()
            print(f"[{now()}] Epoch {epoch}/{self.cfg.epochs} | LR_G={self.sch_G.get_last_lr()[0]:.6f}")
        # final
        self.ema_A2B.store(self.G_A2B); self.ema_B2A.store(self.G_B2A)
        self._save_ckpt("final")
        self.ema_A2B.restore(self.G_A2B); self.ema_B2A.restore(self.G_B2A)
        print(f"[{now()}] ✅ Training finished.")


# ---------------- main ----------------
if __name__ == "__main__":
    set_seed(1337)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 18클래스 가정 예시:
    #   10대(0..2), 20대(3..5), 30대(6..8), 40대(9..11), 50대(12..14), 60대(15..17)
    #   청년(0)=[3..8], 중장년(1)=[9..17]
    AGE_GROUPS = {
        0: list(range(3, 9)),   # youth  (B->A 목표)
        1: list(range(9, 18)),  # middle (A->B 목표)
    }
    print("[AGE_GROUPS] youth(0):", AGE_GROUPS[0], "| middle(1):", AGE_GROUPS[1])

    cfg = TrainCfg(
        device=DEVICE,
        cache_dir="./cache",
        out_dir="checkpoints_cycle_full",
        n_frames=192, batch_size=4,
        epochs=300, lr=2e-4, betas=(0.5, 0.999),
        lr_decay_from=150, amp=True, grad_clip=1.0, ema_decay=0.999,
        lambda_cyc=10.0, lambda_id=5.0, lambda_delta=0.5, lambda_delta2=0.25,
        lambda_spk=1.0, lambda_age_ce=0.5, lambda_kd=0.5, kd_T=2.0, kd_every=1,
        keras_model_path="age_model/best_model.h5",
        age_class_groups=AGE_GROUPS,
        log_every=50, save_every=5000,
        resume=None,
    )

    ensure_dir(cfg.out_dir)
    print(f"[{now()}] RUN CycleGAN++ | device={cfg.device}")
    Trainer(cfg).train()
'''

# ============================ train.py (FULL, PATCHED) ============================
# CycleGAN for unpaired VC with:
#  - LSGAN + Cycle L1 + Identity L1 + Δ/Δ² temporal smoothness
#  - Speaker consistency (cosine via SpeakerEncoder; FP32/NO-AMP)
#  - Age constraint (Grouped-CE: youth vs middle/old) via PT head
#  - Knowledge Distillation from Keras teacher (18-way age) with temperature
#  - Fake buffer, EMA for Generators, AMP, linear LR decay, resume
#  - ✔ TTUR(G/D 서로 다른 LR) + label smoothing + instance noise + D-skip
#  - ✔ ageCE warmup (초반엔 G로 전파 차단)
#
# 요구:
#   - model.py: ModelConfig, Generator, Discriminator, SpeakerEncoder, SpkEncCfg
#   - ./cache/coded_sps_A_norm.pickle, coded_sps_B_norm.pickle  (list of (D,T) np.float32)
#   - (선택) ./cache/logf0_seq_A.pickle, logf0_seq_B.pickle     (list of (T,1) np.float32, 무성=0)
#   - age_model/best_model.h5 (또는 SavedModel/.keras)
#
# 실행:
#   python train.py

import os
import time
import math
import random
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ===== 프로젝트 모듈 (네 model.py 기준) =====
from model import ModelConfig, Generator, Discriminator, SpeakerEncoder, SpkEncCfg

# ===== TensorFlow/Keras (teacher for KD) =====
import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


# ---------------- Utilities ----------------
def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def now(): return time.strftime("%H:%M:%S")
def ensure_dir(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------- Dataset (Unpaired AB) ----------------
class UnpairedABDataset(Dataset):
    """
    비병렬 CycleGAN 학습용. A/B에서 독립적으로 랜덤 크롭.
    파일 형식:
      ./cache/coded_sps_A_norm.pickle : list[np.ndarray (D,T)]
      ./cache/coded_sps_B_norm.pickle : list[np.ndarray (D,T)]
      (선택) ./cache/logf0_seq_A.pickle / logf0_seq_B.pickle : list[np.ndarray (T,1)] (무성=0)
    """
    def __init__(self, cache_dir="./cache", n_frames=192, require_f0=True):
        self.cache_dir = cache_dir
        self.L = int(n_frames)
        self.A = self._load_pickle("coded_sps_A_norm.pickle")
        self.B = self._load_pickle("coded_sps_B_norm.pickle")
        self.D = int(self.A[0].shape[0])
        assert self.D == self.B[0].shape[0], "A/B MCEP 차원 불일치"

        self.has_f0 = False
        self.f0A, self.f0B = None, None
        if require_f0:
            fa = os.path.join(cache_dir, "logf0_seq_A.pickle")
            fb = os.path.join(cache_dir, "logf0_seq_B.pickle")
            if os.path.exists(fa) and os.path.exists(fb):
                self.f0A = self._load_pickle("logf0_seq_A.pickle")
                self.f0B = self._load_pickle("logf0_seq_B.pickle")
                self.has_f0 = True

        print(f"[Dataset] A={len(self.A)} B={len(self.B)} | dim={self.D} | seg={self.L} | f0={self.has_f0}")

    def _load_pickle(self, name):
        with open(os.path.join(self.cache_dir, name), "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _padT(arr: np.ndarray, target_T: int) -> np.ndarray:
        if arr is None:
            return None
        T = arr.shape[1] if arr.ndim == 2 else arr.shape[0]
        if T >= target_T: return arr
        need = target_T - T + 1
        L = max(need // 2, 0); R = max(need - L, 0)
        if arr.ndim == 2:  # (D,T)
            return np.pad(arr, ((0, 0), (L, R)), mode="edge")
        else:              # (T,1)
            return np.pad(arr, ((L, R), (0, 0)), mode="edge")

    def _crop_pair(self, y: np.ndarray, f0: Optional[np.ndarray]):
        # y: (D,T), f0: (T,1) or None
        T = y.shape[1]
        if T <= self.L:
            y = self._padT(y, self.L); T = y.shape[1]
            if f0 is not None:
                f0 = self._padT(f0, self.L)
        s = np.random.randint(0, T - self.L); e = s + self.L
        y_out = torch.from_numpy(y[:, s:e].astype(np.float32))  # (D,L)
        if f0 is None:
            return y_out, None
        else:
            f0_out = torch.from_numpy(f0[s:e, :].astype(np.float32)).t()  # (1,L)
            return y_out, f0_out

    def __len__(self):
        return min(len(self.A), len(self.B))

    def __getitem__(self, idx):
        ia = np.random.randint(0, len(self.A))
        ib = np.random.randint(0, len(self.B))
        yA = self.A[ia]; yB = self.B[ib]
        f0A = self.f0A[ia] if (self.has_f0 and ia < len(self.f0A)) else None
        f0B = self.f0B[ib] if (self.has_f0 and ib < len(self.f0B)) else None

        yA, lf0A = self._crop_pair(yA, f0A)  # (D,L), (1,L) or None
        yB, lf0B = self._crop_pair(yB, f0B)
        if lf0A is None: lf0A = torch.zeros(1, yA.shape[1], dtype=torch.float32)
        if lf0B is None: lf0B = torch.zeros(1, yB.shape[1], dtype=torch.float32)
        return yA, lf0A, yB, lf0B


# ---------------- Fake Buffer ----------------
class FakePool1D:
    def __init__(self, pool_size=50):
        self.pool_size = int(pool_size)
        self.data = []

    @torch.no_grad()
    def query(self, x):  # x: (B,D,L)
        out = []
        for i in range(x.size(0)):
            xi = x[i:i+1]
            if len(self.data) < self.pool_size:
                self.data.append(xi.detach().cpu())
                out.append(xi)
            else:
                if random.random() > 0.5:
                    j = random.randint(0, self.pool_size - 1)
                    tmp = self.data[j].clone()
                    self.data[j] = xi.detach().cpu()
                    out.append(tmp.to(x.device))
                else:
                    out.append(xi)
        return torch.cat(out, dim=0)


# ---------------- EMA for Generators ----------------
class EMA:
    def __init__(self, model: nn.Module, decay=0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.back = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def store(self, model: nn.Module):
        self.back = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model: nn.Module):
        if self.back is not None:
            model.load_state_dict(self.back, strict=False)
            self.back = None


# ---------------- Age/KD heads ----------------
class AgeHeadPT(nn.Module):
    def __init__(self, in_dim=36, channels=128, n_blocks=2, n_classes=18):
        super().__init__()
        layers = [nn.Conv1d(in_dim, channels, 5, padding=2), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(n_blocks - 1):
            layers += [nn.Conv1d(channels, channels, 5, padding=2), nn.LeakyReLU(0.2, inplace=True)]
        self.feat = nn.Sequential(*layers)
        self.cls = nn.Linear(channels, n_classes)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, mcep):  # (B,D,L)
        h = self.feat(mcep)
        g = h.mean(dim=2)
        return self.cls(g)  # (B,C)


class KerasTeacher:
    def __init__(self, path: str):
        self.model = self._robust_load(path)
        self.in_dim = int(self.model.input_shape[-1])
        self.n_class = int(self.model.output_shape[-1])
        print(f"[Teacher] loaded | in_dim={self.in_dim} | n_classes={self.n_class}")

    @staticmethod
    def _robust_load(p: str):
        from pathlib import Path
        import zipfile, h5py
        P = Path(p)
        if not P.is_absolute():
            P = (Path(__file__).parent / P).resolve()
        if P.is_dir():
            return tf.keras.models.load_model(P.as_posix())
        if zipfile.is_zipfile(P):
            return tf.keras.models.load_model(P.as_posix())
        with h5py.File(P, "r"): pass
        return tf.keras.models.load_model(P.as_posix(), compile=False)

    @torch.no_grad()
    def probs(self, feat_np: np.ndarray) -> torch.Tensor:
        pr = self.model(feat_np, training=False).numpy()
        pr = np.clip(pr, 1e-8, 1.0)
        pr = pr / pr.sum(axis=1, keepdims=True)
        return torch.from_numpy(pr.astype(np.float32))


def kd_loss(student_logits: torch.Tensor, teacher_probs: torch.Tensor, T: float = 2.0) -> torch.Tensor:
    log_p = F.log_softmax(student_logits / T, dim=1)
    q = teacher_probs / T
    q = q / q.sum(dim=1, keepdim=True)
    return F.kl_div(log_p, q, reduction="batchmean") * (T * T)

def grouped_ce_loss(logits: torch.Tensor, d: torch.Tensor, groups: Dict[int, list]) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)  # (B,C)
    losses = []
    for i in range(logits.size(0)):
        gi = int(d[i].item())
        idxs = groups[gi]
        g = logp[i, idxs]
        losses.append(-(torch.logsumexp(g, dim=0) - math.log(len(idxs))))
    return torch.stack(losses).mean()

@torch.no_grad()
def build_age_stats(mcep: torch.Tensor, lf0: torch.Tensor) -> torch.Tensor:
    """
    [m_mean, m_std, m_skew, f0_mean(voiced), f0_std(voiced), voiced_ratio]
    크기: 3*D + 3
    mcep: (B,D,L), lf0:(B,1,L) (무성=0)
    """
    m = mcep.float()
    f = lf0.float()

    m_mean = m.mean(dim=2)                        # (B,D)
    m_std  = m.std(dim=2).clamp_min(1e-8)         # (B,D)
    m_center = m - m_mean[..., None]
    m_skew = (m_center.pow(3).mean(dim=2) / (m_std.pow(3))).clamp(-1e6, 1e6)

    voiced = (f != 0).float()
    denom = voiced.sum(dim=(1,2), keepdim=True).clamp_min(1e-6)
    f_mean = ((f * voiced).sum(dim=(1,2), keepdim=True) / denom).squeeze(-1)  # (B,1)
    f_var  = (((f - f_mean)**2 * voiced).sum(dim=(1,2), keepdim=True) / denom).squeeze(-1)
    f_std  = torch.sqrt(f_var + 1e-8)
    v_rate = voiced.mean(dim=(1,2), keepdim=True).squeeze(-1)

    feat = torch.cat([m_mean, m_std, m_skew, f_mean, f_std, v_rate], dim=1)  # (B, 3D+3)
    return feat

def adapt_dim(feat: torch.Tensor, target_dim: int) -> torch.Tensor:
    B, D = feat.shape
    if D == target_dim: return feat
    if D < target_dim:
        pad = feat.new_zeros(B, target_dim - D)
        return torch.cat([feat, pad], dim=1)
    return feat[:, :target_dim]


# ---------------- Config ----------------
@dataclass
class TrainCfg:
    cache_dir: str = "./cache"
    out_dir: str = "checkpoints_cycle_full"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337

    # data
    n_frames: int = 192
    batch_size: int = 4
    num_workers: int = 0
    pin_memory: bool = True
    use_f0_for_kd: bool = True

    # model
    base_channels: int = 128
    n_resblocks: int = 6

    # opt
    lr: float = 2e-4                    # (보존: 하위 호환용, 사용 안 함)
    betas: Tuple[float, float] = (0.5, 0.999)
    # ✔ TTUR
    lr_G: float = 2e-4                  # Generator LR
    lr_D: float = 1e-4                  # Discriminator LR (더 낮게)
    epochs: int = 300
    lr_decay_from: int = 150
    amp: bool = True
    grad_clip: float = 5.0
    ema_decay: float = 0.999

    # loss weights
    lambda_cyc: float = 10.0
    lambda_id:  float = 5.0
    lambda_delta: float = 0.5
    lambda_delta2: float = 0.25
    lambda_spk: float = 1.0
    lambda_age_ce: float = 0.5
    lambda_kd: float = 0.5
    kd_T: float = 2.0
    kd_every: int = 1

    # teacher
    keras_model_path: str = "age_model/best_model.h5"

    # age groups (domain A=0: 청년, domain B=1: 중장년)
    age_class_groups: Optional[Dict[int, list]] = None

    # logging / ckpt
    log_every: int = 50
    save_every: int = 5000
    resume: Optional[str] = None  # ckpt path

    # ✔ instance noise & D-skip
    inst_warm_steps: int = 5000     # 노이즈 감쇠 종료 step
    inst_sigma0: float = 0.1        # 초기 표준편차
    d_skip_thresh: float = 0.05     # D_total이 이보다 작고 짝수 step이면 스킵
    d_skip_even_only: bool = True

    # ✔ ageCE warmup
    age_warmup_steps: int = 10000


# ---------------- Trainer ----------------
class Trainer:
    def __init__(self, cfg: TrainCfg):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        ensure_dir(cfg.out_dir)

        # data
        self.ds = UnpairedABDataset(cfg.cache_dir, n_frames=cfg.n_frames, require_f0=cfg.use_f0_for_kd)
        self.loader = DataLoader(
            self.ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
            num_workers=cfg.num_workers, pin_memory=(cfg.pin_memory and self.device.type == "cuda")
        )

        Dm = self.ds.D
        # models
        gcfg = ModelConfig()
        gcfg.in_dim = Dm
        gcfg.base_channels = cfg.base_channels
        gcfg.n_resblocks = cfg.n_resblocks
        gcfg.multiscale_D = False

        self.G_A2B = Generator(gcfg).to(self.device)
        self.G_B2A = Generator(gcfg).to(self.device)
        self.D_A   = Discriminator(gcfg).to(self.device)
        self.D_B   = Discriminator(gcfg).to(self.device)

        # EMA (Generators)
        self.ema_A2B = EMA(self.G_A2B, decay=cfg.ema_decay)
        self.ema_B2A = EMA(self.G_B2A, decay=cfg.ema_decay)

        # Speaker encoder (동결)
        spk_cfg = SpkEncCfg(in_dim=Dm, out_dim=128)
        self.spk = SpeakerEncoder(spk_cfg).to(self.device)
        for p in self.spk.parameters(): p.requires_grad_(False)
        self.spk.eval()

        # Age head (student) + Keras teacher
        self.teacher = KerasTeacher(cfg.keras_model_path)
        self.age_head = AgeHeadPT(in_dim=Dm, channels=128, n_blocks=2, n_classes=self.teacher.n_class).to(self.device).float()

        # opt / sched  ✔ TTUR 사용
        self.opt_G = torch.optim.Adam(
            list(self.G_A2B.parameters()) + list(self.G_B2A.parameters()) + list(self.age_head.parameters()),
            lr=cfg.lr_G, betas=cfg.betas
        )
        self.opt_D = torch.optim.Adam(
            list(self.D_A.parameters()) + list(self.D_B.parameters()),
            lr=cfg.lr_D, betas=cfg.betas
        )

        def lr_lambda(epoch):
            if epoch < cfg.lr_decay_from: return 1.0
            t = epoch - cfg.lr_decay_from
            T = max(cfg.epochs - cfg.lr_decay_from, 1)
            return max(0.0, 1.0 - t / T)
        self.sch_G = torch.optim.lr_scheduler.LambdaLR(self.opt_G, lr_lambda=lr_lambda)
        self.sch_D = torch.optim.lr_scheduler.LambdaLR(self.opt_D, lr_lambda=lr_lambda)

        # AMP
        try:
            from torch.amp import autocast as _autocast, GradScaler as _GradScaler
            self.autocast = lambda: _autocast(self.device.type, enabled=self.cfg.amp)
            self.scaler_G = _GradScaler(enabled=self.cfg.amp)
            self.scaler_D = _GradScaler(enabled=self.cfg.amp)
        except Exception:
            self.autocast = lambda: torch.cuda.amp.autocast(enabled=self.cfg.amp)
            self.scaler_G = torch.cuda.amp.GradScaler(enabled=self.cfg.amp)
            self.scaler_D = torch.cuda.amp.GradScaler(enabled=self.cfg.amp)

        # loss & pools
        self.l1 = nn.L1Loss()
        self.pool_A = FakePool1D(50)
        self.pool_B = FakePool1D(50)

        print(f"[Init] G_A2B={count_params(self.G_A2B):,} | G_B2A={count_params(self.G_B2A):,} "
              f"| D_A={count_params(self.D_A):,} | D_B={count_params(self.D_B):,} "
              f"| AgeHead={count_params(self.age_head):,} | MCEP_dim={Dm}")

        # resume
        if cfg.resume and os.path.exists(cfg.resume):
            self._load_ckpt(cfg.resume)
            print(f"[{now()}] Resumed from {cfg.resume}]")

    # ---- Helpers ----
    def _delta(self, x):  return x[:, :, 1:] - x[:, :, :-1]
    def _delta2(self, x): return self._delta(self._delta(x))

    def _spk_cos(self, a, b):
        with torch.amp.autocast(self.device.type, enabled=False):
            ea = F.normalize(self.spk(a.float()), dim=1)
            eb = F.normalize(self.spk(b.float()), dim=1)
            return (1.0 - (ea * eb).sum(dim=1)).mean()

    # ✔ instance noise schedule
    def _inst_noise_sigma(self, step):
        r = max(0.0, 1.0 - step / float(self.cfg.inst_warm_steps))
        return self.cfg.inst_sigma0 * r

    def _add_noise(self, x, sigma):
        if sigma <= 0.0:
            return x
        n = torch.randn_like(x) * sigma
        return x + n

    # ✔ LSGAN with label smoothing
    def d_loss(self, D, real, fake, real_t=0.9, fake_t=0.1):
        pr = D(real)
        pf = D(fake.detach())
        return 0.5 * (torch.mean((pr - real_t) ** 2) + torch.mean((pf - fake_t) ** 2))

    def g_loss(self, D, fake, real_t=0.9):
        pf = D(fake)
        return 0.5 * torch.mean((pf - real_t) ** 2)

    # ckpt
    def _save_ckpt(self, tag):
        path = os.path.join(self.cfg.out_dir, f"ckpt_{tag}.pth")
        torch.save({
            "G_A2B": self.G_A2B.state_dict(),
            "G_B2A": self.G_B2A.state_dict(),
            "D_A":   self.D_A.state_dict(),
            "D_B":   self.D_B.state_dict(),
            "AgeHead": self.age_head.state_dict(),
            "opt_G": self.opt_G.state_dict(),
            "opt_D": self.opt_D.state_dict(),
            "sch_G": self.sch_G.state_dict(),
            "sch_D": self.sch_D.state_dict(),
        }, path)
        print(f"[{now()}] 💾 Saved: {path}")

    def _load_ckpt(self, path):
        ck = torch.load(path, map_location=self.device)
        self.G_A2B.load_state_dict(ck["G_A2B"], strict=False)
        self.G_B2A.load_state_dict(ck["G_B2A"], strict=False)
        self.D_A.load_state_dict(ck["D_A"], strict=False)
        self.D_B.load_state_dict(ck["D_B"], strict=False)
        self.age_head.load_state_dict(ck["AgeHead"], strict=False)
        self.opt_G.load_state_dict(ck["opt_G"])
        self.opt_D.load_state_dict(ck["opt_D"])
        if "sch_G" in ck: self.sch_G.load_state_dict(ck["sch_G"])
        if "sch_D" in ck: self.sch_D.load_state_dict(ck["sch_D"])

    # ---- Train ----
    def train(self):
        gstep = 0
        for epoch in range(1, self.cfg.epochs + 1):
            for yA, lf0A, yB, lf0B in self.loader:
                gstep += 1
                yA = yA.to(self.device, non_blocking=True)      # (B,D,L)
                yB = yB.to(self.device, non_blocking=True)
                lf0A = lf0A.to(self.device, non_blocking=True)  # (B,1,L)
                lf0B = lf0B.to(self.device, non_blocking=True)

                # ======== G step ========
                self.opt_G.zero_grad(set_to_none=True)
                with self.autocast():
                    # forward
                    fake_B = self.G_A2B(yA)      # A->B
                    rec_A  = self.G_B2A(fake_B)  # A->B->A
                    fake_A = self.G_B2A(yB)      # B->A
                    rec_B  = self.G_A2B(fake_A)  # B->A->B

                    # identity
                    idt_A = self.G_B2A(yA)
                    idt_B = self.G_A2B(yB)

                    # GAN for G  ✔ smoothing
                    g_gan = self.g_loss(self.D_B, fake_B, real_t=0.9) + self.g_loss(self.D_A, fake_A, real_t=0.9)

                    # Cycle L1
                    g_cyc = self.l1(rec_A, yA) + self.l1(rec_B, yB)

                    # Identity L1
                    g_id  = self.l1(idt_A, yA) + self.l1(idt_B, yB)

                    # Δ/Δ² smoothness
                    sm1 = self.l1(self._delta(rec_A),  self._delta(yA)) + \
                          self.l1(self._delta(rec_B),  self._delta(yB))
                    sm2 = self.l1(self._delta2(rec_A), self._delta2(yA)) + \
                          self.l1(self._delta2(rec_B), self._delta2(yB))

                # Speaker consistency (FP32, NO-AMP)
                Lspk = self._spk_cos(yA, fake_B) + self._spk_cos(yB, fake_A)

                # Age + KD (FP32, NO-AMP)
                with torch.amp.autocast(self.device.type, enabled=False):
                    logit_fakeB = self.age_head(fake_B.float())
                    logit_fakeA = self.age_head(fake_A.float())

                    domB = torch.full((yA.size(0),), 1, dtype=torch.long, device=self.device)  # fakeB -> 중장년(1)
                    domA = torch.full((yB.size(0),), 0, dtype=torch.long, device=self.device)  # fakeA -> 청년(0)

                    if self.cfg.age_class_groups:
                        Lce = grouped_ce_loss(logit_fakeB, domB, self.cfg.age_class_groups) + \
                              grouped_ce_loss(logit_fakeA, domA, self.cfg.age_class_groups)
                    else:
                        Lce = torch.tensor(0.0, device=self.device)

                    # KD 준비: teacher feats
                    feat_fakeB = build_age_stats(fake_B, lf0A)   # (B,3D+3)
                    feat_fakeA = build_age_stats(fake_A, lf0B)
                    feat_realA = build_age_stats(yA,   lf0A)
                    feat_realB = build_age_stats(yB,   lf0B)

                    tdim = self.teacher.in_dim
                    f_fakeB = adapt_dim(feat_fakeB, tdim).cpu().numpy()
                    f_fakeA = adapt_dim(feat_fakeA, tdim).cpu().numpy()
                    f_realA = adapt_dim(feat_realA, tdim).cpu().numpy()
                    f_realB = adapt_dim(feat_realB, tdim).cpu().numpy()

                    with torch.no_grad():
                        tprob_fakeB = self.teacher.probs(f_fakeB).to(self.device)
                        tprob_fakeA = self.teacher.probs(f_fakeA).to(self.device)
                        tprob_realA = self.teacher.probs(f_realA).to(self.device)
                        tprob_realB = self.teacher.probs(f_realB).to(self.device)

                    Lkd = 0.25 * (
                        kd_loss(logit_fakeB, self._detach_prob(tprob_fakeB), self.cfg.kd_T) +
                        kd_loss(logit_fakeA, self._detach_prob(tprob_fakeA), self.cfg.kd_T) +
                        kd_loss(self.age_head(yA.float()), self._detach_prob(tprob_realA), self.cfg.kd_T) +
                        kd_loss(self.age_head(yB.float()), self._detach_prob(tprob_realB), self.cfg.kd_T)
                    )

                # ✔ ageCE warmup: 초반엔 G로 전파 차단
                if gstep < self.cfg.age_warmup_steps:
                    Lce_for_G = torch.tensor(0.0, device=self.device)
                else:
                    Lce_for_G = Lce

                G_total = (g_gan + self.cfg.lambda_cyc * g_cyc + self.cfg.lambda_id * g_id +
                           self.cfg.lambda_delta * sm1 + self.cfg.lambda_delta2 * sm2 +
                           self.cfg.lambda_spk * Lspk + self.cfg.lambda_age_ce * Lce_for_G + self.cfg.lambda_kd * Lkd)

                self.scaler_G.scale(G_total).backward()
                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    self.scaler_G.unscale_(self.opt_G)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.G_A2B.parameters()) + list(self.G_B2A.parameters()) + list(self.age_head.parameters()),
                        self.cfg.grad_clip
                    )
                # ↑ 오타 방지: list(self.age_head.parameters()) 로 교체
                # (아래에서 실제 step 전에 교정)
                try:
                    self.scaler_G.step(self.opt_G)
                except TypeError:
                    # parameters()() 실수 대비 안전장치
                    self.opt_G.zero_grad(set_to_none=True)
                    G_total.backward()  # AMP 비활성 환경에서도 안전
                    self.opt_G.step()
                self.scaler_G.update()

                # EMA update
                self.ema_A2B.update(self.G_A2B)
                self.ema_B2A.update(self.G_B2A)

                # ======== D step ========
                with torch.no_grad():
                    fake_A_pool = self.pool_A.query(fake_A)
                    fake_B_pool = self.pool_B.query(fake_B)

                self.opt_D.zero_grad(set_to_none=True)
                with self.autocast():
                    # ✔ 인스턴스 노이즈
                    sigma = self._inst_noise_sigma(gstep)
                    yA_noisy      = self._add_noise(yA, sigma)
                    yB_noisy      = self._add_noise(yB, sigma)
                    fake_A_noisy  = self._add_noise(fake_A_pool, sigma)
                    fake_B_noisy  = self._add_noise(fake_B_pool, sigma)

                    dA_loss = self.d_loss(self.D_A, yA_noisy, fake_A_noisy, real_t=0.9, fake_t=0.1)
                    dB_loss = self.d_loss(self.D_B, yB_noisy, fake_B_noisy, real_t=0.9, fake_t=0.1)
                    D_total = dA_loss + dB_loss

                # ✔ D-skip: D가 과도하게 이기면 가끔 스킵
                do_skip = (D_total.item() < self.cfg.d_skip_thresh)
                if self.cfg.d_skip_even_only:
                    do_skip = do_skip and (gstep % 2 == 0)

                if not do_skip:
                    self.scaler_D.scale(D_total).backward()
                    if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                        self.scaler_D.unscale_(self.opt_D)
                        torch.nn.utils.clip_grad_norm_(
                            list(self.D_A.parameters()) + list(self.D_B.parameters()),
                            self.cfg.grad_clip
                        )
                    self.scaler_D.step(self.opt_D); self.scaler_D.update()

                # ---- log ----
                if gstep % self.cfg.log_every == 0:
                    with torch.no_grad():
                        rms_fake = 0.5 * (fake_A.float().pow(2).mean().sqrt() + fake_B.float().pow(2).mean().sqrt()).item()
                    print(f"[{now()}][e{epoch:03d} s{gstep:07d}] "
                          f"G: gan={float(g_gan):.3f} cyc={float(g_cyc):.3f} id={float(g_id):.3f} "
                          f"sm1={float(sm1):.3f} sm2={float(sm2):.3f} "
                          f"| spk={float(Lspk):.3f} ageCE={float(Lce):.3f} kd={float(Lkd):.3f} "
                          f"| D={float(D_total):.3f} | sigma={sigma:.4f} | rms={rms_fake:.4f}")

                if gstep % self.cfg.save_every == 0:
                    # EMA 가중치로 안전 저장
                    self.ema_A2B.store(self.G_A2B); self.ema_B2A.store(self.G_B2A)
                    self._save_ckpt(f"step{gstep}")
                    self.ema_A2B.restore(self.G_A2B); self.ema_B2A.restore(self.G_B2A)

            # epoch end
            self.sch_G.step(); self.sch_D.step()
            print(f"[{now()}] Epoch {epoch}/{self.cfg.epochs} | LR_G={self.sch_G.get_last_lr()[0]:.6f} | LR_D={self.sch_D.get_last_lr()[0]:.6f}")
        # final
        self.ema_A2B.store(self.G_A2B); self.ema_B2A.store(self.G_B2A)
        self._save_ckpt("final")
        self.ema_A2B.restore(self.G_A2B); self.ema_B2A.restore(self.G_B2A)
        print(f"[{now()}] ✅ Training finished.")

    @staticmethod
    def _detach_prob(p: torch.Tensor) -> torch.Tensor:
        # 안전용: 확률 텐서가 NaN/Inf 없고 정규화 유지되도록 (필요 시 약간 clip)
        q = p.clamp_(1e-8, 1.0)
        q = q / q.sum(dim=1, keepdim=True)
        return q.detach()


# ---------------- main ----------------
if __name__ == "__main__":
    set_seed(1337)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 18클래스 가정 예시:
    #   10대(0..2), 20대(3..5), 30대(6..8), 40대(9..11), 50대(12..14), 60대(15..17)
    #   청년(0)=[3..8], 중장년(1)=[9..17]
    AGE_GROUPS = {
        0: list(range(3, 9)),   # youth  (B->A 목표)
        1: list(range(9, 18)),  # middle (A->B 목표)
    }
    print("[AGE_GROUPS] youth(0):", AGE_GROUPS[0], "| middle(1):", AGE_GROUPS[1])

    cfg = TrainCfg(
        device=DEVICE,
        cache_dir="./cache",
        out_dir="checkpoints_cycle_full",
        n_frames=192, batch_size=4,
        epochs=300, betas=(0.5, 0.999),
        lr_decay_from=150, amp=True, grad_clip=1.0, ema_decay=0.999,
        # ✔ TTUR
        lr_G=2e-4, lr_D=1e-4,
        # loss weights
        lambda_cyc=10.0, lambda_id=5.0, lambda_delta=0.5, lambda_delta2=0.25,
        lambda_spk=1.0, lambda_age_ce=0.5, lambda_kd=0.5, kd_T=2.0, kd_every=1,
        keras_model_path="age_model/best_model.h5",
        age_class_groups=AGE_GROUPS,
        log_every=50, save_every=5000,
        resume=None,
        # ✔ instance noise & D-skip & age warmup
        inst_warm_steps=5000, inst_sigma0=0.1,
        d_skip_thresh=0.05, d_skip_even_only=True,
        age_warmup_steps=10000,
    )

    ensure_dir(cfg.out_dir)
    print(f"[{now()}] RUN CycleGAN++(patched) | device={cfg.device}")
    Trainer(cfg).train()



