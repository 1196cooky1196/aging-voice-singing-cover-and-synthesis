
# test.py — Disentanglement VC (WORLD) with Target-Domain Mean Style (matches train.py)
# 실행:
#   1) MAIN CONFIGURATION에서 input_dir / output_dir / direction 설정
#   2) python test.py
#
# 필요:
#   - ./checkpoints_recon/content_decoder_final.pth   (train.py가 저장)
#   - ./cache/*  (전처리 캐시; 최소 mcep_norm, logf0_norm, content/logf0_global 통계)
#
# 핵심 변경:
#   - train.py와 동일하게 SpeakerEncoder를 사용하되, 추론 시엔
#     캐시의 타깃 도메인 MCEP들을 SpeakerEncoder에 통과시켜 '평균 스타일 벡터'를 만들고,
#     그 평균 벡터를 ContentDecoder에 주입합니다.
#   - SpeakerEncoder 가중치는 train.py와 동일한 초기화(시드 1337)로 생성합니다
#     (train.py에서 spk_encoder_trainable=False 였으므로 저장본이 없어도 일치합니다).

import os
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import pyworld

# ===== 프로젝트 모듈 =====
from data_preprocess import CacheIO
from model import ContentDecoder, DecoderCfg, SpeakerEncoder, SpkEncCfg

# ===== Utils =====
def now() -> str:
    return time.strftime("%H:%M:%S")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pad_wav_to_multiple(wav: np.ndarray, sr: int, frame_period_ms: float, multiple: int = 4) -> np.ndarray:
    frame_size = int(sr * frame_period_ms / 1000)
    n = len(wav)
    blocks = max(int(np.floor(n / frame_size)) + 1, 1)
    padded_frames = int((np.ceil(blocks / multiple + 1) * multiple - 1) * frame_size)
    pad = max(padded_frames - n, 0)
    if pad <= 0:
        return wav
    L = pad // 2
    R = pad - L
    return np.pad(wav, (L, R), mode="constant", constant_values=0.0)

def world_decompose(wav: np.ndarray, sr: int, frame_period_ms: float = 5.0):
    wav64 = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav64, sr, frame_period_ms)
    sp = pyworld.cheaptrick(wav64, f0, timeaxis, sr)
    ap = pyworld.d4c(wav64, f0, timeaxis, sr)
    return f0, sp, ap

def world_code_sp(sp: np.ndarray, sr: int, mcep_dim: int):
    return pyworld.code_spectral_envelope(sp, sr, mcep_dim)  # (T, mcep_dim)

def world_decode_sp(coded_sp: np.ndarray, sr: int):
    fftlen = pyworld.get_cheaptrick_fft_size(sr)
    return pyworld.decode_spectral_envelope(coded_sp, sr, fftlen)

def world_synthesize(f0: np.ndarray, dec_sp: np.ndarray, ap: np.ndarray, sr: int, frame_period_ms: float = 5.0):
    y = pyworld.synthesize(f0, dec_sp, ap, sr, frame_period_ms)
    return y.astype(np.float32)

def convert_pitch_statistically(f0: np.ndarray,
                                mean_log_src: float, std_log_src: float,
                                mean_log_tgt: float, std_log_tgt: float) -> np.ndarray:
    out = np.array(f0, dtype=np.float64, copy=True)
    idx = out > 0
    if np.any(idx):
        out[idx] = np.exp((np.log(out[idx] + 1e-8) - mean_log_src) / (std_log_src + 1e-8) * std_log_tgt + mean_log_tgt)
    return out

# ===== Content Embedder (torchaudio wav2vec2-base) =====
try:
    import torchaudio
    from torchaudio.functional import resample as ta_resample
    from torchaudio.pipelines import WAV2VEC2_BASE
    _TORCHAUDIO_OK = True
except Exception:
    _TORCHAUDIO_OK = False

class ContentEmbedder:
    def __init__(self, device: torch.device):
        if not _TORCHAUDIO_OK:
            raise RuntimeError("ContentEmbedder requires torchaudio. `pip install torchaudio`")
        bundle = WAV2VEC2_BASE
        self.model = bundle.get_model().to(device).eval()
        self.target_sr = bundle.sample_rate
        self.device = device

    @torch.inference_mode()
    def extract(self, wav: np.ndarray, sr: int) -> np.ndarray:
        t = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)  # (1,T)
        if sr != self.target_sr:
            t = ta_resample(t, sr, self.target_sr)
        embs = self.model(t)[0].squeeze(0).cpu().numpy()  # (T', C)
        return embs

    @staticmethod
    def align_to_length(feat_TxC: np.ndarray, L: int) -> np.ndarray:
        T, C = feat_TxC.shape
        if T == L:
            return feat_TxC.T.astype(np.float32)  # (C, L)
        xp = np.linspace(0.0, 1.0, T)
        xq = np.linspace(0.0, 1.0, L)
        out = np.empty((C, L), dtype=np.float32)
        for ch in range(C):
            out[ch] = np.interp(xq, xp, feat_TxC[:, ch])
        return out  # (C, L)

# ===== Config =====
@dataclass
class TestConfig:
    # 경로
    cache_dir: str = "./cache"
    decoder_ckpt: str = "./checkpoints_recon/content_decoder_final.pth"

    # 입출력 & 방향
    input_dir: str = "my_version_song"
    output_dir: str = "converted_outputs/my_song_50s"
    direction: str = "A2B"  # "A2B": 20/30대 → 40/50/60대, "B2A": 역

    # 오디오/모델 파라미터 (train.py의 ReconConfig와 일치)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    sr: int = 16000
    frame_period: float = 5.0
    mcep_dim: int = 36
    content_dim: int = 768
    style_dim: int = 128
    channels: int = 256
    n_resblocks: int = 8
    use_vuv: bool = True

    # 타깃 스타일 평균 설정
    n_style_refs: int = 16       # 평균에 사용할 타깃 도메인 샘플 수
    style_seg_len: int = 192     # 각 샘플에서 사용할 프레임 길이 (train과 맞춤)

# ===== Converter =====
class Converter:
    def __init__(self, cfg: TestConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        print(f"[{now()}] 🔧 초기화 시작...")

        # train.py와 동일한 시드로 초기화 (SpeakerEncoder 무저장 시 동일 파라미터 보장)
        set_seed(1337)

        # 1) Decoder
        self.decoder = self._load_decoder()

        # 2) SpeakerEncoder (학습 때와 동일 구조, 기본은 초기화 고정)
        self.spk_enc = SpeakerEncoder(SpkEncCfg(in_dim=36, out_dim=cfg.style_dim)).to(self.device).eval()
        spk_ckpt_guess = os.path.join(os.path.dirname(cfg.decoder_ckpt), "spk_encoder_final.pth")
        if os.path.isfile(spk_ckpt_guess):
            try:
                self.spk_enc.load_state_dict(torch.load(spk_ckpt_guess, map_location=self.device), strict=False)
                print(f"[{now()}] 🎙️ SpeakerEncoder 로드: {spk_ckpt_guess}")
            except Exception as e:
                print(f"[{now()}] ⚠️ SpeakerEncoder 로드 실패(무시): {e}  (초기화된 고정 가중치 사용)")

        # 3) Content embedder
        self.content_embedder = ContentEmbedder(self.device)

        # 4) 통계 로드
        self._load_statistics()

        # 5) 타깃 도메인 평균 스타일 벡터 준비
        self.target_style = self._build_target_mean_style().to(self.device)  # (1, style_dim)

        print(f"[{now()}] ✅ 초기화 완료.")

    def _load_decoder(self) -> ContentDecoder:
        dcfg = DecoderCfg(
            content_dim=self.cfg.content_dim, pitch_dim=1,
            vuv_dim=(1 if self.cfg.use_vuv else 0),
            style_dim=self.cfg.style_dim,
            out_dim=self.cfg.mcep_dim, channels=self.cfg.channels,
            n_resblocks=self.cfg.n_resblocks, use_film=True
        )
        decoder = ContentDecoder(dcfg).to(self.device).eval()
        path = os.path.normpath(self.cfg.decoder_ckpt)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Decoder ckpt not found: {path}")
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        decoder.load_state_dict(state, strict=False)
        print(f"[{now()}] ✔️ Decoder 로드: {path}")
        return decoder

    def _load_statistics(self):
        cache = CacheIO.load_preprocessed_data(self.cfg.cache_dir)
        # 필수 통계
        self.mA_mean, self.mA_std = cache["mcep_norm"]["mean_A"], cache["mcep_norm"]["std_A"]
        self.mB_mean, self.mB_std = cache["mcep_norm"]["mean_B"], cache["mcep_norm"]["std_B"]
        self.logf0_A_mean, self.logf0_A_std = float(cache["logf0_norm"]["mean_A"]), float(cache["logf0_norm"]["std_A"])
        self.logf0_B_mean, self.logf0_B_std = float(cache["logf0_norm"]["mean_B"]), float(cache["logf0_norm"]["std_B"])
        # 선택 통계 (없으면 폴백)
        cg = cache.get("content_normalization")
        lg = cache.get("logf0_global_normalization")
        if cg is not None:
            self.c_mean = cg["mean"].astype(np.float32)
            self.c_std  = cg["std"].astype(np.float32) + 1e-8
        else:
            self.c_mean = 0.0; self.c_std = 1.0
            print(f"[{now()}] ℹ️ content_normalization 없음 → 무정규화")
        if lg is not None:
            self.lg_mean = float(lg["mean"]); self.lg_std = float(lg["std"]) + 1e-8
        else:
            self.lg_mean = 0.0; self.lg_std = 1.0
            print(f"[{now()}] ℹ️ logf0_global_normalization 없음 → 무정규화")
        # 타깃/소스 리스트
        self.coded_A = CacheIO.load_pickle(os.path.join(self.cfg.cache_dir, "coded_sps_A_norm.pickle"))
        self.coded_B = CacheIO.load_pickle(os.path.join(self.cfg.cache_dir, "coded_sps_B_norm.pickle"))
        if not isinstance(self.coded_A, list) or not isinstance(self.coded_B, list):
            raise RuntimeError("coded_sps_{A,B}_norm.pickle 로드 실패 혹은 포맷 오류")

        print(f"[{now()}] ✔️ 통계/리스트 로드 완료  (A:{len(self.coded_A)} / B:{len(self.coded_B)})")

    @torch.no_grad()
    def _build_target_mean_style(self) -> torch.Tensor:
        """
        캐시에서 타깃 도메인의 normalized MCEP들을 골라 SpeakerEncoder에 통과,
        평균 스타일 벡터 (1, E)를 만든다.
        """
        if self.cfg.direction.upper() == "A2B":
            pool = self.coded_B  # 타깃: B (중장년)
        elif self.cfg.direction.upper() == "B2A":
            pool = self.coded_A  # 타깃: A (청년)
        else:
            raise ValueError("direction은 'A2B' 또는 'B2A' 여야 합니다.")

        if len(pool) == 0:
            raise RuntimeError("타깃 도메인의 coded_sps_*_norm 리스트가 비어 있습니다.")

        n = min(self.cfg.n_style_refs, len(pool))
        segL = int(self.cfg.style_seg_len)
        idxs = np.linspace(0, len(pool)-1, n, dtype=int)  # 균일 샘플링

        vecs = []
        for i in idxs:
            mcep_36xT = np.asarray(pool[i], dtype=np.float32)  # (36, T)
            T = mcep_36xT.shape[1]
            if T < segL:
                # pad to segL
                need = segL - T + 1
                L = max(need // 2, 0); R = max(need - L, 0)
                mcep_36xT = np.pad(mcep_36xT, ((0,0),(L,R)), mode="edge")
                T = mcep_36xT.shape[1]
            # 랜덤/센터 크롭 중 센터 크롭으로 안정화
            s = (T - segL) // 2
            e = s + segL
            seg = mcep_36xT[:, s:e]  # (36, segL)
            t = torch.from_numpy(seg).unsqueeze(0).to(self.device)  # (1,36,L)
            v = self.spk_enc(t)  # (1,E)
            vecs.append(v)

        V = torch.cat(vecs, dim=0)        # (n,E)
        mean_v = V.mean(dim=0, keepdim=True)  # (1,E)
        print(f"[{now()}] 🎯 타깃 평균 스타일 준비 완료: refs={n}, seg={segL}, norm={float(mean_v.norm().item()):.4f}")
        return mean_v  # (1,E)

    @torch.no_grad()
    def convert_file(self, wav_path: str) -> np.ndarray:
        cfg = self.cfg

        # 1) 오디오 로드 & 패딩
        wav, _ = librosa.load(wav_path, sr=cfg.sr, mono=True)
        wav_pad = pad_wav_to_multiple(wav, sr=cfg.sr, frame_period_ms=cfg.frame_period)

        # 2) WORLD 분해
        f0, sp, ap = world_decompose(wav_pad, sr=cfg.sr, frame_period_ms=cfg.frame_period)
        T = f0.shape[0]

        # 3) 콘텐츠 임베딩
        c_emb = self.content_embedder.extract(wav, sr=cfg.sr)              # (T', Cc)
        c_aligned = self.content_embedder.align_to_length(c_emb, T)        # (Cc, T)
        # content 정규화
        if isinstance(self.c_mean, np.ndarray):
            c_norm = (c_aligned - self.c_mean) / (self.c_std)
        else:
            c_norm = (c_aligned - self.c_mean) / (self.c_std + 1e-8)

        # 4) logF0 정규화 & V/UV
        vuv = (f0 > 0).astype(np.float32)
        logf0_norm = np.zeros_like(f0, dtype=np.float32)
        if np.any(vuv > 0):
            logf0_norm[vuv > 0] = (np.log(f0[vuv > 0] + 1e-8) - self.lg_mean) / (self.lg_std)

        # 5) 도메인별 통계 & F0 변환
        if cfg.direction.upper() == "A2B":
            mean_log_src, std_log_src = self.logf0_A_mean, self.logf0_A_std
            mean_log_tgt, std_log_tgt = self.logf0_B_mean, self.logf0_B_std
            mcep_mean_tgt, mcep_std_tgt = self.mB_mean, self.mB_std
        elif cfg.direction.upper() == "B2A":
            mean_log_src, std_log_src = self.logf0_B_mean, self.logf0_B_std
            mean_log_tgt, std_log_tgt = self.logf0_A_mean, self.logf0_A_std
            mcep_mean_tgt, mcep_std_tgt = self.mA_mean, self.mA_std
        else:
            raise ValueError("direction은 'A2B' 또는 'B2A' 여야 합니다.")
        f0_conv = convert_pitch_statistically(f0, mean_log_src, std_log_src, mean_log_tgt, std_log_tgt)

        # 6) 디코더 추론 (타깃 평균 스타일 벡터 주입)
        c_tensor = torch.from_numpy(c_norm).unsqueeze(0).to(self.device)             # (1,Cc,T)
        l_tensor = torch.from_numpy(logf0_norm).reshape(1,1,T).to(self.device)       # (1,1,T)
        v_tensor = torch.from_numpy(vuv).reshape(1,1,T).to(self.device)              # (1,1,T)
        style_vec = self.target_style                                              # (1,E)

        mcep_norm_pred = self.decoder(c_tensor, l_tensor, vuv=v_tensor, style=style_vec)  # (1,36,T)

        # 7) 역정규화 & WORLD 합성
        mcep_pred = mcep_norm_pred.squeeze(0).cpu().numpy() * (mcep_std_tgt + 1e-8) + mcep_mean_tgt  # (36,T)
        coded_sp_conv = np.ascontiguousarray(mcep_pred.T).astype(np.float64)  # (T,36)
        dec_sp = world_decode_sp(coded_sp_conv, cfg.sr)
        y = world_synthesize(f0_conv.astype(np.float64),
                             dec_sp.astype(np.float64),
                             ap.astype(np.float64),
                             cfg.sr, cfg.frame_period)

        # 8) 안전 레벨링/무음 가드
        peak = float(np.max(np.abs(y)) + 1e-9)
        if peak > 1.0:
            y = y / peak
        rms = float(np.sqrt(np.mean(y*y) + 1e-12))
        if rms < 1e-4:
            print(f"[{now()}] ⚠️ 출력이 매우 작습니다 (rms={rms:.2e}). 통계/하이퍼/입력레벨을 확인하세요.")
        return y

    def run(self):
        ensure_dir(self.cfg.output_dir)
        wav_files = sorted([f for f in os.listdir(self.cfg.input_dir) if f.lower().endswith('.wav')])
        if not wav_files:
            print(f"⚠️ '{self.cfg.input_dir}' 폴더에 변환할 wav 파일이 없습니다.")
            return

        print(f"[{now()}] ▶️ 변환 시작: {len(wav_files)}개 | {self.cfg.direction} | "
              f"'{self.cfg.input_dir}' -> '{self.cfg.output_dir}'")
        for i, fname in enumerate(wav_files, 1):
            t0 = time.time()
            try:
                in_path = os.path.join(self.cfg.input_dir, fname)
                y = self.convert_file(in_path)
                out_name, _ = os.path.splitext(fname)
                out_path = os.path.join(self.cfg.output_dir, f"{out_name}_converted_{self.cfg.direction}.wav")
                sf.write(out_path, y, self.cfg.sr)
                print(f"[{now()}] ({i}/{len(wav_files)}) ✅ {fname} -> {os.path.basename(out_path)} "
                      f"({time.time()-t0:.2f}s)")
\
\






# ===== MAIN =====
if __name__ == "__main__":
    # ⚙️ MAIN CONFIGURATION
    config = TestConfig(
        input_dir="my_version_song",
        output_dir="converted_outputs/my_song_converted_to_50s",
        direction="A2B",  # "A2B" or "B2A"
        decoder_ckpt="./checkpoints_recon/content_decoder_final.pth",
        cache_dir="./cache",
        # 하이퍼는 train.py와 반드시 일치
        sr=16000, frame_period=5.0, mcep_dim=36,
        content_dim=768, style_dim=128, channels=256, n_resblocks=8, use_vuv=True,
        # 평균 스타일 설정 (필요시 조절)
        n_style_refs=16, style_seg_len=192,
    )
    converter = Converter(config)
    converter.run()



'''
# test.py — WORLD 합성 제거 + HiFi-GAN으로 최종 파형 합성 (기존 파이프라인 유지)
# -*- coding: utf-8 -*-
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf
import librosa
import torch
import pyworld
import torch.nn as nn

# (NEW) HiFi-GAN용 동적 임포트와 멜 변환 유틸
import json
import librosa.filters as lfilters
import importlib.util
from pathlib import Path

# =========================
# dict -> attribute 접근 유틸 (HiFi-GAN models가 점 표기 접근을 사용)
# =========================
class AttrDict(dict):
    """d['x']와 d.x 둘 다 되는 래퍼. 중첩 dict도 재귀 변환."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            if isinstance(v, dict):
                self[k] = AttrDict(v)
            elif isinstance(v, list):
                self[k] = [AttrDict(x) if isinstance(x, dict) else x for x in v]
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# =========================
# HiFi-GAN models.py 동적 로드
# =========================
def load_hifigan_models(repo_root: str):
    """
    repo_root 안의 utils.py를 먼저 로드하여 sys.modules['utils']에 주입한 뒤,
    models.py를 파일 경로로 직접 로드한다.
    반환: (Generator, remove_weight_norm or None)
    """
    import sys, importlib.util
    from pathlib import Path

    repo = Path(repo_root).resolve()
    if not repo.exists():
        raise FileNotFoundError(f"hifigan repo not found: {repo_root}")

    # 1) utils.py 찾기 (루트 우선, 없으면 재귀 탐색)
    utils_py = repo / "utils.py"
    if not utils_py.is_file():
        for p in repo.rglob("utils.py"):
            utils_py = p
            break
    if not utils_py.is_file():
        raise ModuleNotFoundError(f"HiFi-GAN utils.py를 찾을 수 없습니다: {repo_root}")

    # 2) utils.py 로드 → sys.modules['utils']에 주입 (models.py가 from utils import ... 할 때 쓰이도록)
    spec_u = importlib.util.spec_from_file_location("hifigan_utils", str(utils_py))
    mod_u = importlib.util.module_from_spec(spec_u)
    assert spec_u and spec_u.loader, "importlib spec/loader 생성 실패(utils)"
    spec_u.loader.exec_module(mod_u)
    # 외부 패키지 utils보다 우리가 로드한 걸 우선 사용하게 덮어쓰기
    sys.modules["utils"] = mod_u

    # 3) models.py 찾기 (루트 우선, 없으면 재귀 탐색)
    models_py = repo / "models.py"
    if not models_py.is_file():
        for p in repo.rglob("models.py"):
            models_py = p
            break
    if not models_py.is_file():
        raise ModuleNotFoundError(f"HiFi-GAN models.py를 찾을 수 없습니다: {repo_root}")

    # 4) models.py 로드
    spec_m = importlib.util.spec_from_file_location("hifigan_models", str(models_py))
    mod_m = importlib.util.module_from_spec(spec_m)
    assert spec_m and spec_m.loader, "importlib spec/loader 생성 실패(models)"
    spec_m.loader.exec_module(mod_m)

    Generator = getattr(mod_m, "Generator")
    remove_weight_norm = getattr(mod_m, "remove_weight_norm", None)
    return Generator, remove_weight_norm


# =========================
# Content 임베딩 (기존 유지: torchaudio Wav2Vec2)
# =========================
try:
    import torchaudio
    from torchaudio.functional import resample as ta_resample
    from torchaudio.pipelines import WAV2VEC2_BASE
    _TORCHAUDIO_OK = True
except ImportError:
    _TORCHAUDIO_OK = False

from data_preprocess import CacheIO  # 캐시 로더
from model import ContentDecoder, DecoderCfg  # 모델 구조

# =========================
# Utils & WORLD Codec (분해/복원만 사용)
# =========================
def now() -> str:
    return time.strftime("%H:%M:%S")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def pad_wav_to_multiple(wav: np.ndarray, sr: int, frame_period_ms: float, multiple: int = 4) -> np.ndarray:
    frame_size = int(sr * frame_period_ms / 1000)
    n = len(wav)
    padded_frames = int((np.ceil((np.floor(n / frame_size) + 1) / multiple + 1) * multiple - 1) * frame_size)
    pad = max(padded_frames - n, 0)
    if pad <= 0: return wav
    L = pad // 2
    R = pad - L
    return np.pad(wav, (L, R), mode="constant", constant_values=0.0)

def world_decompose(wav: np.ndarray, sr: int, frame_period_ms: float = 5.0):
    wav64 = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav64, sr, frame_period_ms)
    sp = pyworld.cheaptrick(wav64, f0, timeaxis, sr)
    ap = pyworld.d4c(wav64, f0, timeaxis, sr)
    return f0, sp, ap

def world_decode_sp(coded_sp: np.ndarray, sr: int, mcep_dim: int):
    # coded_sp expected (T, dim) float64, but (dim, T)도 실제로 통과되던 점을 유지
    fftlen = pyworld.get_cheaptrick_fft_size(sr)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, sr, fftlen)
    return decoded_sp

def convert_pitch_statistically(f0: np.ndarray,
                                mean_log_src: float, std_log_src: float,
                                mean_log_tgt: float, std_log_tgt: float) -> np.ndarray:
    out = np.copy(f0)
    idx = f0 > 0
    if np.any(idx):
        out[idx] = np.exp((np.log(f0[idx] + 1e-8) - mean_log_src) / (std_log_src + 1e-8) * std_log_tgt + mean_log_tgt)
    return out

# =========================
# Content Embedder (기존 유지)
# =========================
class ContentEmbedder:
    def __init__(self, device: torch.device):
        if not _TORCHAUDIO_OK:
            raise RuntimeError("ContentEmbedder requires torchaudio. Please install it.")
        bundle = WAV2VEC2_BASE
        self.model = bundle.get_model().to(device).eval()
        self.target_sr = bundle.sample_rate
        self.device = device

    @torch.inference_mode()
    def extract(self, wav: np.ndarray, sr: int) -> np.ndarray:
        t = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)
        if sr != self.target_sr:
            t = ta_resample(t, sr, self.target_sr)
        embs = self.model(t)[0].squeeze(0).cpu().numpy()
        return embs  # (T', C)

    @staticmethod
    def align_to_length(feat_TxC: np.ndarray, L: int) -> np.ndarray:
        T, C = feat_TxC.shape
        if T == L:
            return feat_TxC.T.astype(np.float32)  # (C, T)
        xp = np.linspace(0.0, 1.0, T)
        xq = np.linspace(0.0, 1.0, L)
        out = np.empty((C, L), dtype=np.float32)
        for ch in range(C):
            out[ch] = np.interp(xq, xp, feat_TxC[:, ch])
        return out  # (C, L)

# =========================
# HiFi-GAN Vocoder 래퍼
# =========================
class HiFiGANVocoder:
    """
    jik876/HiFi-GAN 구조 호환. config.json의 오디오 파라미터에 맞춰 멜 변환 후 Generator로 합성.
    """
    def __init__(self, repo_root: str, ckpt_path: str, config_path: str, device: torch.device):
        # 1) 모델 소스 로드
        Generator, remove_weight_norm = load_hifigan_models(repo_root)

        # 2) config 로드
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        # 3) 오디오 파라미터 (멜 투영/프레임 정렬용)
        audio = cfg_dict.get("audio", cfg_dict)
        self.sr   = int(audio.get("sampling_rate", 22050))
        self.nfft = int(audio.get("filter_length", 1024))
        self.hop  = int(audio.get("hop_length", 256))
        self.win  = int(audio.get("win_length", 1024))
        self.nmel = int(audio.get("num_mels", 80))
        self.fmin = float(audio.get("mel_fmin", 0.0))
        self.fmax = float(audio.get("mel_fmax", self.sr // 2))

        # 4) Generator 생성 (AttrDict로 하이퍼 전달)
        h = AttrDict(cfg_dict)
        self.device = device if torch.cuda.is_available() else torch.device("cpu")
        self.gen = Generator(h).to(self.device)

        # 5) 체크포인트 로드 (다양한 키 형태 대응)
        states = torch.load(ckpt_path, map_location=self.device)
        if isinstance(states, dict):
            sd = states.get("generator", states.get("state_dict", states))
        else:
            sd = states
        self.gen.load_state_dict(sd, strict=False)
        self.gen.eval()
        try:
            if remove_weight_norm:
                remove_weight_norm(self.gen)
        except Exception:
            pass

        # 6) 선형→멜 변환용 고정 필터 (WORLD 복원 스펙을 멜로 투영)
        self.mel_basis = lfilters.mel(
            sr=self.sr, n_fft=self.nfft, n_mels=self.nmel,
            fmin=self.fmin, fmax=self.fmax, htk=True, norm=None
        ).astype(np.float32)

    @staticmethod
    def _log10(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        return np.log10(np.maximum(x, eps))

    @torch.no_grad()
    def infer_from_linear_spectrogram(self, sp_lin_TxF: np.ndarray, dur_sec: float) -> np.ndarray:
        """
        sp_lin_TxF: [T_mcep, nfft//2+1] — WORLD decode로 얻은 선형 스펙트럼 (시간축=첫번째 차원)
        dur_sec: 입력 wav 길이(초). mel 프레임 수 계산에 사용.
        """
        # [freq, T_mcep]로 전치 → mel_basis @ ... → [n_mels, T_mcep]
        mel = (self.mel_basis @ sp_lin_TxF.T).astype(np.float32)
        mel = np.maximum(mel, 1e-9)
        mel = self._log10(mel)

        # HiFi-GAN 멜 프레임 수로 시간축 보간
        T_mel = int(np.round(dur_sec * self.sr / self.hop))
        if mel.shape[1] != T_mel:
            src = np.linspace(0.0, 1.0, mel.shape[1], endpoint=True)
            tgt = np.linspace(0.0, 1.0, T_mel, endpoint=True)
            mel = np.stack([np.interp(tgt, src, mel[ch]) for ch in range(mel.shape[0])], axis=0).astype(np.float32)

        m = torch.from_numpy(mel[None, ...]).to(self.device)  # [1, n_mels, T]
        y = self.gen(m).squeeze().detach().cpu().numpy().astype(np.float32)
        return y

# =========================
# Test Configuration (기존 + HiFi-GAN 경로 추가)
# =========================
@dataclass
class TestConfig:
    # --- 경로 설정 ---
    cache_dir: str = "cache"
    decoder_ckpt: str = "checkpoints_recon/content_decoder_final.pth"
    style_embedder_ckpt: str = "checkpoints_recon/style_embedder_final.pth"

    # (NEW) HiFi-GAN 경로
    hifigan_repo: str = "third_party/hifigan"
    hifigan_ckpt: str = "third_party\hifigan\generator_universal_v1"
    hifigan_config: str = "third_party/hifigan/config.json"
    
    # --- 변환 설정 ---
    input_dir: str = "my_version_song"              # 변환할 wav 파일이 있는 폴더
    output_dir: str = "converted_outputs/my_song_50s"
    direction: str = "A2B"                          # "A2B": 20대->50대, "B2A": 50대->20대

    # --- 모델 및 오디오 파라미터 ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    sr: int = 16000
    frame_period: float = 5.0
    mcep_dim: int = 36
    # 아래 값들은 train.py의 ReconConfig와 일치해야 함
    content_dim: int = 768  # Wav2Vec2-base
    style_dim: int = 128
    channels: int = 256
    n_resblocks: int = 8
    use_vuv: bool = True

# =========================
# Conversion Engine
# =========================
class Converter:
    def __init__(self, cfg: TestConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        print(f"[{now()}] 🔧 초기화 시작...")
        # 1) 모델 로드
        self.decoder = self._load_decoder()
        self.style_embedder = self._load_style_embedder()

        # 2) 콘텐츠 임베더
        self.content_embedder = ContentEmbedder(self.device)

        # 3) 정규화 통계
        self._load_statistics()

        # 4) HiFi-GAN vocoder
        self.vocoder = HiFiGANVocoder(
            repo_root=cfg.hifigan_repo,
            ckpt_path=cfg.hifigan_ckpt,
            config_path=cfg.hifigan_config,
            device=self.device
        )
        print(f"[{now()}] ✅ 초기화 완료.")

    def _load_decoder(self) -> ContentDecoder:
        dcfg = DecoderCfg(
            content_dim=self.cfg.content_dim, pitch_dim=1,
            vuv_dim=1 if self.cfg.use_vuv else 0, style_dim=self.cfg.style_dim,
            out_dim=self.cfg.mcep_dim, channels=self.cfg.channels,
            n_resblocks=self.cfg.n_resblocks, use_film=True
        )
        decoder = ContentDecoder(dcfg).to(self.device).eval()
        decoder.load_state_dict(torch.load(self.cfg.decoder_ckpt, map_location=self.device))
        return decoder

    def _load_style_embedder(self) -> nn.Embedding:
        embedder = nn.Embedding(2, self.cfg.style_dim).to(self.device).eval()
        embedder.load_state_dict(torch.load(self.cfg.style_embedder_ckpt, map_location=self.device))
        return embedder

    def _load_statistics(self):
        cache = CacheIO.load_preprocessed_data(self.cfg.cache_dir)
        # Content & Global logF0 norm
        self.c_mean = cache["content_normalization"]["mean"].astype(np.float32)
        self.c_std = cache["content_normalization"]["std"].astype(np.float32)
        self.lg_mean = float(cache["logf0_global_normalization"]["mean"])
        self.lg_std = float(cache["logf0_global_normalization"]["std"])
        # MCEP norm per domain
        self.mA_mean, self.mA_std = cache["mcep_norm"]["mean_A"], cache["mcep_norm"]["std_A"]
        self.mB_mean, self.mB_std = cache["mcep_norm"]["mean_B"], cache["mcep_norm"]["std_B"]
        # logF0 stats per domain
        self.logf0_A_mean, self.logf0_A_std = cache["logf0_norm"]["mean_A"], cache["logf0_norm"]["std_A"]
        self.logf0_B_mean, self.logf0_B_std = cache["logf0_norm"]["mean_B"], cache["logf0_norm"]["std_B"]

    @torch.no_grad()
    def convert_file(self, wav_path: str) -> np.ndarray:
        cfg = self.cfg
        
        # 1) 오디오 로드 + WORLD 분해
        wav, _ = librosa.load(wav_path, sr=cfg.sr, mono=True)
        dur_sec = len(wav) / cfg.sr
        wav_padded = pad_wav_to_multiple(wav, sr=cfg.sr, frame_period_ms=cfg.frame_period)
        f0, sp, ap = world_decompose(wav_padded, sr=cfg.sr, frame_period_ms=cfg.frame_period)
        T = f0.shape[0]  # WORLD 프레임 수

        # 2) 특징 추출/정규화
        # Content
        c_emb = self.content_embedder.extract(wav, sr=cfg.sr)      # (T', Cc)
        c_aligned = self.content_embedder.align_to_length(c_emb, T)  # (Cc, T)
        c_norm = (c_aligned - self.c_mean) / self.c_std
        
        # logF0 & V/UV
        vuv = (f0 > 0).astype(np.float32)
        logf0_norm = np.zeros_like(f0, dtype=np.float32)
        if np.any(vuv > 0):
            logf0_norm[vuv > 0] = (np.log(f0[vuv > 0] + 1e-8) - self.lg_mean) / self.lg_std

        # 3) 목표 스타일/운율(F0)
        if cfg.direction.upper() == "A2B":
            target_domain_id = 1  # B (50대)
            mean_log_src, std_log_src = self.logf0_A_mean, self.logf0_A_std
            mean_log_tgt, std_log_tgt = self.logf0_B_mean, self.logf0_B_std
            mcep_mean_tgt, mcep_std_tgt = self.mB_mean, self.mB_std
        elif cfg.direction.upper() == "B2A":
            target_domain_id = 0  # A (20대)
            mean_log_src, std_log_src = self.logf0_B_mean, self.logf0_B_std
            mean_log_tgt, std_log_tgt = self.logf0_A_mean, self.logf0_A_std
            mcep_mean_tgt, mcep_std_tgt = self.mA_mean, self.mA_std
        else:
            raise ValueError("direction은 'A2B' 또는 'B2A' 여야 합니다.")
        
        # F0 변환
        f0_converted = convert_pitch_statistically(f0, mean_log_src, std_log_src, mean_log_tgt, std_log_tgt)

        # 4) 디코더 추론
        c_tensor = torch.from_numpy(c_norm).unsqueeze(0).to(self.device)                 # [1, Cc, T]
        l_tensor = torch.from_numpy(logf0_norm).unsqueeze(0).unsqueeze(0).to(self.device)  # [1,1,T]
        v_tensor = torch.from_numpy(vuv).unsqueeze(0).unsqueeze(0).to(self.device)       # [1,1,T]
        domain_tensor = torch.LongTensor([target_domain_id]).to(self.device)
        
        style_vector = self.style_embedder(domain_tensor)
        mcep_norm_pred = self.decoder(c_tensor, l_tensor, vuv=v_tensor, style=style_vector)
        
        # 5) 역정규화
        mcep_pred = mcep_norm_pred.squeeze(0).cpu().numpy() * mcep_std_tgt + mcep_mean_tgt
        coded_sp_converted = np.ascontiguousarray(mcep_pred.T).astype(np.float64)  # (dim, T)

        # 6) WORLD '복원'만 수행: MCEP → 선형 스펙트럼 (합성은 HiFi-GAN)
        decoded_sp = world_decode_sp(coded_sp_converted, cfg.sr, cfg.mcep_dim)  # (T, nfft//2+1)

        # 7) HiFi-GAN: 선형 스펙트럼을 멜로 변환 후 신경 보코딩
        y_hifi = self.vocoder.infer_from_linear_spectrogram(decoded_sp, dur_sec=dur_sec)

        # 8) 최종 SR 맞추기 (HiFi-GAN 훈련 SR ≠ 프로젝트 SR일 수 있음)
        if self.vocoder.sr != cfg.sr:
            y_out = librosa.resample(y_hifi, orig_sr=self.vocoder.sr, target_sr=cfg.sr)
        else:
            y_out = y_hifi

        return y_out

    def run(self):
        ensure_dir(self.cfg.output_dir)
        
        wav_files = sorted([f for f in os.listdir(self.cfg.input_dir) if f.lower().endswith('.wav')])
        if not wav_files:
            print(f"⚠️ 경고: '{self.cfg.input_dir}' 폴더에 변환할 wav 파일이 없습니다.")
            return

        print(f"[{now()}] ▶️ 변환 시작: {len(wav_files)}개 파일 | {self.cfg.direction} | "
              f"'{self.cfg.input_dir}' -> '{self.cfg.output_dir}'")
        
        for i, fname in enumerate(wav_files, 1):
            t0 = time.time()
            try:
                in_path = os.path.join(self.cfg.input_dir, fname)
                wav_out = self.convert_file(in_path)
                
                out_fname, _ = os.path.splitext(fname)
                out_path = os.path.join(self.cfg.output_dir, f"{out_fname}_converted_{self.cfg.direction}.wav")
                sf.write(out_path, wav_out, self.cfg.sr)
                
                elapsed = time.time() - t0
                print(f"[{now()}] ({i}/{len(wav_files)}) ✅ 성공: {fname} -> {os.path.basename(out_path)} ({elapsed:.2f}초)")
            except Exception as e:
                print(f"[{now()}] ({i}/{len(wav_files)}) ❌ 실패: {fname}. 오류: {e}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # 필요 시 여기서 경로/파라미터만 바꿔서 실행
    config = TestConfig(
        input_dir = "my_version_song",
        output_dir = "converted_outputs/my_song_converted_to_50s",
        direction = "A2B",
        # 가중치/캐시 경로
        decoder_ckpt = "checkpoints_recon/content_decoder_final.pth",
        style_embedder_ckpt = "checkpoints_recon/style_embedder_final.pth",
        cache_dir = "cache",
        # HiFi-GAN 경로 (다운로드/배치한 파일명과 맞추기)
        hifigan_repo = "third_party/hifigan",
        hifigan_ckpt = "third_party\hifigan\generator_universal_v1",
        hifigan_config = "third_party/hifigan/config.json",
        # 오디오/모델 파라미터
        sr = 16000,
        frame_period = 5.0,
        mcep_dim = 36,
        content_dim = 768,
        style_dim = 128,
        channels = 256,
        n_resblocks = 8,
        use_vuv = True,
    )

    converter = Converter(config)
    converter.run()
'''



