
# data_preprocess.py — Disentangle-ready preprocessing
# - WORLD(MCEP/f0/ap) + (옵션) Wav2Vec2/HuBERT content embedding
# - MCEP 정규화(mean/std, A/B), 도메인별 log-f0 통계, 글로벌 log-f0 정규화 통계
# - 콘텐츠 임베딩 정규화(mean/std)
# - 캐시 산출물(파일명/포맷) — train.py / test.py와 100% 호환
#   ./cache/
#     mcep_normalization.npz           # mean_A,std_A,mean_B,std_B  (각 shape (dim,1))
#     logf0s_normalization.npz         # mean_A,std_A,mean_B,std_B  (float)
#     coded_sps_A_norm.pickle          # List[np.ndarray], 각 (dim,T_i)
#     coded_sps_B_norm.pickle          # List[np.ndarray], 각 (dim,T_i)
#   (옵션, 콘텐츠 추출시 추가)
#     content_emb_A.pickle, content_emb_B.pickle     # List[(T,Cc)]
#     content_normalization.npz                      # mean, std  (각 (Cc,1))
#     logf0_seq_A.pickle, logf0_seq_B.pickle        # List[(T,1)]  (무성=0)
#     vuv_A.pickle, vuv_B.pickle                    # List[(T,1)]  (0/1)
#     logf0_global_normalization.npz                # mean, std (float)
import os
import sys
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import librosa
import pyworld

# ===== (옵션) 콘텐츠 임베딩 =====
try:
    import torch
    import torchaudio
    from torchaudio.functional import resample as ta_resample
    _TORCHAUDIO_OK = True
except Exception:
    _TORCHAUDIO_OK = False


# =========================
# Config
# =========================
@dataclass(frozen=True)
class PreprocessConfig:
    sr: int = 16000
    dim: int = 36
    frame_period: float = 5.0  # ms (WORLD hop)
    exts: Tuple[str, ...] = (".wav",)
    # Content embedding options
    extract_content: bool = True
    content_backend: str = "wav2vec2"   # "wav2vec2" | "hubert"
    device: str = "cpu"                 # for content embedder
    # (선택) 전처리 편의옵션
    trim_silence: bool = False
    trim_top_db: int = 60


# =========================
# I/O
# =========================
class AudioLoader:
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg

    def load_wav_files(self, wav_dir: str) -> List[np.ndarray]:
        if not os.path.isdir(wav_dir):
            raise FileNotFoundError(f"Directory not found: {wav_dir}")
        wavs: List[np.ndarray] = []
        for fname in sorted(os.listdir(wav_dir)):
            if not fname.lower().endswith(self.cfg.exts):
                continue
            path = os.path.join(wav_dir, fname)
            wav, _ = librosa.load(path, sr=self.cfg.sr, mono=True)
            if self.cfg.trim_silence:
                wav, _ = librosa.effects.trim(wav, top_db=self.cfg.trim_top_db)
            wavs.append(wav.astype(np.float32))
        if not wavs:
            raise RuntimeError(f"No audio files found in: {wav_dir}")
        return wavs


class CacheIO:
    @staticmethod
    def save_pickle(obj: Any, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_pickle(path: str) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_preprocessed_data(cache_dir: str) -> Dict[str, Any]:
        out = {
            "coded_sps_A_norm": CacheIO.load_pickle(os.path.join(cache_dir, "coded_sps_A_norm.pickle")),
            "coded_sps_B_norm": CacheIO.load_pickle(os.path.join(cache_dir, "coded_sps_B_norm.pickle")),
            "mcep_norm": np.load(os.path.join(cache_dir, "mcep_normalization.npz")),
            "logf0_norm": np.load(os.path.join(cache_dir, "logf0s_normalization.npz")),
        }
        # 선택적 리소스들 있으면 함께 로드
        opt_keys = [
            ("content_emb_A", "content_emb_A.pickle"),
            ("content_emb_B", "content_emb_B.pickle"),
            ("logf0_seq_A", "logf0_seq_A.pickle"),
            ("logf0_seq_B", "logf0_seq_B.pickle"),
            ("vuv_A", "vuv_A.pickle"),
            ("vuv_B", "vuv_B.pickle"),
        ]
        for k, fn in opt_keys:
            p = os.path.join(cache_dir, fn)
            if os.path.exists(p):
                out[k] = CacheIO.load_pickle(p)

        for name in ["content_normalization.npz", "logf0_global_normalization.npz"]:
            p = os.path.join(cache_dir, name)
            if os.path.exists(p):
                out[name.replace(".npz", "")] = np.load(p)
        return out

# 기존 함수 이름을 기대하는 외부 코드 호환(필요시)
def load_preprocessed_data(cache_dir: str) -> Dict[str, Any]:
    return CacheIO.load_preprocessed_data(cache_dir)


# =========================
# WORLD Codec
# =========================
class WorldCodec:
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg

    @staticmethod
    def pad_wav_to_multiple(wav: np.ndarray, sr: int, frame_period: float = 5.0, multiple: int = 4) -> np.ndarray:
        frame_size = int(sr * frame_period / 1000)
        n = len(wav)
        padded_frames = int((np.ceil((np.floor(n / frame_size) + 1) / multiple + 1) * multiple - 1) * frame_size)
        pad = max(padded_frames - n, 0)
        if pad <= 0:
            return wav
        L = pad // 2
        R = pad - L
        return np.pad(wav, (L, R), mode="constant", constant_values=0)

    @staticmethod
    def convert_pitch_statistically(f0, mean_log_src, std_log_src, mean_log_tgt, std_log_tgt):
        f0 = np.asarray(f0, dtype=np.float64)
        out = np.copy(f0)
        nz = f0 > 0
        if np.any(nz):
            out[nz] = np.exp((np.log(f0[nz] + 1e-8) - mean_log_src) / (std_log_src + 1e-8) * std_log_tgt + mean_log_tgt)
        return out

    def decompose(self, wav: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        w = wav.astype(np.float64)
        f0, timeaxis = pyworld.harvest(w, self.cfg.sr, self.cfg.frame_period)
        sp = pyworld.cheaptrick(w, f0, timeaxis, self.cfg.sr)
        ap = pyworld.d4c(w, f0, timeaxis, self.cfg.sr)
        return f0, sp, ap

    def encode_sp(self, sp: np.ndarray) -> np.ndarray:
        return pyworld.code_spectral_envelope(sp, self.cfg.sr, self.cfg.dim)

    def decode_sp(self, coded_sp: np.ndarray) -> np.ndarray:
        fftlen = pyworld.get_cheaptrick_fft_size(self.cfg.sr)
        return pyworld.decode_spectral_envelope(coded_sp, self.cfg.sr, fftlen)

    def synthesize(self, f0: np.ndarray, decoded_sp: np.ndarray, ap: np.ndarray) -> np.ndarray:
        wav = pyworld.synthesize(f0, decoded_sp, ap, self.cfg.sr, self.cfg.frame_period)
        return wav.astype(np.float32)


# =========================
# Content Embedder
# =========================
class ContentEmbedder:
    """
    torchaudio.pipelines: Wav2Vec2/HuBERT Base
    반환: (T', Cc), 후에 WORLD 프레임 길이(T)에 선형보간 정렬
    """
    def __init__(self, backend: str = "wav2vec2", device: str = "cpu"):
        if not _TORCHAUDIO_OK:
            raise RuntimeError("torchaudio 가 필요합니다. (pip install torchaudio)")
        backend = backend.lower()
        if backend == "hubert":
            bundle = torchaudio.pipelines.HUBERT_BASE
        else:
            bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.model = bundle.get_model().to(device).eval()
        self.target_sr = bundle.sample_rate
        self.device = device

    @torch.inference_mode()
    def extract(self, wav: np.ndarray, sr: int) -> np.ndarray:
        x = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)  # (1,T)
        if sr != self.target_sr:
            x = ta_resample(x, sr, self.target_sr)
        feats = self.model(x)[0].squeeze(0).cpu().numpy()               # (T',Cc)
        return feats.astype(np.float32)


# =========================
# Feature Extraction
# =========================
class FeatureExtractor:
    def __init__(self, codec: WorldCodec, content: Optional[ContentEmbedder] = None):
        self.codec = codec
        self.content = content

    @staticmethod
    def _align_time(feat: np.ndarray, T: int) -> np.ndarray:
        """feat: (Tin,C) → (T,C) 선형보간 정렬"""
        if feat.shape[0] == T:
            return feat
        xp = np.linspace(0, 1, feat.shape[0], dtype=np.float64)
        xq = np.linspace(0, 1, T, dtype=np.float64)
        out = np.stack([np.interp(xq, xp, feat[:, c]) for c in range(feat.shape[1])], axis=1)
        return out.astype(np.float32)

    def extract(self, wavs: List[np.ndarray], sr: int, dim: int):
        f0s: List[np.ndarray] = []
        coded_sps: List[np.ndarray] = []
        contents: Optional[List[np.ndarray]] = [] if self.content else None
        logf0_seqs: List[np.ndarray] = []
        vuvs: List[np.ndarray] = []
        for wav in wavs:
            f0, sp, ap = self.codec.decompose(wav)      # f0 length = sp.shape[0] = T
            coded_sp = self.codec.encode_sp(sp)         # (T,dim)
            T = coded_sp.shape[0]

            # content
            if self.content:
                c = self.content.extract(wav, sr)       # (T',Cc)
                c = self._align_time(c, T)              # (T,Cc)
                assert contents is not None
                contents.append(c)

            # logf0, vuv
            vuv = (f0 > 0).astype(np.float32)[:, None]  # (T,1)
            logf0 = np.zeros((T, 1), dtype=np.float32)
            nz = vuv[:, 0] > 0
            if np.any(nz):
                logf0[nz, 0] = np.log(f0[nz] + 1e-8)

            f0s.append(f0)
            coded_sps.append(coded_sp)
            logf0_seqs.append(logf0)
            vuvs.append(vuv)
        return f0s, coded_sps, (contents if self.content else None), logf0_seqs, vuvs


# =========================
# Statistics / Normalization
# =========================
class StatsNormalizer:
    @staticmethod
    def compute_log_f0_statistics(f0s: List[np.ndarray]) -> Tuple[float, float]:
        all_f0 = np.concatenate(f0s, axis=0)
        valid = all_f0[all_f0 > 0]
        if valid.size == 0:
            return 0.0, 1.0
        logf0 = np.log(valid + 1e-8)
        return float(np.mean(logf0)), float(np.std(logf0) + 1e-8)

    @staticmethod
    def normalize_mcep_and_transpose(coded_sps: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """
        입력: List[(T,dim)] → 출력: List[(dim,T)], mean/std = (dim,1)
        """
        transposed = [sp.T for sp in coded_sps]  # (dim,T_i)
        concat = np.concatenate(transposed, axis=1)  # (dim, sum_T)
        mean = np.mean(concat, axis=1, keepdims=True)
        std = np.std(concat, axis=1, keepdims=True)
        std = np.maximum(std, 1e-8)
        normalized = [(sp - mean) / std for sp in transposed]
        return normalized, mean, std

    @staticmethod
    def compute_content_norm(contents: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        cat = np.concatenate(contents, axis=0)         # (sum_T, Cc)
        mean = np.mean(cat, axis=0, keepdims=True).T   # (Cc,1)
        std = np.std(cat, axis=0, keepdims=True).T
        std = np.maximum(std, 1e-8)
        return mean.astype(np.float32), std.astype(np.float32)

    @staticmethod
    def compute_logf0_global_norm(logf0_A: List[np.ndarray], logf0_B: List[np.ndarray]) -> Tuple[float, float]:
        arrs = []
        for seqs in (logf0_A, logf0_B):
            for l in seqs:
                nz = l[:, 0] > 0
                if np.any(nz):
                    arrs.append(l[nz, 0])
        if not arrs:
            return 0.0, 1.0
        allv = np.concatenate(arrs)
        return float(np.mean(allv)), float(np.std(allv) + 1e-8)


# =========================
# Pipeline
# =========================
class PreprocessPipeline:
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self.loader = AudioLoader(cfg)
        self.codec = WorldCodec(cfg)
        self.content = ContentEmbedder(cfg.content_backend, cfg.device) if (cfg.extract_content) else None
        self.extractor = FeatureExtractor(self.codec, self.content)
        self.normalizer = StatsNormalizer()

    def preprocess_and_cache(self, dir_A: str, dir_B: str, cache_dir: str) -> None:
        os.makedirs(cache_dir, exist_ok=True)

        # ----- Domain A -----
        print("🔹 Domain A 처리 중...")
        wavs_A = self.loader.load_wav_files(dir_A)
        f0s_A, coded_sps_A, contents_A, logf0_A, vuv_A = self.extractor.extract(wavs_A, self.cfg.sr, self.cfg.dim)
        mean_log_A, std_log_A = self.normalizer.compute_log_f0_statistics(f0s_A)
        norm_A, mean_A, std_A = self.normalizer.normalize_mcep_and_transpose(coded_sps_A)

        # ----- Domain B -----
        print("🔹 Domain B 처리 중...")
        wavs_B = self.loader.load_wav_files(dir_B)
        f0s_B, coded_sps_B, contents_B, logf0_B, vuv_B = self.extractor.extract(wavs_B, self.cfg.sr, self.cfg.dim)
        mean_log_B, std_log_B = self.normalizer.compute_log_f0_statistics(f0s_B)
        norm_B, mean_B, std_B = self.normalizer.normalize_mcep_and_transpose(coded_sps_B)

        # ----- Save (기본 캐시: 기존 파이프라인과 100% 호환) -----
        np.savez(os.path.join(cache_dir, "logf0s_normalization.npz"),
                 mean_A=mean_log_A, std_A=std_log_A, mean_B=mean_log_B, std_B=std_log_B)
        np.savez(os.path.join(cache_dir, "mcep_normalization.npz"),
                 mean_A=mean_A, std_A=std_A, mean_B=mean_B, std_B=std_B)
        CacheIO.save_pickle(norm_A, os.path.join(cache_dir, "coded_sps_A_norm.pickle"))
        CacheIO.save_pickle(norm_B, os.path.join(cache_dir, "coded_sps_B_norm.pickle"))

        # ----- 추가 리소스(콘텐츠/로그f0 시퀀스) -----
        if self.cfg.extract_content:
            assert contents_A is not None and contents_B is not None
            CacheIO.save_pickle(contents_A, os.path.join(cache_dir, "content_emb_A.pickle"))
            CacheIO.save_pickle(contents_B, os.path.join(cache_dir, "content_emb_B.pickle"))

            # content norm (A/B 각각 → 단순 평균으로 통합)
            mean_c_A, std_c_A = self.normalizer.compute_content_norm(contents_A)
            mean_c_B, std_c_B = self.normalizer.compute_content_norm(contents_B)
            mean_c = (mean_c_A + mean_c_B) / 2.0
            std_c = (std_c_A + std_c_B) / 2.0
            np.savez(os.path.join(cache_dir, "content_normalization.npz"), mean=mean_c, std=std_c)

            # logf0 sequences & vuv
            CacheIO.save_pickle(logf0_A, os.path.join(cache_dir, "logf0_seq_A.pickle"))
            CacheIO.save_pickle(logf0_B, os.path.join(cache_dir, "logf0_seq_B.pickle"))
            CacheIO.save_pickle(vuv_A, os.path.join(cache_dir, "vuv_A.pickle"))
            CacheIO.save_pickle(vuv_B, os.path.join(cache_dir, "vuv_B.pickle"))

            # global logf0 norm
            gmean, gstd = self.normalizer.compute_logf0_global_norm(logf0_A, logf0_B)
            np.savez(os.path.join(cache_dir, "logf0_global_normalization.npz"), mean=gmean, std=gstd)

        print(f"✅ 전처리 완료: 캐시는 '{cache_dir}'에 저장됨.")




# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Preprocess (WORLD/MCEP + optional content embedding)")
    # 기본값을 넣어 “그냥 실행”도 되게 구성
    ap.add_argument("--dirA", type=str, default=r"20s_singer_all",
                    help="도메인 A (예: 20대) wav 디렉토리")
    ap.add_argument("--dirB", type=str, default=r"50s_singer_all",
                    help="도메인 B (예: 50대) wav 디렉토리")
    ap.add_argument("--cache", type=str, default="./cache", help="캐시 출력 디렉토리")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--dim", type=int, default=36)
    ap.add_argument("--frame_period", type=float, default=5.0)
    # 콘텐츠 임베딩 옵션
    ap.add_argument("--no_content", action="store_true", help="콘텐츠 임베딩 비활성화")
    ap.add_argument("--backend", choices=["wav2vec2", "hubert"], default="wav2vec2")
    ap.add_argument("--device", default="cpu")
    # (선택) 무성 구간 트림
    ap.add_argument("--trim_silence", action="store_true")
    ap.add_argument("--trim_top_db", type=int, default=60)
    args = ap.parse_args()

    if not _TORCHAUDIO_OK and not args.no_content:
        print("⚠️ torchaudio 미탑재 상태이므로 콘텐츠 임베딩을 비활성화합니다. (--no_content 적용)")
        args.no_content = True

    cfg = PreprocessConfig(
        sr=args.sr,
        dim=args.dim,
        frame_period=args.frame_period,


        extract_content=not args.no_content,
        content_backend=args.backend,
        device=args.device,
        trim_silence=args.trim_silence,
        trim_top_db=args.trim_top_db,
    )

    pipeline = PreprocessPipeline(cfg)
    pipeline.preprocess_and_cache(args.dirA, args.dirB, args.cache)
