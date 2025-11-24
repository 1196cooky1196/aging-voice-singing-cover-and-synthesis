import time
start_time = time.time()

import os
import re
import tempfile
import sys
import numpy as np
import soundfile as sf
import librosa
import shutil
import stat
import subprocess
import whisper
from mutagen.mp3 import MP3, HeaderNotFoundError
from scipy.signal import butter, lfilter
import fnmatch
import importlib.util


class AudioConverter:
    def __init__(self, sr=16000):
        self.sr = sr

    def mp3_to_wav(self, input_mp3, output_wav):
        try:
            audio_info = MP3(input_mp3)
            print(f"🎵 MP3 길이: {audio_info.info.length:.2f}초")
            samples, _ = librosa.load(input_mp3, sr=self.sr, mono=True)
            sf.write(output_wav, samples, samplerate=self.sr, subtype='PCM_16')
            print(f"✅ MP3 → WAV 변환 완료: {output_wav}")
        except HeaderNotFoundError as e:
            print(f"❌ MP3 파일을 읽을 수 없습니다: {e}")


class BandpassFilter:
    def __init__(self, sr=16000, lowcut=300.0, highcut=3000.0):
        self.sr = sr
        self.lowcut = lowcut
        self.highcut = highcut

    def apply(self, y):
        nyquist = 0.5 * self.sr
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(N=4, Wn=[low, high], btype='band')
        return lfilter(b, a, y)


class VocalDetector:
    def __init__(self, sr=16000, rms_threshold=0.015, min_duration=1.0):
        self.sr = sr
        self.rms_threshold = rms_threshold
        self.min_duration = min_duration
        self.filter = BandpassFilter(sr=sr)
        # 무거우니 실제 STT 쓸 때 로딩하는 게 좋지만,
        # 네 원 코드를 존중해 즉시 로딩 유지
        self.model = whisper.load_model("small")

    def has_vocal_rms(self, audio_path):
        samples, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        filtered = self.filter.apply(samples)
        rms = librosa.feature.rms(y=filtered)[0]
        return np.max(rms) > self.rms_threshold

    def has_vocal_stt(self, audio_path):
        samples, _ = librosa.load(audio_path, sr=self.sr, mono=True)

        result_ko = self.model.transcribe(samples, language="ko", fp16=False)
        segments_ko = result_ko.get("segments", [])
        total_duration_ko = sum(s["end"] - s["start"] for s in segments_ko)
        if total_duration_ko >= self.min_duration:
            return True, result_ko["text"].strip()

        result_en = self.model.transcribe(samples, language="en", fp16=False)
        segments_en = result_en.get("segments", [])
        total_duration_en = sum(s["end"] - s["start"] for s in segments_en)
        if total_duration_en >= self.min_duration:
            return True, result_en["text"].strip()

        return False, ""


class VocalExtractor:
    """
    Demucs 실행 → 산출물 폴더에서 'vocals.wav'와 'no_vocals.wav/other.wav/accompaniment.wav'를
    '그대로' 집어와 최상위에 복사합니다. (판별/스코어/스왑 없음)

    - Windows cp949 출력 이슈 회피: ASCII 임시 파일명 + UTF-8 강제
    - OneDrive 잠금 회피: copy2 후 열어보기 검증
    """
    def __init__(self, model_name="htdemucs", sr=16000):
        self.model_name = model_name
        self.sr = sr

    # ---------- 내부 유틸 ----------
    def _remove_readonly(self, func, path, _):
        import os, stat
        os.chmod(path, stat.S_IWRITE)
        func(path)

    def _check_demucs_installed(self):
        import importlib.util
        return importlib.util.find_spec("demucs") is not None

    def _ascii_safe_copy(self, src_path: str, work_dir: str) -> str:
        import os, shutil, time
        os.makedirs(work_dir, exist_ok=True)
        _, ext = os.path.splitext(src_path)
        safe_name = f"input_tmp_{int(time.time())}{ext if ext else '.mp3'}"
        safe_path = os.path.join(work_dir, safe_name)
        shutil.copy2(src_path, safe_path)
        return safe_path

    def _verify_readable(self, path: str, retries: int = 5, sleep_sec: float = 0.3) -> None:
        """OneDrive 잠금 등으로 바로 못 여는 경우를 대비해 재시도."""
        import time, soundfile as sf
        last_err = None
        for _ in range(retries):
            try:
                with sf.SoundFile(path, 'r'):
                    return
            except Exception as e:
                last_err = e
                time.sleep(sleep_sec)
        raise RuntimeError(f"WAV 검증 실패: {path}\n{last_err}")

    def _find_stem_dir(self, out_dir: str, base_noext: str) -> str:
        """
        Demucs 출력 트리에서 stem들이 들어있는 최종 트랙 폴더를 찾는다.
        - 반드시 'vocals.wav'가 있고
        - 'no_vocals.wav' 또는 'other.wav' 또는 'accompaniment.wav' 중 하나가 있는 폴더를 선택
        - 파일명 일치, base_noext 일치 폴더를 우선
        """
        import os
        candidates = []
        for root, _, files in os.walk(out_dir):
            lfiles = [f.lower() for f in files]
            has_voc = "vocals.wav" in lfiles
            has_instr = any(x in lfiles for x in ("no_vocals.wav", "other.wav", "accompaniment.wav"))
            if has_voc and has_instr:
                score = 0
                if os.path.basename(root).lower() == base_noext.lower():
                    score += 2
                # 최근 수정시간 가점
                try:
                    mtime = max(os.path.getmtime(os.path.join(root, f)) for f in files)
                except Exception:
                    mtime = 0.0
                candidates.append((score, mtime, root))

        if not candidates:
            return ""

        # 점수(우선순위) → 최신 mtime 순으로 선택
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][2]

    # ---------- 메인 ----------
    def extract(self, input_audio, output_dir):
        import os, sys, shutil, subprocess

        if not self._check_demucs_installed():
            raise RuntimeError("demucs가 설치되어 있지 않습니다. 'pip install demucs' 후 다시 시도하세요.")

        os.makedirs(output_dir, exist_ok=True)
        final_vocals_path = os.path.join(output_dir, "vocals.wav")
        final_instr_path  = os.path.join(output_dir, "no_vocals.wav")

        # 1) 입력 파일명 안전화 (cp949 출력 이슈 회피)
        tmp_dir = os.path.join(output_dir, "_tmp_input")
        os.makedirs(tmp_dir, exist_ok=True)
        safe_input = self._ascii_safe_copy(input_audio, tmp_dir)

        base_noext = os.path.splitext(os.path.basename(safe_input))[0]

        print("🎶 보컬 및 반주 추출 중... (Demucs)")
        # UTF-8 강제
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals", safe_input, "--out", output_dir]
        completed = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",  errors="replace", env=env)

        if completed.returncode != 0:
            print("❌ Demucs 실행 실패")
            if completed.stderr:
                print(completed.stderr)
            try:
                shutil.rmtree(tmp_dir, onerror=self._remove_readonly)
            except Exception:
                pass
            raise RuntimeError("Demucs 실행에 실패했습니다.")

        # 2) stem 폴더 찾기 (정확 파일명 매칭)
        stem_dir = self._find_stem_dir(output_dir, base_noext)
        if not stem_dir:
            # 힌트
            tree = []
            for root, _, files in os.walk(output_dir):
                for f in files:
                    tree.append(os.path.join(root, f))
            hint = "\n".join(tree[:20])
            try:
                shutil.rmtree(tmp_dir, onerror=self._remove_readonly)
            except Exception:
                pass
            raise RuntimeError(
                "Demucs 출력에서 stem 폴더를 찾지 못했습니다.\n"
                f"검색 경로: {output_dir}\n"
                f"발견된 파일 일부:\n{hint}"
            )

        # 3) 정확한 파일명으로만 매핑 (이름 그대로 복사, 스왑/평가 없음)
        src_vocals = os.path.join(stem_dir, "vocals.wav")
        # 반주 후보: no_vocals > other > accompaniment 우선순
        for cand in ("no_vocals.wav", "other.wav", "accompaniment.wav"):
            cand_path = os.path.join(stem_dir, cand)
            if os.path.exists(cand_path):
                src_instr = cand_path
                break
        else:
            src_instr = None

        if not os.path.exists(src_vocals):
            raise RuntimeError(f"'{stem_dir}'에 vocals.wav가 없습니다. 실제 출력 구조를 확인하세요.")

        # 4) 최상위로 복사(copy2) 후, 파일 열리는지 검증
        shutil.copy2(src_vocals, final_vocals_path)
        if src_instr:
            shutil.copy2(src_instr, final_instr_path)

        self._verify_readable(final_vocals_path)

        # 5) 임시/중간 폴더 정리(최상위 파일은 남김)
        try:
            shutil.rmtree(tmp_dir, onerror=self._remove_readonly)
        except Exception:
            pass

        # output_dir 바로 아래 생성된 모델/트랙 하위 폴더 제거
        for entry in list(os.scandir(output_dir)):
            if entry.is_dir():
                try:
                    shutil.rmtree(entry.path, onerror=self._remove_readonly)
                except Exception:
                    pass

        print(f"✅ 보컬 파일: {final_vocals_path}")
        return final_vocals_path



class AudioSplitter:
    def __init__(self, segment_length=5, sr=16000):
        self.segment_length = segment_length
        self.sr = sr

    def split(self, input_wav, output_dir, detector: VocalDetector):
        import soundfile as sf
        os.makedirs(output_dir, exist_ok=True)

        # librosa.load 대신 soundfile.read 사용 (WAV 직독)
        samples, file_sr = sf.read(input_wav, dtype='float32', always_2d=False)
        if samples.ndim == 2:  # 스테레오면 모노 합성
            samples = samples.mean(axis=1)
        if file_sr != self.sr:
            # 필요 시 리샘플 (librosa.resample 사용)
            samples = librosa.resample(samples, orig_sr=file_sr, target_sr=self.sr)

        segment_samples = int(self.segment_length * self.sr)
        saved_segments = 0

        for i in range(0, len(samples), segment_samples):
            segment = samples[i:i + segment_samples]
            if len(segment) == segment_samples:
                segment_path = os.path.join(output_dir, f"part_{saved_segments + 1}.wav")
                sf.write(segment_path, segment, samplerate=self.sr, subtype='PCM_16')

                if detector.has_vocal_rms(segment_path):
                    saved_segments += 1
                else:
                    try:
                        os.remove(segment_path)
                    except FileNotFoundError:
                        pass

        print(f"✅ {saved_segments} 개의 WAV 파일이 생성되었습니다!")



class VocalSegmentFilter:
    def __init__(self, detector: VocalDetector):
        self.detector = detector

    def filter(self, folder_path, output_path):
        os.makedirs(output_path, exist_ok=True)
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".wav"):
                input_path = os.path.join(folder_path, file_name)
                is_vocal, transcript = self.detector.has_vocal_stt(input_path)
                if is_vocal:
                    shutil.copy(input_path, os.path.join(output_path, file_name))
                    print(f"✅ 보컬로 판단: {file_name} - {transcript}")
                else:
                    print(f"❌ 보컬 없음: {file_name}")
        print(f"✅ 최종 보컬 파일 저장 완료: {output_path}")


# =====================
# main 실행부
# =====================
if __name__ == "__main__":
    base_dir = r"singer\singer_yb50"
    input_mp3 = os.path.join(
        r"song_collection\old\[2021 MBC 가요대제전] YB - 흰수염고래 (YB - Blue whale), MBC 211231 방송 - MBCkpop.mp3"
    )
    temp_wav = os.path.join(base_dir, "temp_audio.wav")
    split_music_dir = os.path.join(base_dir, "split_audiowithmusic_voice")
    split_vocals_dir = os.path.join(base_dir, "split_only_voice")
    final_music_dir = os.path.join(base_dir, "final_audiowithmusic_voice")
    final_vocals_dir = os.path.join(base_dir, "final_only_voice")
    vocals_dir = os.path.join(base_dir, "vocals")

    os.makedirs(base_dir, exist_ok=True)

    converter = AudioConverter()
    extractor = VocalExtractor()
    detector = VocalDetector()
    splitter = AudioSplitter()
    filterer = VocalSegmentFilter(detector)

    print("\n💡 [1] MP3 → WAV 변환 및 5초 분할 (RMS 필터)")
    converter.mp3_to_wav(input_mp3, temp_wav)
    splitter.split(temp_wav, split_music_dir, detector)
    try:
        os.remove(temp_wav)
    except FileNotFoundError:
        pass

    print("\n💡 [2] Demucs로 보컬 분리 후 5초 분할 (RMS 필터)")
    vocals_wav = extractor.extract(input_mp3, vocals_dir)   # ← robust하게 수정
    splitter.split(vocals_wav, split_vocals_dir, detector)

    # (필요하면) Whisper STT 2차 필터 추가 사용
    # print("\n💡 [3] Whisper 기반 STT 검증")
    # filterer.filter(split_music_dir, final_music_dir)
    # filterer.filter(split_vocals_dir, final_vocals_dir)

    print("✅ 전체 작업 완료")

end_time = time.time()
print(f"⏱️ Total time: {end_time - start_time:.2f} seconds")
