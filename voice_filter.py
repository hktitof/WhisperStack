"""
voice_filter.py — Target Speaker Verification and Acoustic Gating for WhisperStack.

This module provides real-time speaker verification using SpeechBrain's ECAPA-TDNN
model (192-dimensional x-vector embeddings). It filters out foreign voices
(e.g., podcasts, YouTube videos, phone audio, other speakers in the room)
so that WhisperStack ONLY transcribes when the enrolled target speaker is talking.

Architecture: SpeechBrain ECAPA-TDNN + Cosine Similarity Gating
"""

from __future__ import annotations

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Optional

# Default paths and parameters
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "speechbrain" / "spkrec-ecapa-voxceleb"
DEFAULT_PROFILE_PATH = PROJECT_ROOT / "voiceprints" / "target_voiceprint.npy"
DEFAULT_THRESHOLD = 0.30  # Cosine similarity threshold for target speaker acceptance
SAMPLE_RATE = 16_000


class VoiceFilter:
    """
    Real-time speaker verification and audio masking gate.

    Attributes:
        profile_path: Path to the enrolled speaker numpy embedding (.npy).
        threshold: Cosine similarity threshold for speaker acceptance.
        enrolled_embedding: Loaded 192-dimensional unit vector of the target speaker.
        classifier: SpeechBrain ECAPA-TDNN pretrained encoder.
    """

    def __init__(
        self,
        profile_path: str | Path | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        device: str | None = None,
    ):
        self.profile_path = Path(profile_path) if profile_path else DEFAULT_PROFILE_PATH
        self.threshold = float(threshold)
        if device is None:
            import torch

            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
        self.classifier = None
        self.enrolled_embedding: Optional[np.ndarray] = None
        self._is_ready = False

        # Attempt to load classifier and profile
        self._initialize_model()
        self.load_profile()

    def _initialize_model(self) -> bool:
        """Loads the ECAPA-TDNN encoder from local cache without redownloading."""
        try:
            # Avoid offline blocking issues if already downloaded
            os.environ.pop("HF_HUB_OFFLINE", None)
            from speechbrain.inference.speaker import EncoderClassifier

            DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(DEFAULT_CACHE_DIR),
                run_opts={"device": self.device},
            )
            self._is_ready = True
            print(
                "🛡️ [VoiceFilter] ECAPA-TDNN speaker encoder initialized successfully."
            )
            return True
        except Exception as exc:
            print(f"⚠️ [VoiceFilter] Could not load speaker encoder: {exc}")
            self.classifier = None
            self._is_ready = False
            return False

    def load_profile(self) -> bool:
        """Loads the enrolled target speaker embedding from disk."""
        target_path = self.profile_path
        if not target_path.exists():
            vp_dir = self.profile_path.parent
            if vp_dir.exists():
                npy_files = list(vp_dir.glob("*.npy"))
                if npy_files:
                    target_path = npy_files[0]

        if not target_path.exists():
            self.enrolled_embedding = None
            return False

        try:
            emb = np.load(target_path)
            # Ensure it is a 1D float32 normalized vector
            emb = emb.flatten().astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            self.enrolled_embedding = emb
            print(
                f"✅ [VoiceFilter] Loaded target voiceprint: {target_path.name} (dim={len(emb)})"
            )
            return True
        except Exception as exc:
            print(f"⚠️ [VoiceFilter] Failed to load voice profile: {exc}")
            self.enrolled_embedding = None
            return False

    def save_profile(self, embedding: np.ndarray) -> bool:
        """Saves an enrolled speaker embedding to disk."""
        try:
            self.profile_path.parent.mkdir(parents=True, exist_ok=True)
            # Normalize before saving
            emb = embedding.flatten().astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            np.save(self.profile_path, emb)
            self.enrolled_embedding = emb
            print(
                f"💾 [VoiceFilter] Target voiceprint saved successfully to: {self.profile_path}"
            )
            return True
        except Exception as exc:
            print(f"⚠️ [VoiceFilter] Failed to save voice profile: {exc}")
            return False

    def compute_embedding(
        self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE
    ) -> Optional[np.ndarray]:
        """
        Extracts a 192-dimensional unit vector embedding from a raw audio array.

        Args:
            audio: 1D numpy array of audio samples (int16 or float32).
            sample_rate: Audio sampling rate (expected 16000 Hz).
        """
        if not self._is_ready or self.classifier is None:
            return None

        # CRITICAL FIX: Always flatten audio to 1D to prevent multi-dimensional tensor shape mismatch in PyTorch
        audio = np.asarray(audio).flatten()

        if len(audio) < int(sample_rate * 0.4):
            # Audio is too short for reliable speaker identification (<400ms)
            return None

        # Convert to float32 in range [-1.0, 1.0]
        if audio.dtype == np.int16:
            audio_f32 = audio.astype(np.float32) / 32768.0
        else:
            audio_f32 = audio.astype(np.float32)
            # If peak exceeds 1.0, normalize safely
            max_val = np.max(np.abs(audio_f32))
            if max_val > 1.0:
                audio_f32 = audio_f32 / max_val

        # Convert to PyTorch tensor [batch=1, time]
        tensor = torch.from_numpy(audio_f32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            try:
                emb = self.classifier.encode_batch(tensor)  # shape: (1, 1, 192)
                emb_np = emb.squeeze().cpu().numpy().astype(np.float32)
                norm = np.linalg.norm(emb_np)
                if norm > 0:
                    emb_np = emb_np / norm
                return emb_np
            except Exception as exc:
                print(f"⚠️ [VoiceFilter] Error computing speaker embedding: {exc}")
                return None

    def verify_audio(
        self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE
    ) -> Tuple[bool, float]:
        """
        Verifies if the audio utterance matches the enrolled target speaker.

        Returns:
            (is_match, similarity_score)
            If no profile is enrolled, defaults to (True, 1.0) so recording is uninterrupted.
        """
        if self.enrolled_embedding is None:
            # Hot-reload if the profile was recently created on disk
            if self.profile_path.exists():
                self.load_profile()
            if self.enrolled_embedding is None:
                return True, 1.0

        emb = self.compute_embedding(audio, sample_rate=sample_rate)
        if emb is None:
            # Inconclusive/too short -> pass through defensively
            return True, 0.0

        # Cosine similarity between two unit vectors is simply their dot product
        similarity = float(np.dot(self.enrolled_embedding, emb))
        is_match = similarity >= self.threshold

        return is_match, similarity

    def filter_utterance(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        window_sec: float = 2.0,
        hop_sec: float = 1.0,
        is_tail: bool = False,
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Surgically inspects an audio recording across sliding windows:
        - If NO window matches the enrolled speaker (e.g. only foreign podcast/phone audio),
          the entire audio is marked as foreign and rejected.
        - If the enrolled speaker is detected, returns the audio with confirmation.

        Returns:
            (processed_audio, has_target_speaker, peak_similarity)
        """
        if self.enrolled_embedding is None and self.profile_path.exists():
            self.load_profile()

        if self.enrolled_embedding is None or not self._is_ready:
            return audio, True, 1.0

        # CRITICAL FIX: Always flatten audio to 1D array
        audio = np.asarray(audio).flatten()
        total_samples = len(audio)
        window_samples = int(window_sec * sample_rate)
        hop_samples = int(hop_sec * sample_rate)

        # For short recordings under window size, check the whole segment
        if total_samples <= window_samples:
            is_match, sim = self.verify_audio(audio, sample_rate=sample_rate)
            # If this is the tail of an already-verified target speaker session,
            # never zero out trailing syllables due to short-slice embedding variance.
            if is_match or is_tail or sim >= 0.18:
                return audio, True, max(sim, self.threshold)
            else:
                return np.zeros_like(audio), False, sim

        # Evaluate sliding windows across the utterance with active masking
        window_scores: list[float] = []
        target_detected = False
        peak_score = -1.0
        mask = np.zeros(total_samples, dtype=np.float32)

        for start in range(0, total_samples - window_samples + 1, hop_samples):
            chunk = audio[start : start + window_samples]
            chunk_peak = np.max(np.abs(chunk))

            # Skip dead silence chunks to avoid wasting inference time
            if chunk_peak < 300:
                window_scores.append(0.0)
                continue

            emb = self.compute_embedding(chunk, sample_rate=sample_rate)
            if emb is not None:
                sim = float(np.dot(self.enrolled_embedding, emb))
                window_scores.append(sim)
                if sim > peak_score:
                    peak_score = sim
                if sim >= self.threshold:
                    target_detected = True
                    # Unmask target speaker window
                    mask[start : start + window_samples] = 1.0
            else:
                window_scores.append(0.0)

        # If target speaker was never detected across any window, reject entirely
        if not target_detected:
            return np.zeros_like(audio), False, max(0.0, peak_score)

        # Hangover Bridge (1.5s): Prevent brief acoustic dips or overlapping speaker spikes from chopping syllables mid-sentence
        hangover_samples = int(1.5 * sample_rate)
        dilated_mask = np.zeros_like(mask)
        active_runs = np.where(mask > 0.5)[0]
        if len(active_runs) > 0:
            for idx in active_runs:
                end_pos = min(total_samples, idx + hangover_samples)
                dilated_mask[idx:end_pos] = 1.0
        else:
            dilated_mask = mask

        # Surgical Masking: Zero out foreign speaker sections so only target speaker speech is transcribed
        filtered_audio = (audio.astype(np.float32) * dilated_mask).astype(audio.dtype)
        return filtered_audio, True, peak_score


# Standalone CLI test & enrollment tool
if __name__ == "__main__":
    import sys

    print("=== WhisperStack VoiceFilter Diagnostic & Test ===")
    vf = VoiceFilter()

    if "--test" in sys.argv:
        print("Running synthetic benchmark...")
        dummy = np.random.randn(32000).astype(np.float32)
        emb = vf.compute_embedding(dummy)
        print(
            "Embedding test successful. Shape:",
            emb.shape if emb is not None else "None",
        )
