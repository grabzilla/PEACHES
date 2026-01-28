"""
Audio Processor Module
Handles loading and preprocessing of audio files for pitch detection.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional


class AudioProcessor:
    """Loads and preprocesses audio files for pitch detection."""
    
    def __init__(self, target_sr: int = 44100):
        """
        Initialize the AudioProcessor.
        
        Args:
            target_sr: Target sample rate in Hz (default: 44100)
        """
        self.target_sr = target_sr
        self.audio_data = None
        self.sample_rate = None
        
    def load_audio(self, file_path: str, mono: bool = True) -> Tuple[np.ndarray, int]:
        """
        Load an audio file (WAV or MP3) and convert to mono if needed.
        
        Args:
            file_path: Path to the audio file
            mono: Convert to mono if True (default: True)
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load audio file using librosa
            audio_data, sample_rate = librosa.load(
                file_path,
                sr=self.target_sr,
                mono=mono
            )
            
            self.audio_data = audio_data
            self.sample_rate = sample_rate
            
            print(f"Audio loaded successfully:")
            print(f"  - Duration: {len(audio_data) / sample_rate:.2f} seconds")
            print(f"  - Sample rate: {sample_rate} Hz")
            print(f"  - Samples: {len(audio_data)}")
            
            return audio_data, sample_rate
            
        except Exception as e:
            raise ValueError(f"Error loading audio file: {str(e)}")
    
    def ensure_mono(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Convert stereo audio to mono by averaging channels.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Mono audio data
        """
        if audio_data.ndim > 1:
            return np.mean(audio_data, axis=0)
        return audio_data
    
    def resample(self, audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio_data: Input audio data
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio data
        """
        if orig_sr != target_sr:
            audio_data = librosa.resample(
                audio_data,
                orig_sr=orig_sr,
                target_sr=target_sr
            )
        return audio_data
    
    def normalize(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Normalized audio data
        """
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data
    
    def preprocess(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Complete preprocessing pipeline: load, mono conversion, and normalization.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (preprocessed_audio, sample_rate)
        """
        # Load audio
        audio_data, sample_rate = self.load_audio(file_path)
        
        # Ensure mono
        audio_data = self.ensure_mono(audio_data)
        
        # Normalize
        audio_data = self.normalize(audio_data)
        
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        
        return audio_data, sample_rate
    
    def get_duration(self) -> float:
        """
        Get the duration of the loaded audio in seconds.
        
        Returns:
            Duration in seconds
        """
        if self.audio_data is None or self.sample_rate is None:
            raise ValueError("No audio loaded")
        return len(self.audio_data) / self.sample_rate
    
    def get_info(self) -> dict:
        """
        Get information about the loaded audio.
        
        Returns:
            Dictionary with audio information
        """
        if self.audio_data is None:
            return {"status": "No audio loaded"}
        
        return {
            "duration_seconds": self.get_duration(),
            "sample_rate": self.sample_rate,
            "num_samples": len(self.audio_data),
            "max_amplitude": float(np.max(np.abs(self.audio_data))),
            "mean_amplitude": float(np.mean(np.abs(self.audio_data)))
        }
