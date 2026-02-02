"""
Noise Reducer Module
Applies noise filtering and normalization to audio signals.
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, wiener
from typing import Tuple


class NoiseReducer:
    """Applies noise reduction and filtering to audio signals."""
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize the NoiseReducer.
        
        Args:
            sample_rate: Sample rate of the audio in Hz
        """
        self.sample_rate = sample_rate
    
    def apply_lowpass_filter(
        self, 
        audio_data: np.ndarray, 
        cutoff_freq: float = 3000.0,
        order: int = 5
    ) -> np.ndarray:
        """
        Apply a Butterworth low-pass filter to remove high-frequency noise.
        
        Args:
            audio_data: Input audio data
            cutoff_freq: Cutoff frequency in Hz (default: 3000 Hz)
            order: Filter order (default: 5)
            
        Returns:
            Filtered audio data
        """
        nyquist = self.sample_rate / 2.0
        normal_cutoff = cutoff_freq / nyquist
        
        # Design Butterworth filter
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        
        # Apply filter
        filtered_audio = filtfilt(b, a, audio_data)
        
        return filtered_audio
    
    def apply_highpass_filter(
        self,
        audio_data: np.ndarray,
        cutoff_freq: float = 80.0,
        order: int = 5
    ) -> np.ndarray:
        """
        Apply a Butterworth high-pass filter to remove low-frequency rumble.
        
        Args:
            audio_data: Input audio data
            cutoff_freq: Cutoff frequency in Hz (default: 80 Hz)
            order: Filter order (default: 5)
            
        Returns:
            Filtered audio data
        """
        nyquist = self.sample_rate / 2.0
        normal_cutoff = cutoff_freq / nyquist
        
        # Design Butterworth filter
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        
        # Apply filter
        filtered_audio = filtfilt(b, a, audio_data)
        
        return filtered_audio
    
    def apply_bandpass_filter(
        self,
        audio_data: np.ndarray,
        low_freq: float = 80.0,
        high_freq: float = 3000.0,
        order: int = 5
    ) -> np.ndarray:
        """
        Apply a bandpass filter to keep only vocal frequency range.
        
        Args:
            audio_data: Input audio data
            low_freq: Lower cutoff frequency in Hz (default: 80 Hz)
            high_freq: Upper cutoff frequency in Hz (default: 3000 Hz)
            order: Filter order (default: 5)
            
        Returns:
            Filtered audio data
        """
        nyquist = self.sample_rate / 2.0
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Design Butterworth bandpass filter
        b, a = butter(order, [low, high], btype='band', analog=False)
        
        # Apply filter
        filtered_audio = filtfilt(b, a, audio_data)
        
        return filtered_audio
    
    def reduce_noise_spectral_gating(
        self,
        audio_data: np.ndarray,
        noise_threshold: float = 0.02
    ) -> np.ndarray:
        """
        Simple spectral gating noise reduction.
        Attenuates signals below a threshold.
        
        Args:
            audio_data: Input audio data
            noise_threshold: Amplitude threshold for noise gate
            
        Returns:
            Noise-reduced audio data
        """
        # Create a copy
        processed = audio_data.copy()
        
        # Apply soft gating
        mask = np.abs(processed) > noise_threshold
        processed = processed * mask
        
        return processed
    
    def normalize_volume(self, audio_data: np.ndarray, target_level: float = 0.7) -> np.ndarray:
        """
        Normalize audio to target RMS level.
        
        Args:
            audio_data: Input audio data
            target_level: Target RMS level (default: 0.7)
            
        Returns:
            Normalized audio data
        """
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio_data ** 2))
        
        if rms > 0:
            # Scale to target level
            scaling_factor = target_level / rms
            normalized = audio_data * scaling_factor
            
            # Ensure we don't clip
            max_val = np.max(np.abs(normalized))
            if max_val > 1.0:
                normalized = normalized / max_val
            
            return normalized
        
        return audio_data
    
    def estimate_snr(self, original: np.ndarray, processed: np.ndarray) -> float:
        """
        Estimate Signal-to-Noise Ratio improvement.
        
        Args:
            original: Original audio data
            processed: Processed audio data
            
        Returns:
            SNR in dB
        """
        signal_power = np.mean(processed ** 2)
        noise = original - processed
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            return snr
        return float('inf')
    
    def process(
        self,
        audio_data: np.ndarray,
        apply_bandpass: bool = True,
        apply_normalization: bool = True,
        low_freq: float = 80.0,
        high_freq: float = 3000.0
    ) -> np.ndarray:
        """
        Complete noise reduction pipeline.
        
        Args:
            audio_data: Input audio data
            apply_bandpass: Apply bandpass filter
            apply_normalization: Apply volume normalization
            low_freq: Lower frequency for bandpass filter
            high_freq: Upper frequency for bandpass filter
            
        Returns:
            Processed audio data
        """
        processed = audio_data.copy()
        
        # Apply bandpass filter for vocal range
        if apply_bandpass:
            processed = self.apply_bandpass_filter(
                processed,
                low_freq=low_freq,
                high_freq=high_freq
            )
        
        # Normalize volume
        if apply_normalization:
            processed = self.normalize_volume(processed)
        
        # Calculate SNR improvement
        snr = self.estimate_snr(audio_data, processed)
        print(f"Noise reduction complete. Estimated SNR improvement: {snr:.2f} dB")
        
        return processed
