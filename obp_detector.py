"""
OneBitPitch (OBP) Detector Module
Implements the OneBitPitch algorithm for efficient pitch detection.

The OBP algorithm uses 1-bit quantization and autocorrelation to detect
fundamental frequency with lower computational cost compared to YIN or SWIPE.
"""

import numpy as np
from typing import List, Tuple, Optional


class OBPDetector:
    """
    Implements the OneBitPitch (OBP) algorithm for pitch detection.
    
    The algorithm works by:
    1. Converting the waveform to a binary signal (+1/-1)
    2. Applying autocorrelation to find periodicity
    3. Detecting fundamental frequency from autocorrelation peaks
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        frame_length: int = 2048,
        hop_length: int = 512,
        min_freq: float = 80.0,
        max_freq: float = 800.0
    ):
        """
        Initialize the OBPDetector.
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_length: Analysis frame length in samples
            hop_length: Hop length between frames in samples
            min_freq: Minimum detectable frequency in Hz
            max_freq: Maximum detectable frequency in Hz
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.min_freq = min_freq
        self.max_freq = max_freq
        
        # Calculate lag range from frequency range
        self.min_lag = int(sample_rate / max_freq)
        self.max_lag = int(sample_rate / min_freq)
    
    def quantize_to_one_bit(self, signal: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """
        Convert signal to 1-bit representation (+1 or -1).
        
        This is the key step that differentiates OBP from other algorithms.
        
        Args:
            signal: Input audio signal
            threshold: Threshold for binary conversion
            
        Returns:
            Binary signal with values +1 or -1
        """
        return np.where(signal >= threshold, 1.0, -1.0)
    
    def autocorrelation(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute autocorrelation of a signal using FFT for efficiency.
        
        Args:
            signal: Input signal
            
        Returns:
            Autocorrelation values
        """
        # Normalize signal
        signal = signal - np.mean(signal)
        
        # Compute autocorrelation via FFT
        fft = np.fft.rfft(signal, n=2 * len(signal))
        autocorr = np.fft.irfft(fft * np.conj(fft))
        autocorr = autocorr[:len(signal)]
        
        # Normalize
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        
        return autocorr
    
    def find_fundamental_period(self, autocorr: np.ndarray) -> Optional[int]:
        """
        Find the fundamental period from autocorrelation.
        
        Args:
            autocorr: Autocorrelation values
            
        Returns:
            Fundamental period in samples, or None if not found
        """
        # Search within the valid lag range
        search_range = autocorr[self.min_lag:min(self.max_lag, len(autocorr))]
        
        if len(search_range) == 0:
            return None
        
        # Find the maximum peak
        peak_idx = np.argmax(search_range)
        period = int(peak_idx) + self.min_lag
        
        # Check if peak is significant enough
        if search_range[peak_idx] < 0.1:  # Threshold for peak significance
            return None
        
        return period
    
    def detect_pitch_frame(self, frame: np.ndarray) -> Optional[float]:
        """
        Detect pitch in a single frame using OBP algorithm.
        
        Args:
            frame: Audio frame
            
        Returns:
            Detected frequency in Hz, or None if no pitch detected
        """
        # Check if frame has enough energy
        energy = np.sum(frame ** 2)
        if energy < 0.001:  # Silence threshold
            return None
        
        # Step 1: 1-bit quantization
        binary_signal = self.quantize_to_one_bit(frame)
        
        # Step 2: Autocorrelation
        autocorr = self.autocorrelation(binary_signal)
        
        # Step 3: Find fundamental period
        period = self.find_fundamental_period(autocorr)
        
        if period is None:
            return None
        
        # Convert period to frequency
        frequency = self.sample_rate / period
        
        # Validate frequency is in expected range
        if self.min_freq <= frequency <= self.max_freq:
            return frequency
        
        return None
    
    def detect_pitch(self, audio_data: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Detect pitch over entire audio signal.
        
        Args:
            audio_data: Input audio signal
            
        Returns:
            Tuple of (time_stamps, frequencies)
            - time_stamps: List of time stamps in seconds
            - frequencies: List of detected frequencies in Hz (0 for unvoiced)
        """
        num_frames = 1 + (len(audio_data) - self.frame_length) // self.hop_length
        
        time_stamps = []
        frequencies = []
        
        for i in range(num_frames):
            # Extract frame
            start = i * self.hop_length
            end = start + self.frame_length
            frame = audio_data[start:end]
            
            # Detect pitch in frame
            freq = self.detect_pitch_frame(frame)
            
            # Calculate time stamp
            time = start / self.sample_rate
            
            time_stamps.append(time)
            frequencies.append(freq if freq is not None else 0.0)
        
        return time_stamps, frequencies
    
    def smooth_pitch_contour(
        self,
        frequencies: List[float],
        window_size: int = 5
    ) -> List[float]:
        """
        Smooth pitch contour using median filtering.
        
        Args:
            frequencies: List of detected frequencies
            window_size: Size of smoothing window
            
        Returns:
            Smoothed frequencies
        """
        smoothed = []
        half_window = window_size // 2
        
        for i in range(len(frequencies)):
            start = max(0, i - half_window)
            end = min(len(frequencies), i + half_window + 1)
            window = [f for f in frequencies[start:end] if f > 0]
            
            if window:
                smoothed.append(np.median(window))
            else:
                smoothed.append(0.0)
        
        return smoothed
    
    def get_info(self) -> dict:
        """
        Get detector configuration information.
        
        Returns:
            Dictionary with detector settings
        """
        return {
            "algorithm": "OneBitPitch (OBP)",
            "sample_rate": self.sample_rate,
            "frame_length": self.frame_length,
            "hop_length": self.hop_length,
            "min_freq": self.min_freq,
            "max_freq": self.max_freq,
            "min_lag": self.min_lag,
            "max_lag": self.max_lag,
            "description": "1-bit quantization + autocorrelation for efficient pitch detection"
        }
