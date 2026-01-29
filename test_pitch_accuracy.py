"""
Test pitch detection accuracy
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from obp_detector import OBPDetector


def generate_sine_wave(frequency, duration, sample_rate=44100):
    """Generate a pure sine wave for testing."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    return np.sin(2 * np.pi * frequency * t)


def test_single_frequency():
    """Test pitch detection on a single frequency."""
    print("Testing single frequency detection...")
    
    # Generate test signal
    target_freq = 440.0  # A4
    sample_rate = 44100
    duration = 1.0
    signal = generate_sine_wave(target_freq, duration, sample_rate)
    
    # Detect pitch
    detector = OBPDetector(sample_rate=sample_rate, min_freq=80, max_freq=800)
    time_stamps, frequencies = detector.detect_pitch(signal)
    
    # Calculate accuracy
    valid_freqs = [f for f in frequencies if f > 0]
    if valid_freqs:
        mean_freq = np.mean(valid_freqs)
        error = abs(mean_freq - target_freq)
        accuracy = 100 * (1 - error / target_freq)
        
        print(f"  Target frequency: {target_freq:.2f} Hz")
        print(f"  Detected frequency: {mean_freq:.2f} Hz")
        print(f"  Error: {error:.2f} Hz")
        print(f"  Accuracy: {accuracy:.2f}%")
        
        return accuracy > 95  # 95% accuracy threshold
    
    return False


def test_multiple_frequencies():
    """Test pitch detection on multiple frequencies."""
    print("\nTesting multiple frequency detection...")
    
    test_freqs = [220.0, 330.0, 440.0, 550.0]  # A3, E4, A4, C#5
    sample_rate = 44100
    duration = 0.5
    
    for target_freq in test_freqs:
        signal = generate_sine_wave(target_freq, duration, sample_rate)
        
        detector = OBPDetector(sample_rate=sample_rate, min_freq=80, max_freq=800)
        time_stamps, frequencies = detector.detect_pitch(signal)
        
        valid_freqs = [f for f in frequencies if f > 0]
        if valid_freqs:
            mean_freq = np.mean(valid_freqs)
            error = abs(mean_freq - target_freq)
            
            print(f"  {target_freq:.0f} Hz -> {mean_freq:.2f} Hz (error: {error:.2f} Hz)")


if __name__ == "__main__":
    print("=" * 60)
    print("Pitch Detection Accuracy Tests")
    print("=" * 60)
    
    test_single_frequency()
    test_multiple_frequencies()
    
    print("\n" + "=" * 60)
    print("Tests complete!")
    print("=" * 60)