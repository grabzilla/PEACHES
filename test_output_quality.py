"""
Test MIDI output quality
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from obp_detector import OBPDetector
from midi_converter import MIDIConverter


def test_frequency_to_midi_conversion():
    """Test frequency to MIDI note conversion."""
    print("Testing frequency to MIDI conversion...")
    
    converter = MIDIConverter()
    
    # Test known frequencies
    test_cases = [
        (440.0, 69),   # A4
        (261.63, 60),  # C4 (Middle C)
        (880.0, 81),   # A5
        (220.0, 57),   # A3
    ]
    
    all_passed = True
    for freq, expected_midi in test_cases:
        midi_note = converter.frequency_to_midi(freq)
        passed = midi_note == expected_midi
        status = "✓" if passed else "✗"
        
        print(f"  {status} {freq:.2f} Hz -> MIDI {midi_note} (expected {expected_midi})")
        
        if not passed:
            all_passed = False
    
    return all_passed


def test_note_segmentation():
    """Test note segmentation from pitch data."""
    print("\nTesting note segmentation...")
    
    # Create simple pitch sequence
    time_stamps = [i * 0.1 for i in range(20)]
    frequencies = [440.0] * 10 + [0.0] * 5 + [550.0] * 5
    
    converter = MIDIConverter()
    notes = converter.segment_notes(time_stamps, frequencies, min_note_duration=0.1)
    
    print(f"  Detected {len(notes)} notes from test sequence")
    
    for i, (start, duration, midi_note, velocity) in enumerate(notes):
        note_name = converter.get_note_name(midi_note)
        print(f"    Note {i+1}: {note_name} at {start:.2f}s for {duration:.2f}s")
    
    return len(notes) > 0


def test_midi_note_names():
    """Test MIDI note name conversion."""
    print("\nTesting MIDI note names...")
    
    converter = MIDIConverter()
    
    test_cases = [
        (60, "C4"),
        (69, "A4"),
        (72, "C5"),
        (48, "C3"),
    ]
    
    all_passed = True
    for midi_note, expected_name in test_cases:
        note_name = converter.get_note_name(midi_note)
        passed = note_name == expected_name
        status = "✓" if passed else "✗"
        
        print(f"  {status} MIDI {midi_note} -> {note_name} (expected {expected_name})")
        
        if not passed:
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    print("=" * 60)
    print("MIDI Output Quality Tests")
    print("=" * 60)
    
    test_frequency_to_midi_conversion()
    test_note_segmentation()
    test_midi_note_names()
    
    print("\n" + "=" * 60)
    print("Tests complete!")
    print("=" * 60)