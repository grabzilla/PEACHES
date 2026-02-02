"""
Peaches - Voice-to-MIDI Converter using OneBitPitch Algorithm

A modular Python tool for converting vocal recordings to MIDI sequences.
"""

from peaches.audio_processor import AudioProcessor
from peaches.noise_reducer import NoiseReducer
from peaches.obp_detector import OBPDetector
from peaches.midi_converter import MIDIConverter
from peaches.visualizer import Visualizer

__version__ = "1.0.0"
__all__ = [
    "AudioProcessor",
    "NoiseReducer", 
    "OBPDetector",
    "MIDIConverter",
    "Visualizer"
]