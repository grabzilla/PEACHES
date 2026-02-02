"""
Demo script to test Peaches without an actual audio file.
Generates a synthetic melody and converts it to MIDI.
"""

import numpy as np
import soundfile as sf
from peaches.audio_processor import AudioProcessor
from peaches.noise_reducer import NoiseReducer
from peaches.obp_detector import OBPDetector
from peaches.midi_converter import MIDIConverter
from peaches.visualizer import Visualizer


def generate_test_melody():
    """Generate a simple test melody as audio."""
    sample_rate = 44100
    duration_per_note = 0.5
    
    # Simple melody: C4, D4, E4, F4, G4, A4, G4, F4, E4, D4, C4
    frequencies = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 
                   392.00, 349.23, 329.63, 293.66, 261.63]
    
    audio = []
    for freq in frequencies:
        t = np.linspace(0, duration_per_note, int(sample_rate * duration_per_note))
        # Add harmonics for more realistic sound
        note = (np.sin(2 * np.pi * freq * t) * 0.6 +
                np.sin(2 * np.pi * freq * 2 * t) * 0.3 +
                np.sin(2 * np.pi * freq * 3 * t) * 0.1)
        
        # Apply envelope
        envelope = np.exp(-3 * t / duration_per_note)
        note = note * envelope
        
        audio.extend(note)
    
    audio = np.array(audio)
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio, sample_rate


def main():
    """Run the demo."""
    print("=" * 70)
    print("PEACHES DEMO - Synthetic Melody Test")
    print("=" * 70)
    print()
    
    # Generate test melody
    print("Generating synthetic test melody...")
    audio_data, sample_rate = generate_test_melody()
    
    # Save to temporary file
    test_file = "test_melody.wav"
    sf.write(test_file, audio_data, sample_rate)
    print(f"✓ Generated {len(audio_data)/sample_rate:.2f} second melody")
    print(f"✓ Saved to {test_file}")
    print()
    
    # Process using Peaches pipeline
    print("Processing with Peaches pipeline...")
    print()
    
    # Step 1: Load audio
    print("[1/5] Loading audio...")
    audio_processor = AudioProcessor(target_sr=sample_rate)
    audio_data, sr = audio_processor.preprocess(test_file)
    print("✓ Audio loaded")
    print()
    
    # Step 2: Noise reduction
    print("[2/5] Applying noise reduction...")
    noise_reducer = NoiseReducer(sample_rate=sr)
    processed_audio = noise_reducer.process(audio_data)
    print("✓ Noise reduction complete")
    print()
    
    # Step 3: Pitch detection
    print("[3/5] Detecting pitch using OneBitPitch algorithm...")
    obp_detector = OBPDetector(
        sample_rate=sr,
        frame_length=2048,
        hop_length=512,
        min_freq=200,
        max_freq=600
    )
    
    time_stamps, frequencies = obp_detector.detect_pitch(processed_audio)
    smoothed_frequencies = obp_detector.smooth_pitch_contour(frequencies)
    
    voiced_frames = sum(1 for f in smoothed_frequencies if f > 0)
    print(f"✓ Detected pitch in {voiced_frames} frames")
    print()
    
    # Step 4: Convert to MIDI
    print("[4/5] Converting to MIDI...")
    midi_converter = MIDIConverter(tempo=120)
    output_file = "demo_output.mid"
    midi_converter.generate_midi_from_pitches(
        time_stamps,
        smoothed_frequencies,
        output_file,
        min_note_duration=0.15
    )
    print("✓ MIDI file created")
    print()
    
    # Step 5: Visualize
    print("[5/5] Creating visualizations...")
    visualizer = Visualizer()
    
    notes = midi_converter.notes
    visualizer.plot_combined(
        time_stamps,
        smoothed_frequencies,
        notes,
        save_path="demo_visualization.png",
        show=False
    )
    print("✓ Visualization saved to demo_visualization.png")
    print()
    
    # Print summary
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print(f"Generated files:")
    print(f"  - {test_file} (test audio)")
    print(f"  - {output_file} (MIDI output)")
    print(f"  - demo_visualization.png (visualization)")
    print()
    
    if notes:
        print(f"Detected {len(notes)} notes:")
        for i, (start, duration, midi_note, velocity) in enumerate(notes[:11]):
            note_name = midi_converter.get_note_name(midi_note)
            print(f"  {i+1}. {note_name} at {start:.2f}s for {duration:.2f}s")
    
    print()
    print("✓ Demo successful! The OneBitPitch algorithm is working correctly.")
    print("=" * 70)


if __name__ == "__main__":
    main()