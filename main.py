"""
Peaches - Voice-to-MIDI Converter
Main entry point for the application.

Uses the OneBitPitch (OBP) algorithm for efficient pitch detection.
"""

import argparse
import time
import os
from audio_processor import AudioProcessor
from noise_reducer import NoiseReducer
from obp_detector import OBPDetector
from midi_converter import MIDIConverter
from visualizer import Visualizer


def main():
    """Main function to run the voice-to-MIDI conversion pipeline."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Peaches - Convert voice recordings to MIDI using OneBitPitch algorithm'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Input audio file (WAV or MP3)'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Output MIDI file path (default: input_name.mid)'
    )
    parser.add_argument(
        '--visualize',
        '-v',
        action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--save-plots',
        '-s',
        action='store_true',
        help='Save visualization plots to files'
    )
    parser.add_argument(
        '--min-freq',
        type=float,
        default=80.0,
        help='Minimum detectable frequency in Hz (default: 80)'
    )
    parser.add_argument(
        '--max-freq',
        type=float,
        default=800.0,
        help='Maximum detectable frequency in Hz (default: 800)'
    )
    parser.add_argument(
        '--min-note-duration',
        type=float,
        default=0.1,
        help='Minimum note duration in seconds (default: 0.1)'
    )
    parser.add_argument(
        '--tempo',
        type=int,
        default=120,
        help='MIDI tempo in BPM (default: 120)'
    )
    parser.add_argument(
        '--no-smooth',
        action='store_true',
        help='Disable note smoothing'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return
    
    # Determine output file path
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output = f"{base_name}.mid"
    
    print("=" * 70)
    print("PEACHES - Voice-to-MIDI Converter")
    print("Using OneBitPitch (OBP) Algorithm")
    print("=" * 70)
    print(f"\nInput file: {args.input_file}")
    print(f"Output file: {args.output}")
    print()
    
    # Start timing
    start_time = time.time()
    
    # Step 1: Load and preprocess audio
    print("[1/5] Loading and preprocessing audio...")
    audio_processor = AudioProcessor(target_sr=44100)
    audio_data, sample_rate = audio_processor.preprocess(args.input_file)
    print(f"✓ Audio loaded: {audio_processor.get_duration():.2f} seconds")
    print()
    
    # Step 2: Apply noise reduction
    print("[2/5] Applying noise reduction...")
    noise_reducer = NoiseReducer(sample_rate=sample_rate)
    processed_audio = noise_reducer.process(
        audio_data,
        apply_bandpass=True,
        apply_normalization=True,
        low_freq=args.min_freq,
        high_freq=args.max_freq
    )
    print("✓ Noise reduction complete")
    print()
    
    # Step 3: Detect pitch using OBP algorithm
    print("[3/5] Detecting pitch using OneBitPitch algorithm...")
    obp_detector = OBPDetector(
        sample_rate=sample_rate,
        frame_length=2048,
        hop_length=512,
        min_freq=args.min_freq,
        max_freq=args.max_freq
    )
    
    time_stamps, frequencies = obp_detector.detect_pitch(processed_audio)
    
    # Smooth pitch contour (unless disabled)
    if not args.no_smooth:
        smoothed_frequencies = obp_detector.smooth_pitch_contour(frequencies, window_size=5)
    else:
        smoothed_frequencies = frequencies
    
    # Count voiced frames
    voiced_frames = sum(1 for f in smoothed_frequencies if f > 0)
    print(f"✓ Pitch detection complete")
    print(f"  - Total frames: {len(smoothed_frequencies)}")
    print(f"  - Voiced frames: {voiced_frames} ({100*voiced_frames/len(smoothed_frequencies):.1f}%)")
    print()
    
    # Step 4: Convert to MIDI
    print("[4/5] Converting to MIDI...")
    midi_converter = MIDIConverter(tempo=args.tempo)
    midi_converter.generate_midi_from_pitches(
        time_stamps,
        smoothed_frequencies,
        args.output,
        min_note_duration=args.min_note_duration,
        smooth=not args.no_smooth
    )
    print("✓ MIDI conversion complete")
    print()
    
    # Step 5: Visualize (optional)
    if args.visualize or args.save_plots:
        print("[5/5] Generating visualizations...")
        visualizer = Visualizer(figsize=(12, 6))
        
        # Get notes for visualization
        notes = midi_converter.notes
        
        if args.save_plots:
            # Save plots
            base_name = os.path.splitext(args.output)[0]
            
            visualizer.plot_pitch_contour(
                time_stamps,
                smoothed_frequencies,
                save_path=f"{base_name}_pitch.png",
                show=False
            )
            
            visualizer.plot_midi_notes(
                notes,
                save_path=f"{base_name}_notes.png",
                show=False
            )
            
            visualizer.plot_combined(
                time_stamps,
                smoothed_frequencies,
                notes,
                save_path=f"{base_name}_combined.png",
                show=False
            )
            
            visualizer.plot_statistics(
                smoothed_frequencies,
                notes,
                save_path=f"{base_name}_stats.png",
                show=False
            )
            
            print("✓ Visualizations saved")
        else:
            # Show plots interactively
            visualizer.plot_combined(
                time_stamps,
                smoothed_frequencies,
                notes,
                show=True
            )
            print("✓ Visualizations displayed")
        print()
    
    # Print summary
    elapsed_time = time.time() - start_time
    
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    print(f"Output saved to: {args.output}")
    
    if midi_converter.notes:
        print(f"\nMIDI Summary:")
        print(f"  - Total notes: {len(midi_converter.notes)}")
        print(f"  - Duration: {midi_converter.notes[-1][0] + midi_converter.notes[-1][1]:.2f} seconds")
        midi_notes = [n[2] for n in midi_converter.notes]
        print(f"  - Note range: {midi_converter.get_note_name(min(midi_notes))} - "
              f"{midi_converter.get_note_name(max(midi_notes))}")
    
    print("\n✓ Done! Your MIDI file is ready.")
    print("=" * 70)


if __name__ == "__main__":
    main()
