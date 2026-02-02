# Peaches 🍑 - Voice-to-MIDI Converter

An offline Python tool that converts vocal recordings (humming or singing) into MIDI sequences using the **OneBitPitch (OBP) algorithm** for efficient pitch detection.

## Features

- **Efficient Pitch Detection**: Uses OneBitPitch algorithm with 1-bit quantization and autocorrelation
- **Audio Preprocessing**: Automatic noise reduction and bandpass filtering
- **MIDI Export**: Converts detected pitches to clean MIDI note sequences
- **Visualization**: Generate plots of pitch contours and MIDI notes
- **Customizable**: Adjustable frequency ranges, tempo, and note duration

## Installation

All required dependencies are already installed:
- numpy
- scipy
- librosa
- mido
- pretty_midi
- matplotlib
- soundfile

## Usage

### Basic Usage

Convert a vocal recording to MIDI:

```bash
python main.py input_audio.wav
```

This will create `input_audio.mid` in the same directory.

### Advanced Options

```bash
python main.py input_audio.wav \
    --output custom_output.mid \
    --visualize \
    --save-plots \
    --min-freq 80 \
    --max-freq 800 \
    --min-note-duration 0.1 \
    --tempo 120
```

### Command-Line Arguments

- `input_file`: Path to input audio file (WAV or MP3)
- `--output`, `-o`: Output MIDI file path (default: input_name.mid)
- `--visualize`, `-v`: Display visualization plots
- `--save-plots`, `-s`: Save visualization plots to PNG files
- `--min-freq`: Minimum detectable frequency in Hz (default: 80)
- `--max-freq`: Maximum detectable frequency in Hz (default: 800)
- `--min-note-duration`: Minimum note duration in seconds (default: 0.1)
- `--tempo`: MIDI tempo in BPM (default: 120)
- `--no-smooth`: Disable note smoothing

## How It Works

### OneBitPitch Algorithm

The OneBitPitch (OBP) algorithm is a lightweight pitch detection method that differs from traditional algorithms like YIN or SWIPE:

1. **1-bit Quantization**: Converts the audio waveform into a binary signal (+1/-1) based on a threshold
2. **Autocorrelation**: Applies autocorrelation to the binary signal to find periodicity
3. **Peak Detection**: Identifies the fundamental frequency from autocorrelation peaks

This approach prioritizes computational efficiency while maintaining acceptable pitch detection accuracy.

### Processing Pipeline

1. **Audio Loading**: Load and normalize WAV/MP3 files
2. **Noise Reduction**: Apply bandpass filtering and normalization
3. **Pitch Detection**: Use OBP algorithm to detect frequencies over time
4. **Note Segmentation**: Convert continuous pitch to discrete MIDI notes
5. **MIDI Export**: Save as standard MIDI file (.mid)

## Architecture

The project follows a modular, object-oriented design:

- `audio_processor.py`: Audio file loading and preprocessing
- `noise_reducer.py`: Noise filtering and normalization
- `obp_detector.py`: OneBitPitch algorithm implementation
- `midi_converter.py`: Pitch-to-MIDI conversion and export
- `visualizer.py`: Pitch and MIDI visualization
- `main.py`: CLI entry point

## Files

- `test_melody.wav`: A synthetic melody generated for testing (C-D-E-F-G-A-G-F-E-D-C scale)
- `demo_output.mid`: MIDI file generated from the test melody
- `demo_visualization.png`: Visualization showing pitch contour and detected MIDI notes

## Try It Yourself

To generate your own examples:

```bash
# Generate a new demo
python demo.py

# Convert your own audio file
python main.py your_recording.wav --visualize --save-plots
```

The demo will create:
- A test audio file with a simple melody
- A MIDI conversion of that melody
- Visualization plots showing the pitch detection and note extraction

## Example

```bash
# Convert a hummed melody to MIDI with visualization
python main.py my_melody.wav --visualize --save-plots

# Process with custom frequency range for lower voices
python main.py bass_voice.wav --min-freq 60 --max-freq 400

# Create MIDI with different tempo
python main.py recording.wav --tempo 140
```

## Output

The tool generates:
- A `.mid` MIDI file containing the detected notes
- (Optional) Visualization plots:
  - `*_pitch.png`: Detected pitch contour
  - `*_notes.png`: MIDI notes as piano roll
  - `*_combined.png`: Combined pitch and MIDI view
  - `*_stats.png`: Statistical analysis

## Future Enhancements

- Web-based interface (Flask/FastAPI)
- GPU acceleration (numba/cupy)
- Real-time processing mode
- Batch processing
- Advanced noise reduction

## License

This project is created for educational and personal use.
