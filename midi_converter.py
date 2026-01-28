"""
MIDI Converter Module
Converts pitch frequencies to MIDI notes and exports MIDI files.
"""

import numpy as np
import mido
from mido import Message, MidiFile, MidiTrack
import pretty_midi
from typing import List, Tuple, Optional


class MIDIConverter:
    """Converts detected pitches to MIDI notes and exports MIDI files."""
    
    def __init__(self, tempo: int = 120):
        """
        Initialize the MIDIConverter.
        
        Args:
            tempo: Tempo in BPM (default: 120)
        """
        self.tempo = tempo
        self.notes = []
    
    def frequency_to_midi(self, frequency: float) -> Optional[int]:
        """
        Convert frequency in Hz to MIDI note number.
        
        Uses the formula: MIDI = 12 * log2(f/440) + 69
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            MIDI note number (0-127), or None if invalid
        """
        if frequency <= 0:
            return None
        
        # Calculate MIDI note number
        midi_note = 12 * np.log2(frequency / 440.0) + 69
        
        # Round to nearest integer
        midi_note = int(round(midi_note))
        
        # Ensure within valid MIDI range
        if 0 <= midi_note <= 127:
            return midi_note
        
        return None
    
    def midi_to_frequency(self, midi_note: int) -> float:
        """
        Convert MIDI note number to frequency in Hz.
        
        Args:
            midi_note: MIDI note number (0-127)
            
        Returns:
            Frequency in Hz
        """
        return 440.0 * (2 ** ((midi_note - 69) / 12.0))
    
    def segment_notes(
        self,
        time_stamps: List[float],
        frequencies: List[float],
        min_note_duration: float = 0.1
    ) -> List[Tuple[float, float, int, int]]:
        """
        Segment continuous pitch into discrete notes.
        
        Args:
            time_stamps: List of time stamps in seconds
            frequencies: List of frequencies in Hz
            min_note_duration: Minimum note duration in seconds
            
        Returns:
            List of (start_time, duration, midi_note, velocity) tuples
        """
        notes = []
        
        if not frequencies:
            return notes
        
        current_note = None
        note_start = None
        note_count = 0
        
        for i, (time, freq) in enumerate(zip(time_stamps, frequencies)):
            midi_note = self.frequency_to_midi(freq)
            
            if midi_note is not None:
                if current_note is None:
                    # Start new note
                    current_note = midi_note
                    note_start = time
                    note_count = 1
                elif midi_note == current_note:
                    # Continue current note
                    note_count += 1
                else:
                    # Different note - save previous and start new
                    if note_start is not None:
                        duration = time - note_start
                        if duration >= min_note_duration:
                            velocity = min(100, 60 + note_count)  # Dynamic velocity
                            notes.append((note_start, duration, current_note, velocity))
                    
                    current_note = midi_note
                    note_start = time
                    note_count = 1
            else:
                # No pitch detected - save current note if exists
                if current_note is not None and note_start is not None:
                    duration = time - note_start
                    if duration >= min_note_duration:
                        velocity = min(100, 60 + note_count)
                        notes.append((note_start, duration, current_note, velocity))
                
                current_note = None
                note_start = None
                note_count = 0
        
        # Handle final note
        if current_note is not None and note_start is not None:
            duration = time_stamps[-1] - note_start + (time_stamps[1] - time_stamps[0])
            if duration >= min_note_duration:
                velocity = min(100, 60 + note_count)
                notes.append((note_start, duration, current_note, velocity))
        
        self.notes = notes
        return notes
    
    def smooth_notes(
        self,
        notes: List[Tuple[float, float, int, int]],
        pitch_tolerance: int = 1
    ) -> List[Tuple[float, float, int, int]]:
        """
        Smooth note sequences by merging similar pitches.
        
        Args:
            notes: List of (start_time, duration, midi_note, velocity) tuples
            pitch_tolerance: Semitone tolerance for merging
            
        Returns:
            Smoothed notes
        """
        if not notes:
            return notes
        
        smoothed = []
        current = list(notes[0])
        
        for next_note in notes[1:]:
            start, duration, midi, velocity = next_note
            
            # Check if notes are similar and adjacent
            if abs(current[2] - midi) <= pitch_tolerance:
                # Merge notes
                gap = start - (current[0] + current[1])
                if gap < 0.1:  # Small gap threshold
                    current[1] = start + duration - current[0]
                    current[3] = max(current[3], velocity)  # Keep higher velocity
                else:
                    smoothed.append(tuple(current))
                    current = list(next_note)
            else:
                smoothed.append(tuple(current))
                current = list(next_note)
        
        smoothed.append(tuple(current))
        return smoothed
    
    def generate_midi_from_pitches(
        self,
        time_stamps: List[float],
        frequencies: List[float],
        output_path: str,
        min_note_duration: float = 0.1,
        smooth: bool = True
    ) -> None:
        """
        Generate MIDI file from detected pitches.
        
        Args:
            time_stamps: List of time stamps in seconds
            frequencies: List of frequencies in Hz
            output_path: Output MIDI file path
            min_note_duration: Minimum note duration in seconds
            smooth: Apply note smoothing
        """
        # Segment into notes
        notes = self.segment_notes(time_stamps, frequencies, min_note_duration)
        
        if not notes:
            print("Warning: No notes detected")
            return
        
        # Smooth notes if requested
        if smooth:
            notes = self.smooth_notes(notes)
        
        # Create MIDI file using mido
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        
        # Add tempo
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(self.tempo)))
        
        # Convert notes to MIDI messages
        # Sort notes by start time
        notes = sorted(notes, key=lambda x: x[0])
        
        current_time = 0
        ticks_per_beat = mid.ticks_per_beat
        seconds_per_beat = 60.0 / self.tempo
        
        for start_time, duration, midi_note, velocity in notes:
            # Calculate delta time in ticks
            delta_ticks = int((start_time - current_time) * ticks_per_beat / seconds_per_beat)
            delta_ticks = max(0, delta_ticks)
            
            # Note on
            track.append(Message('note_on', note=midi_note, velocity=velocity, time=delta_ticks))
            current_time = start_time
            
            # Note off
            duration_ticks = int(duration * ticks_per_beat / seconds_per_beat)
            duration_ticks = max(1, duration_ticks)
            track.append(Message('note_off', note=midi_note, velocity=0, time=duration_ticks))
            current_time += duration
        
        # Save MIDI file
        mid.save(output_path)
        print(f"MIDI file saved: {output_path}")
        print(f"  - Total notes: {len(notes)}")
        print(f"  - Duration: {notes[-1][0] + notes[-1][1]:.2f} seconds")
        print(f"  - Note range: {min(n[2] for n in notes)} - {max(n[2] for n in notes)}")
    
    def get_note_name(self, midi_note: int) -> str:
        """
        Get the musical name of a MIDI note.
        
        Args:
            midi_note: MIDI note number
            
        Returns:
            Note name (e.g., "C4", "A#5")
        """
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_note // 12) - 1
        note = note_names[midi_note % 12]
        return f"{note}{octave}"
    
    def print_notes(self, notes: Optional[List[Tuple[float, float, int, int]]] = None) -> None:
        """
        Print detected notes in a readable format.
        
        Args:
            notes: List of notes to print (uses self.notes if None)
        """
        if notes is None:
            notes = self.notes
        
        if not notes:
            print("No notes to display")
            return
        
        print("\nDetected Notes:")
        print("-" * 60)
        print(f"{'Time (s)':<10} {'Duration (s)':<15} {'Note':<10} {'MIDI#':<10} {'Velocity':<10}")
        print("-" * 60)
        
        for start, duration, midi, velocity in notes[:20]:  # Show first 20
            note_name = self.get_note_name(midi)
            print(f"{start:<10.2f} {duration:<15.2f} {note_name:<10} {midi:<10} {velocity:<10}")
        
        if len(notes) > 20:
            print(f"... and {len(notes) - 20} more notes")
