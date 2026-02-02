"""
Visualizer Module
Creates visualizations of pitch contours and MIDI notes.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class Visualizer:
    """Creates visualizations for pitch detection and MIDI conversion."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Initialize the Visualizer.
        
        Args:
            figsize: Figure size as (width, height) in inches
        """
        self.figsize = figsize
    
    def plot_pitch_contour(
        self,
        time_stamps: List[float],
        frequencies: List[float],
        title: str = "Detected Pitch Contour",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot the detected pitch contour over time.
        
        Args:
            time_stamps: List of time stamps in seconds
            frequencies: List of detected frequencies in Hz
            title: Plot title
            save_path: Path to save the plot (optional)
            show: Display the plot
        """
        plt.figure(figsize=self.figsize)
        
        # Filter out zero frequencies for plotting
        times = []
        freqs = []
        for t, f in zip(time_stamps, frequencies):
            if f > 0:
                times.append(t)
                freqs.append(f)
        
        if not times:
            print("No pitch data to plot")
            return
        
        plt.plot(times, freqs, 'b-', linewidth=1.5, alpha=0.7)
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Frequency (Hz)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Pitch contour saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_midi_notes(
        self,
        notes: List[Tuple[float, float, int, int]],
        title: str = "MIDI Notes",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot MIDI notes as a piano roll.
        
        Args:
            notes: List of (start_time, duration, midi_note, velocity) tuples
            title: Plot title
            save_path: Path to save the plot (optional)
            show: Display the plot
        """
        if not notes:
            print("No notes to plot")
            return
        
        plt.figure(figsize=self.figsize)
        
        # Plot each note as a horizontal bar
        for start, duration, midi_note, velocity in notes:
            alpha = velocity / 127.0  # Use velocity for transparency
            plt.barh(
                midi_note,
                duration,
                left=start,
                height=0.8,
                color='blue',
                alpha=alpha,
                edgecolor='black',
                linewidth=0.5
            )
        
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('MIDI Note Number', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"MIDI notes visualization saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_combined(
        self,
        time_stamps: List[float],
        frequencies: List[float],
        notes: List[Tuple[float, float, int, int]],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot pitch contour and MIDI notes together.
        
        Args:
            time_stamps: List of time stamps in seconds
            frequencies: List of detected frequencies in Hz
            notes: List of (start_time, duration, midi_note, velocity) tuples
            save_path: Path to save the plot (optional)
            show: Display the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Pitch contour
        times = []
        freqs = []
        for t, f in zip(time_stamps, frequencies):
            if f > 0:
                times.append(t)
                freqs.append(f)
        
        if times:
            ax1.plot(times, freqs, 'b-', linewidth=1.5, alpha=0.7)
            ax1.set_xlabel('Time (seconds)', fontsize=11)
            ax1.set_ylabel('Frequency (Hz)', fontsize=11)
            ax1.set_title('Detected Pitch Contour', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: MIDI notes (piano roll)
        if notes:
            for start, duration, midi_note, velocity in notes:
                alpha = velocity / 127.0
                ax2.barh(
                    midi_note,
                    duration,
                    left=start,
                    height=0.8,
                    color='green',
                    alpha=alpha,
                    edgecolor='black',
                    linewidth=0.5
                )
            
            ax2.set_xlabel('Time (seconds)', fontsize=11)
            ax2.set_ylabel('MIDI Note Number', fontsize=11)
            ax2.set_title('MIDI Notes (Piano Roll)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Combined visualization saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_statistics(
        self,
        frequencies: List[float],
        notes: List[Tuple[float, float, int, int]],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot statistical analysis of pitch and notes.
        
        Args:
            frequencies: List of detected frequencies in Hz
            notes: List of (start_time, duration, midi_note, velocity) tuples
            save_path: Path to save the plot (optional)
            show: Display the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Filter valid frequencies
        valid_freqs = [f for f in frequencies if f > 0]
        
        if valid_freqs:
            # Plot 1: Frequency histogram
            ax1.hist(valid_freqs, bins=50, color='blue', alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Frequency (Hz)', fontsize=10)
            ax1.set_ylabel('Count', fontsize=10)
            ax1.set_title('Frequency Distribution', fontsize=11, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        if notes:
            # Plot 2: Note duration histogram
            durations = [d for _, d, _, _ in notes]
            ax2.hist(durations, bins=30, color='green', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Duration (seconds)', fontsize=10)
            ax2.set_ylabel('Count', fontsize=10)
            ax2.set_title('Note Duration Distribution', fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: MIDI note distribution
            midi_notes = [m for _, _, m, _ in notes]
            ax3.hist(midi_notes, bins=range(min(midi_notes), max(midi_notes) + 2), 
                    color='purple', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('MIDI Note Number', fontsize=10)
            ax3.set_ylabel('Count', fontsize=10)
            ax3.set_title('MIDI Note Distribution', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Velocity distribution
            velocities = [v for _, _, _, v in notes]
            ax4.hist(velocities, bins=20, color='orange', alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Velocity', fontsize=10)
            ax4.set_ylabel('Count', fontsize=10)
            ax4.set_title('Velocity Distribution', fontsize=11, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Statistics visualization saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
