"""
Biosignal Simulation Module

This module provides functions to generate synthetic biosignals for testing.
"""

import numpy as np
from typing import Optional, Tuple


def generate_ecg_signal(duration: float, fs: float = 250, heart_rate: float = 72.0, 
                       noise_level: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic ECG signal.
    
    Args:
        duration: Duration in seconds
        fs: Sampling frequency in Hz
        heart_rate: Heart rate in beats per minute (BPM)
        noise_level: Amplitude of Gaussian noise to add
        
    Returns:
        Tuple of (signal, time) arrays
    """
    t = np.arange(0, duration, 1/fs)
    n_samples = len(t)
    
    # Calculate period from heart rate
    period = 60.0 / heart_rate  # seconds per beat
    
    # Generate simple ECG-like waveform using sum of sinusoids
    ecg = np.zeros(n_samples)
    
    for i, time in enumerate(t):
        # Position within cardiac cycle
        phase = (time % period) / period
        
        # Simplified ECG components
        if 0.0 <= phase < 0.1:  # P wave
            ecg[i] = 0.15 * np.sin(2 * np.pi * phase / 0.1)
        elif 0.15 <= phase < 0.25:  # QRS complex
            if phase < 0.18:  # Q
                ecg[i] = -0.2 * np.sin(2 * np.pi * (phase - 0.15) / 0.03)
            elif phase < 0.21:  # R
                ecg[i] = 1.5 * np.sin(2 * np.pi * (phase - 0.18) / 0.03)
            else:  # S
                ecg[i] = -0.3 * np.sin(2 * np.pi * (phase - 0.21) / 0.04)
        elif 0.35 <= phase < 0.55:  # T wave
            ecg[i] = 0.3 * np.sin(2 * np.pi * (phase - 0.35) / 0.2)
    
    # Add noise
    if noise_level > 0:
        noise = noise_level * np.random.randn(n_samples)
        ecg += noise
    
    return ecg, t


def generate_ppg_signal(duration: float, fs: float = 100, heart_rate: float = 72.0,
                       noise_level: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic PPG (Photoplethysmography) signal.
    
    Args:
        duration: Duration in seconds
        fs: Sampling frequency in Hz
        heart_rate: Heart rate in BPM
        noise_level: Amplitude of Gaussian noise to add
        
    Returns:
        Tuple of (signal, time) arrays
    """
    t = np.arange(0, duration, 1/fs)
    
    # Fundamental frequency from heart rate
    f0 = heart_rate / 60.0  # Hz
    
    # PPG signal is roughly a low-pass filtered pulse
    # Using sum of harmonics to create realistic waveform
    ppg = (1.0 * np.sin(2 * np.pi * f0 * t) +
           0.3 * np.sin(2 * np.pi * 2 * f0 * t) +
           0.1 * np.sin(2 * np.pi * 3 * f0 * t))
    
    # Normalize
    ppg = (ppg - ppg.min()) / (ppg.max() - ppg.min())
    
    # Add noise
    if noise_level > 0:
        noise = noise_level * np.random.randn(len(t))
        ppg += noise
    
    return ppg, t


def generate_eeg_signal(duration: float, fs: float = 256, bands: Optional[dict] = None,
                       noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic EEG signal with different frequency bands.
    
    Args:
        duration: Duration in seconds
        fs: Sampling frequency in Hz
        bands: Dictionary of frequency bands and their amplitudes
               e.g., {'delta': (1, 4, 0.5), 'alpha': (8, 12, 1.0)}
               where tuple is (low_freq, high_freq, amplitude)
        noise_level: Amplitude of Gaussian noise to add
        
    Returns:
        Tuple of (signal, time) arrays
    """
    if bands is None:
        # Default EEG bands with typical amplitudes
        bands = {
            'delta': (1, 4, 0.5),    # Deep sleep
            'theta': (4, 8, 0.3),    # Drowsiness
            'alpha': (8, 13, 1.0),   # Relaxed, eyes closed
            'beta': (13, 30, 0.4),   # Alert, active thinking
        }
    
    t = np.arange(0, duration, 1/fs)
    eeg = np.zeros(len(t))
    
    # Add each frequency band
    for band_name, (f_low, f_high, amplitude) in bands.items():
        # Use center frequency of the band
        f_center = (f_low + f_high) / 2
        # Add some frequency variation
        f_variation = (f_high - f_low) / 4
        
        # Generate band with slight frequency modulation
        phase = 2 * np.pi * f_center * t + f_variation * np.sin(2 * np.pi * 0.1 * t)
        eeg += amplitude * np.sin(phase)
    
    # Add noise
    if noise_level > 0:
        noise = noise_level * np.random.randn(len(t))
        eeg += noise
    
    return eeg, t


def add_artifacts(signal: np.ndarray, artifact_type: str = 'motion', 
                 artifact_probability: float = 0.05) -> np.ndarray:
    """
    Add realistic artifacts to a biosignal.
    
    Args:
        signal: Input signal
        artifact_type: Type of artifact ('motion', 'powerline', 'baseline_drift')
        artifact_probability: Probability of artifact occurrence per sample
        
    Returns:
        Signal with artifacts added
    """
    n_samples = len(signal)
    artifact_signal = signal.copy()
    
    if artifact_type == 'motion':
        # Random motion artifacts
        for i in range(n_samples):
            if np.random.rand() < artifact_probability:
                # Add a spike
                artifact_signal[i] += np.random.randn() * 5
    
    elif artifact_type == 'powerline':
        # 50 or 60 Hz powerline interference
        t = np.arange(n_samples)
        powerline = 0.1 * np.sin(2 * np.pi * 60 * t / n_samples)
        artifact_signal += powerline
    
    elif artifact_type == 'baseline_drift':
        # Slow baseline drift
        t = np.arange(n_samples)
        drift = 0.5 * np.sin(2 * np.pi * 0.1 * t / n_samples)
        artifact_signal += drift
    
    return artifact_signal


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Generate 10 seconds of ECG
    ecg, t_ecg = generate_ecg_signal(duration=10, fs=250, heart_rate=72)
    
    # Generate 10 seconds of PPG
    ppg, t_ppg = generate_ppg_signal(duration=10, fs=100, heart_rate=72)
    
    # Generate 10 seconds of EEG
    eeg, t_eeg = generate_eeg_signal(duration=10, fs=256)
    
    print(f"Generated ECG: {len(ecg)} samples at 250 Hz")
    print(f"Generated PPG: {len(ppg)} samples at 100 Hz")
    print(f"Generated EEG: {len(eeg)} samples at 256 Hz")
    
    # Optional: Plot if matplotlib is available
    try:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        
        axes[0].plot(t_ecg[:1000], ecg[:1000])
        axes[0].set_title('Synthetic ECG Signal')
        axes[0].set_ylabel('Amplitude')
        
        axes[1].plot(t_ppg[:500], ppg[:500])
        axes[1].set_title('Synthetic PPG Signal')
        axes[1].set_ylabel('Amplitude')
        
        axes[2].plot(t_eeg[:1000], eeg[:1000])
        axes[2].set_title('Synthetic EEG Signal')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Amplitude')
        
        plt.tight_layout()
        plt.savefig('/tmp/biosignal_simulation.png')
        print("Plot saved to /tmp/biosignal_simulation.png")
    except ImportError:
        print("Matplotlib not available for plotting")
