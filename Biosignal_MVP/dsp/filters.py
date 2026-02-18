"""
Signal Processing Module

This module provides basic DSP functions for biosignal processing.
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a Butterworth bandpass filter.
    
    Args:
        lowcut: Low cutoff frequency in Hz
        highcut: High cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Filter coefficients (b, a)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Apply a bandpass filter to the signal.
    
    Args:
        data: Input signal
        lowcut: Low cutoff frequency in Hz
        highcut: High cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Filtered signal
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def notch_filter(data: np.ndarray, freq: float, fs: float, Q: float = 30.0) -> np.ndarray:
    """
    Apply a notch filter to remove a specific frequency (e.g., powerline interference).
    
    Args:
        data: Input signal
        freq: Frequency to remove in Hz (e.g., 50 or 60 Hz)
        fs: Sampling frequency in Hz
        Q: Quality factor
        
    Returns:
        Filtered signal
    """
    b, a = signal.iirnotch(freq, Q, fs)
    y = signal.filtfilt(b, a, data)
    return y


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply a moving average filter.
    
    Args:
        data: Input signal
        window_size: Size of the moving window
        
    Returns:
        Smoothed signal
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')


def detect_peaks(data: np.ndarray, height: Optional[float] = None, 
                 distance: Optional[int] = None, prominence: Optional[float] = None) -> np.ndarray:
    """
    Detect peaks in the signal (e.g., R-peaks in ECG).
    
    Args:
        data: Input signal
        height: Minimum peak height
        distance: Minimum distance between peaks in samples
        prominence: Minimum prominence of peaks
        
    Returns:
        Array of peak indices
    """
    peaks, _ = signal.find_peaks(data, height=height, distance=distance, prominence=prominence)
    return peaks


def calculate_heart_rate(peaks: np.ndarray, fs: float) -> float:
    """
    Calculate heart rate from R-peaks.
    
    Args:
        peaks: Array of peak indices
        fs: Sampling frequency in Hz
        
    Returns:
        Heart rate in beats per minute (BPM)
    """
    if len(peaks) < 2:
        return 0.0
    
    # Calculate RR intervals in samples
    rr_intervals = np.diff(peaks)
    
    # Convert to time (seconds)
    rr_intervals_sec = rr_intervals / fs
    
    # Calculate heart rate (60 / mean RR interval)
    mean_rr = np.mean(rr_intervals_sec)
    heart_rate = 60.0 / mean_rr if mean_rr > 0 else 0.0
    
    return heart_rate


def baseline_correction(data: np.ndarray, method: str = 'median') -> np.ndarray:
    """
    Remove baseline drift from the signal.
    
    Args:
        data: Input signal
        method: Method to use ('mean', 'median', 'detrend')
        
    Returns:
        Baseline-corrected signal
    """
    if method == 'mean':
        return data - np.mean(data)
    elif method == 'median':
        return data - np.median(data)
    elif method == 'detrend':
        return signal.detrend(data)
    else:
        raise ValueError(f"Unknown method: {method}")


# Example usage
if __name__ == "__main__":
    # Generate a synthetic ECG-like signal
    fs = 250  # Sampling frequency
    t = np.arange(0, 10, 1/fs)  # 10 seconds
    
    # Simulated ECG with noise
    ecg_signal = np.sin(2 * np.pi * 1.2 * t)  # ~72 BPM
    noise = 0.1 * np.random.randn(len(t))
    noisy_signal = ecg_signal + noise
    
    # Apply bandpass filter (typical ECG range: 0.5-40 Hz)
    filtered = bandpass_filter(noisy_signal, 0.5, 40, fs)
    
    # Detect peaks
    peaks = detect_peaks(filtered, distance=int(fs*0.5))  # Min 0.5s between peaks
    
    # Calculate heart rate
    hr = calculate_heart_rate(peaks, fs)
    print(f"Detected heart rate: {hr:.1f} BPM")
