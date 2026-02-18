"""
Demo Script - Shows how to use the Biosignal MVP components

This script demonstrates the basic workflow without requiring hardware.
It uses the simulation module to generate synthetic signals and processes them.
"""

import sys
import os

# Ensure we can import from parent directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("Biosignal MVP - Demo Script")
print("=" * 70)
print()

# Check if dependencies are installed
try:
    import numpy as np
    import scipy
    print("✓ Dependencies are installed")
    deps_installed = True
except ImportError as e:
    print("✗ Dependencies not installed")
    print(f"  Error: {e}")
    print()
    print("To install dependencies, run:")
    print("  pip install -r requirements.txt")
    print()
    deps_installed = False

if deps_installed:
    print()
    print("-" * 70)
    print("Demo 1: Generating Synthetic Biosignals")
    print("-" * 70)
    
    from simulation.signal_generator import (
        generate_ecg_signal, 
        generate_ppg_signal,
        generate_eeg_signal
    )
    
    # Generate 5 seconds of different signals
    duration = 5.0
    
    # ECG signal
    ecg, t_ecg = generate_ecg_signal(duration=duration, fs=250, heart_rate=72)
    print(f"✓ Generated ECG signal: {len(ecg)} samples at 250 Hz")
    print(f"  Duration: {duration}s, Heart rate: 72 BPM")
    
    # PPG signal
    ppg, t_ppg = generate_ppg_signal(duration=duration, fs=100, heart_rate=72)
    print(f"✓ Generated PPG signal: {len(ppg)} samples at 100 Hz")
    
    # EEG signal
    eeg, t_eeg = generate_eeg_signal(duration=duration, fs=256)
    print(f"✓ Generated EEG signal: {len(eeg)} samples at 256 Hz")
    
    print()
    print("-" * 70)
    print("Demo 2: Processing Signals with DSP")
    print("-" * 70)
    
    from dsp.filters import (
        bandpass_filter,
        detect_peaks,
        calculate_heart_rate,
        baseline_correction
    )
    
    # Process ECG signal
    print("Processing ECG signal...")
    
    # Apply bandpass filter (typical ECG range: 0.5-40 Hz)
    ecg_filtered = bandpass_filter(ecg, lowcut=0.5, highcut=40, fs=250)
    print(f"✓ Applied bandpass filter (0.5-40 Hz)")
    
    # Baseline correction
    ecg_corrected = baseline_correction(ecg_filtered, method='median')
    print(f"✓ Baseline correction applied")
    
    # Detect R-peaks
    peaks = detect_peaks(ecg_corrected, distance=int(250*0.5), prominence=0.5)
    print(f"✓ Detected {len(peaks)} R-peaks")
    
    # Calculate heart rate
    if len(peaks) > 1:
        hr = calculate_heart_rate(peaks, fs=250)
        print(f"✓ Calculated heart rate: {hr:.1f} BPM")
    else:
        print("  (Not enough peaks to calculate heart rate)")
    
    print()
    print("-" * 70)
    print("Demo 3: LSL Stream Information")
    print("-" * 70)
    
    try:
        import pylsl
        print("✓ LSL (Lab Streaming Layer) is installed")
        print()
        print("To use LSL streams:")
        print("  1. Run main_recorder.py to create a demo stream")
        print("  2. Use acquisition/lsl_stream.py to connect and record")
        print()
        print("Example:")
        print("  from acquisition.lsl_stream import BiosignalAcquisition")
        print("  acq = BiosignalAcquisition(stream_type='ECG')")
        print("  acq.connect()")
        print("  samples, timestamps = acq.acquire_duration(10)")
        
    except ImportError:
        print("✗ LSL not installed (optional)")
        print("  To install: pip install pylsl")
    
    print()
    print("=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Install all dependencies: pip install -r requirements.txt")
    print("  2. Run main_recorder.py to explore LSL functionality")
    print("  3. Use PyCharm or Google Antigravity for development")
    print()

else:
    print("Please install dependencies first:")
    print("  pip install -r requirements.txt")
    print()
