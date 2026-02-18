"""
Main Recorder Script for Biosignal MVP
This script demonstrates basic LSL (Lab Streaming Layer) setup for biosignal acquisition.

Usage:
    python main_recorder.py
"""

try:
    import pylsl
    import numpy as np
    from datetime import datetime
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    exit(1)


def create_lsl_outlet(name="BiosignalStream", type_="EEG", channel_count=1, 
                      sampling_rate=250, channel_format='float32', source_id='biosignal_mvp'):
    """
    Create an LSL outlet for streaming biosignal data.
    
    Args:
        name: Name of the stream
        type_: Type of the stream (e.g., 'EEG', 'ECG', 'PPG')
        channel_count: Number of channels
        sampling_rate: Nominal sampling rate in Hz
        channel_format: Data format ('float32', 'double64', etc.)
        source_id: Unique identifier for the stream source
    
    Returns:
        StreamOutlet: LSL outlet object
    """
    print(f"Creating LSL outlet: {name}")
    info = pylsl.StreamInfo(name, type_, channel_count, sampling_rate, 
                           channel_format, source_id)
    
    # Optional: Add metadata
    channels = info.desc().append_child("channels")
    for i in range(channel_count):
        ch = channels.append_child("channel")
        ch.append_child_value("label", f"Channel_{i+1}")
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", type_)
    
    outlet = pylsl.StreamOutlet(info)
    print(f"LSL stream '{name}' is now active")
    return outlet


def find_lsl_streams(stream_type=None, timeout=5.0):
    """
    Search for available LSL streams on the network.
    
    Args:
        stream_type: Optional filter by stream type (e.g., 'EEG', 'ECG')
        timeout: Search timeout in seconds
    
    Returns:
        list: List of available StreamInfo objects
    """
    print(f"Searching for LSL streams (timeout: {timeout}s)...")
    streams = pylsl.resolve_streams(timeout)
    
    if stream_type:
        streams = [s for s in streams if s.type() == stream_type]
    
    print(f"Found {len(streams)} stream(s)")
    for i, stream in enumerate(streams):
        print(f"  [{i}] {stream.name()} ({stream.type()}) - "
              f"{stream.channel_count()} channels @ {stream.nominal_srate()} Hz")
    
    return streams


def record_from_inlet(inlet, duration=10, verbose=True):
    """
    Record data from an LSL inlet for a specified duration.
    
    Args:
        inlet: LSL StreamInlet object
        duration: Recording duration in seconds
        verbose: Print recording status
    
    Returns:
        tuple: (samples, timestamps) as numpy arrays
    """
    print(f"Recording for {duration} seconds...")
    start_time = pylsl.local_clock()
    samples = []
    timestamps = []
    
    while (pylsl.local_clock() - start_time) < duration:
        sample, timestamp = inlet.pull_sample(timeout=1.0)
        if sample:
            samples.append(sample)
            timestamps.append(timestamp)
            if verbose and len(samples) % 100 == 0:
                print(f"  Recorded {len(samples)} samples...")
    
    print(f"Recording complete: {len(samples)} samples")
    return np.array(samples), np.array(timestamps)


def main():
    """
    Main entry point for the biosignal recorder.
    This is a basic example that searches for LSL streams.
    """
    print("=" * 60)
    print("Biosignal MVP - LSL Recorder")
    print("=" * 60)
    print()
    
    # Search for available streams
    streams = find_lsl_streams(timeout=5.0)
    
    if not streams:
        print("\nNo LSL streams found.")
        print("\nExample: Creating a demo outlet...")
        print("(In a real scenario, this would be done by your data source)")
        outlet = create_lsl_outlet(
            name="Demo_HeartRate",
            type_="ECG",
            channel_count=1,
            sampling_rate=250
        )
        print("\nDemo outlet created. You can now connect to it from another script.")
        print("Press Ctrl+C to stop...")
        
        try:
            # Keep the outlet alive
            import time
            while True:
                # Simulate sending some data
                sample = [np.random.randn()]  # Random data for demo
                outlet.push_sample(sample)
                time.sleep(1/250)  # Match sampling rate
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print("\nTo record from a stream, you can create an inlet:")
        print("  inlet = pylsl.StreamInlet(streams[0])")
        print("  samples, timestamps = record_from_inlet(inlet, duration=10)")


if __name__ == "__main__":
    main()
