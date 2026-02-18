"""
LSL Stream Example - Acquisition Module

This module demonstrates how to acquire biosignal data using LSL streams.
"""

import pylsl
import numpy as np
from typing import Optional, List, Tuple


class BiosignalAcquisition:
    """
    Class for acquiring biosignal data from LSL streams.
    """
    
    def __init__(self, stream_name: Optional[str] = None, stream_type: Optional[str] = None):
        """
        Initialize the acquisition system.
        
        Args:
            stream_name: Specific stream name to connect to (optional)
            stream_type: Stream type to filter (e.g., 'EEG', 'ECG', 'PPG')
        """
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.inlet: Optional[pylsl.StreamInlet] = None
        self.stream_info: Optional[pylsl.StreamInfo] = None
        
    def find_streams(self, timeout: float = 5.0) -> List[pylsl.StreamInfo]:
        """
        Find available LSL streams.
        
        Args:
            timeout: Search timeout in seconds
            
        Returns:
            List of available StreamInfo objects
        """
        print(f"Searching for LSL streams (timeout: {timeout}s)...")
        streams = pylsl.resolve_streams(timeout)
        
        # Filter by type if specified
        if self.stream_type:
            streams = [s for s in streams if s.type() == self.stream_type]
        
        # Filter by name if specified
        if self.stream_name:
            streams = [s for s in streams if s.name() == self.stream_name]
        
        return streams
    
    def connect(self, stream_info: Optional[pylsl.StreamInfo] = None) -> bool:
        """
        Connect to an LSL stream.
        
        Args:
            stream_info: StreamInfo to connect to. If None, auto-discover.
            
        Returns:
            True if connection successful
        """
        if stream_info is None:
            streams = self.find_streams()
            if not streams:
                print("No streams found")
                return False
            stream_info = streams[0]
        
        self.stream_info = stream_info
        self.inlet = pylsl.StreamInlet(stream_info)
        print(f"Connected to stream: {stream_info.name()} ({stream_info.type()})")
        return True
    
    def acquire_samples(self, n_samples: int, timeout: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Acquire a specific number of samples.
        
        Args:
            n_samples: Number of samples to acquire
            timeout: Timeout for each sample in seconds
            
        Returns:
            Tuple of (samples, timestamps) as numpy arrays
        """
        if self.inlet is None:
            raise RuntimeError("Not connected to any stream. Call connect() first.")
        
        samples = []
        timestamps = []
        
        for _ in range(n_samples):
            sample, timestamp = self.inlet.pull_sample(timeout=timeout)
            if sample:
                samples.append(sample)
                timestamps.append(timestamp)
        
        return np.array(samples), np.array(timestamps)
    
    def acquire_duration(self, duration: float, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Acquire data for a specific duration.
        
        Args:
            duration: Duration in seconds
            verbose: Print progress
            
        Returns:
            Tuple of (samples, timestamps) as numpy arrays
        """
        if self.inlet is None:
            raise RuntimeError("Not connected to any stream. Call connect() first.")
        
        start_time = pylsl.local_clock()
        samples = []
        timestamps = []
        
        while (pylsl.local_clock() - start_time) < duration:
            sample, timestamp = self.inlet.pull_sample(timeout=1.0)
            if sample:
                samples.append(sample)
                timestamps.append(timestamp)
                if verbose and len(samples) % 100 == 0:
                    print(f"Acquired {len(samples)} samples...")
        
        return np.array(samples), np.array(timestamps)
    
    def disconnect(self):
        """Disconnect from the stream."""
        if self.inlet:
            self.inlet.close_stream()
            self.inlet = None
            self.stream_info = None
            print("Disconnected from stream")


# Example usage
if __name__ == "__main__":
    # Create acquisition object
    acq = BiosignalAcquisition(stream_type="ECG")
    
    # Connect to stream
    if acq.connect():
        # Acquire 10 seconds of data
        samples, timestamps = acq.acquire_duration(10, verbose=True)
        print(f"Acquired {len(samples)} samples")
        print(f"Sample rate: {len(samples) / 10:.2f} Hz")
        
        # Disconnect
        acq.disconnect()
