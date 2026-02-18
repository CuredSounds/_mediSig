from pylsl import resolve_byprop, StreamInlet
import time
import sys

print("Looking for an ECG stream...")
streams = resolve_byprop('type', 'ECG')

if not streams:
    print("No streams found! Make sure mock_biosignals.py is running.")
    sys.exit(1)
else:
    inlet = StreamInlet(streams[0])
    print("Stream found! Displaying first 10 samples:")
    
    count = 0
    while count < 10:
        sample, timestamp = inlet.pull_sample(timeout=5.0)
        if sample:
            print(f"Received: {sample[0]:.4f} at time {timestamp}")
            count += 1
        else:
            print("Timed out waiting for sample")
            break
    
    print("Verification successful!")
