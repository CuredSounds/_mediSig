from pylsl import resolve_byprop, StreamInlet
import time

print("Looking for an ECG stream...")
streams = resolve_byprop('type', 'ECG')

if not streams:
    print("No streams found! Make sure mock_biosignals.py is running.")
else:
    inlet = StreamInlet(streams[0])
    print("Stream found! Displaying live data:")
    
    while True:
        try:
            sample, timestamp = inlet.pull_sample()
            print(f"Received: {sample[0]:.4f} at time {timestamp}")
        except KeyboardInterrupt:
            print("\nListener stopped.")
            break
