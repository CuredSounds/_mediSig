**MVP "Biosignal aka MediSig"**.

This script creates your **Synthetic Biosignal Generator**. It fulfills the "Simulation" requirement from your Tab 2 by creating two LSL streams (ECG and Respiration) without needing the physical sensors connected yet.

### **Option 1: The "God Prompt" for your IDE Agent**

If you want your Antigravity or JetBrains AI to build the whole project structure for you, paste this prompt into the chat window:

> "Create a Python project named 'Biosignal_MVP' for real-time physiological data acquisition.
> 1. Set up a virtual environment and a `requirements.txt` with: `pylsl`, `numpy`, `scipy`, `neurokit2`.
> 2. Create a folder structure: `acquisition`, `dsp`, `simulation`.
> 3. Write a Python script in `simulation/mock_stream.py` that generates two LSL outlets: one for 'ECG' (simulating heart rate) and one for 'Respiration'.
> 4. Ensure the script pushes samples at 100Hz."
> 
> 

---

### **Option 2: The Manual Code (Copy & Paste)**

If you prefer to just run it yourself, here is the complete, working code.

**1. Install Dependencies (Terminal):**

```bash
pip install pylsl numpy neurokit2

```

**2. Create the file `mock_biosignals.py`:**

```python
"""
Biosignal_MVP: Stage 1 - Synthetic Data Generator
--------------------------------------------------
This script simulates a human physiology stream (ECG & Respiration) 
and broadcasts it over LSL (Lab Streaming Layer).

It serves as a placeholder for the Polar H10 and Note 9 until 
the hardware integration is complete.
"""

import time
import numpy as np
import neurokit2 as nk
from pylsl import StreamInfo, StreamOutlet

# --- CONFIGURATION ---
SAMPLING_RATE = 100  # Hz
CHUNK_SIZE = 1       # Push 1 sample at a time for real-time feel

def main():
    print("Initializing Biosignal Simulation Layer...")

    # 1. SETUP LSL OUTLETS
    # Stream A: ECG (Heart)
    info_ecg = StreamInfo(name='Simulated_ECG', type='ECG', channel_count=1, 
                          nominal_srate=SAMPLING_RATE, channel_format='float32', 
                          source_id='sim_ecg_001')
    outlet_ecg = StreamOutlet(info_ecg)

    # Stream B: Respiration (Lungs)
    info_resp = StreamInfo(name='Simulated_Resp', type='Respiration', channel_count=1, 
                           nominal_srate=SAMPLING_RATE, channel_format='float32', 
                           source_id='sim_resp_001')
    outlet_resp = StreamOutlet(info_resp)

    print(f"Streams Active! Broadcasting at {SAMPLING_RATE} Hz.")
    print("Press Ctrl+C to stop.")

    # 2. GENERATE SYNTHETIC DATA (10 seconds buffer to loop)
    # We use neurokit2 to create realistic biological noise/shapes
    duration = 10
    print("Generating synthetic waveforms...")
    ecg_signal = nk.ecg_simulate(duration=duration, sampling_rate=SAMPLING_RATE, heart_rate=70)
    resp_signal = nk.rsp_simulate(duration=duration, sampling_rate=SAMPLING_RATE, respiratory_rate=15)
    
    # Iterator to loop through the buffer
    idx = 0
    buffer_len = len(ecg_signal)

    # 3. STREAMING LOOP
    try:
        while True:
            # Get current sample
            val_ecg = ecg_signal[idx]
            val_resp = resp_signal[idx]

            # Push to LSL network
            outlet_ecg.push_sample([val_ecg])
            outlet_resp.push_sample([val_resp])

            # Loop index
            idx = (idx + 1) % buffer_len

            # Maintain timing (simple sleep for MVP; precise timing comes later)
            time.sleep(1.0 / SAMPLING_RATE)

    except KeyboardInterrupt:
        print("\nStreaming stopped.")

if __name__ == "__main__":
    main()

```

### **How to verify it works**

You need a "Listener" to confirm the data is actually flowing.

**Create a second file called `test_listener.py`:**

```python
from pylsl import resolve_stream, StreamInlet
import time

print("Looking for an ECG stream...")
streams = resolve_stream('type', 'ECG')

if not streams:
    print("No streams found! Make sure mock_biosignals.py is running.")
else:
    inlet = StreamInlet(streams[0])
    print("Stream found! Displaying live data:")
    
    while True:
        sample, timestamp = inlet.pull_sample()
        print(f"Received: {sample[0]:.4f} at time {timestamp}")

```

**Next Step:**
Run `mock_biosignals.py` in one terminal tab, and `test_listener.py` in another. If you see numbers scrolling, your architecture is alive.

Let me know once you have confirmed the "heartbeat" is working! s