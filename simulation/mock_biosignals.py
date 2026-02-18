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

    # Stream C: EEG (Brain - 4 Channels)
    # 4 Channels: Frontal-Left, Frontal-Right, Parietal-Left, Parietal-Right
    info_eeg = StreamInfo(name='Simulated_EEG', type='EEG', channel_count=4, 
                          nominal_srate=SAMPLING_RATE, channel_format='float32', 
                          source_id='sim_eeg_001')
    outlet_eeg = StreamOutlet(info_eeg)

    print(f"Streams Active! Broadcasting at {SAMPLING_RATE} Hz.")
    print("Press Ctrl+C to stop.")

    # 2. GENERATE SYNTHETIC DATA (10 seconds buffer to loop)
    # We use neurokit2 to create realistic biological noise/shapes
    duration = 10
    print("Generating synthetic waveforms...")
    ecg_signal = nk.ecg_simulate(duration=duration, sampling_rate=SAMPLING_RATE, heart_rate=70)
    resp_signal = nk.rsp_simulate(duration=duration, sampling_rate=SAMPLING_RATE, respiratory_rate=15)
    
    # Generate 4 channels of EEG (Alpha waves + noise)
    # We'll create slightly different signals for each channel
    eeg_c1 = nk.signal_simulate(duration=duration, sampling_rate=SAMPLING_RATE, frequency=10, amplitude=0.5, noise=0.1) # Alpha
    eeg_c2 = nk.signal_simulate(duration=duration, sampling_rate=SAMPLING_RATE, frequency=11, amplitude=0.5, noise=0.1)
    eeg_c3 = nk.signal_simulate(duration=duration, sampling_rate=SAMPLING_RATE, frequency=14, amplitude=0.3, noise=0.2) # Beta
    eeg_c4 = nk.signal_simulate(duration=duration, sampling_rate=SAMPLING_RATE, frequency=3, amplitude=1.0, noise=0.05) # Delta
    eeg_signal = np.vstack([eeg_c1, eeg_c2, eeg_c3, eeg_c4]).T
    
    # Iterator to loop through the buffer
    idx = 0
    buffer_len = len(ecg_signal)

    # 3. STREAMING LOOP
    try:
        while True:
            # Get current sample
            val_ecg = ecg_signal[idx]
            val_resp = resp_signal[idx]
            val_eeg = eeg_signal[idx]

            # Push to LSL network
            outlet_ecg.push_sample([val_ecg])
            outlet_resp.push_sample([val_resp])
            outlet_eeg.push_sample(val_eeg)

            # Loop index
            idx = (idx + 1) % buffer_len

            # Maintain timing (simple sleep for MVP; precise timing comes later)
            time.sleep(1.0 / SAMPLING_RATE)

    except KeyboardInterrupt:
        print("\nStreaming stopped.")

if __name__ == "__main__":
    main()
