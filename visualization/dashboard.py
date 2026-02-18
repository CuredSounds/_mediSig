import streamlit as st
import numpy as np
import pandas as pd
import time
from pylsl import resolve_byprop, StreamInlet
import matplotlib.pyplot as plt
import os
from datetime import datetime

# --- CONFIGURATION ---
WINDOW_SECONDS = 10
SAMPLING_RATE = 100
BUFFER_SIZE = WINDOW_SECONDS * SAMPLING_RATE

# --- SETUP SESSION STATE ---
if 'data_ecg' not in st.session_state:
    st.session_state.data_ecg = np.zeros(BUFFER_SIZE)
if 'data_resp' not in st.session_state:
    st.session_state.data_resp = np.zeros(BUFFER_SIZE)
if 'data_eeg' not in st.session_state:
    st.session_state.data_eeg = np.zeros((BUFFER_SIZE, 4)) # 4 Channels
if 'inlet_ecg' not in st.session_state:
    st.session_state.inlet_ecg = None
if 'inlet_resp' not in st.session_state:
    st.session_state.inlet_resp = None
if 'inlet_eeg' not in st.session_state:
    st.session_state.inlet_eeg = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

def connect_lsl():
    """Attempts to connect to LSL streams."""
    if st.session_state.inlet_ecg is None:
        streams_ecg = resolve_byprop('type', 'ECG', timeout=1)
        if streams_ecg:
            st.session_state.inlet_ecg = StreamInlet(streams_ecg[0])
            st.toast("Connected to ECG Stream", icon="❤️")
        else:
            st.error("ECG Stream not found")

    if st.session_state.inlet_resp is None:
        streams_resp = resolve_byprop('type', 'Respiration', timeout=1)
        if streams_resp:
            st.session_state.inlet_resp = StreamInlet(streams_resp[0])
            st.toast("Connected to Respiration Stream", icon="🫁")
        else:
            st.error("Respiration Stream not found")

    if st.session_state.inlet_eeg is None:
        streams_eeg = resolve_byprop('type', 'EEG', timeout=1)
        if streams_eeg:
            st.session_state.inlet_eeg = StreamInlet(streams_eeg[0])
            st.toast("Connected to EEG Stream", icon="🧠")
        else:
            st.error("EEG Stream not found")

def update_buffers():
    """Pulls new samples and updates the buffers."""
    if st.session_state.inlet_ecg:
        samples, timestamps = st.session_state.inlet_ecg.pull_chunk()
        if samples:
            new_data = np.array(samples).flatten()
            st.session_state.data_ecg = np.roll(st.session_state.data_ecg, -len(new_data))
            st.session_state.data_ecg[-len(new_data):] = new_data
    
    if st.session_state.inlet_resp:
        samples, timestamps = st.session_state.inlet_resp.pull_chunk()
        if samples:
            new_data = np.array(samples).flatten()
            st.session_state.data_resp = np.roll(st.session_state.data_resp, -len(new_data))
            st.session_state.data_resp[-len(new_data):] = new_data

    if st.session_state.inlet_eeg:
        samples, timestamps = st.session_state.inlet_eeg.pull_chunk()
        if samples:
            new_data = np.array(samples) # Shape: (n_samples, 4)
            n_new = len(new_data)
            st.session_state.data_eeg = np.roll(st.session_state.data_eeg, -n_new, axis=0)
            st.session_state.data_eeg[-n_new:, :] = new_data

def save_snapshot():
    """Saves the current buffer and plot to disk."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "captured_data"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save CSV
    data_dict = {
        'ECG': st.session_state.data_ecg,
        'Respiration': st.session_state.data_resp,
    }
    # Add EEG cols
    for i in range(4):
        data_dict[f'EEG_C{i+1}'] = st.session_state.data_eeg[:, i]
        
    df = pd.DataFrame(data_dict)
    csv_path = os.path.join(save_dir, f"biosignals_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    
    # Save Plot Image
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(st.session_state.data_ecg, color='red')
    ax1.set_title("ECG")
    
    ax2.plot(st.session_state.data_resp, color='blue')
    ax2.set_title("Respiration")
    
    # EEG Plot (Multi-channel)
    ax3.plot(st.session_state.data_eeg)
    ax3.set_title("EEG (4 Channels)")
    ax3.legend([f'Ch{i+1}' for i in range(4)], loc='upper right', fontsize='small')
    
    plt.tight_layout()
    img_path = os.path.join(save_dir, f"plot_{timestamp}.png")
    plt.savefig(img_path)
    plt.close(fig)
    
    st.toast(f"Snapshot saved: {timestamp}", icon="📸")

# --- UI LAYOUT ---
st.title("MediSig: Real-time Biosignal Visualizer")

col1, col2 = st.columns(2)
with col1:
    if st.button("Connect Streams"):
        connect_lsl()

with col2:
    if st.button("Capture Snapshot for ML"):
        save_snapshot()

# Real-time Loop
placeholder = st.empty()

if st.session_state.inlet_ecg or st.session_state.inlet_resp or st.session_state.inlet_eeg:
    while True:
        update_buffers()
        
        with placeholder.container():
            # Create charts
            st.subheader("Physiology (ECG + Resp)")
            chart_physio = pd.DataFrame({
                "ECG": st.session_state.data_ecg,
                "Respiration": st.session_state.data_resp
            })
            st.line_chart(chart_physio)
            
            st.subheader("Brain Activity (EEG)")
            chart_eeg = pd.DataFrame(
                st.session_state.data_eeg,
                columns=['F-Left', 'F-Right', 'P-Left', 'P-Right']
            )
            st.line_chart(chart_eeg)
            
            time.sleep(0.05)
else:
    st.info("Click 'Connect Streams' to start.")
