# Biosignal MVP

A Python application for biosignal acquisition, processing, and simulation using Lab Streaming Layer (LSL).

## Project Structure

```
Biosignal_MVP/
├── acquisition/     # Data acquisition scripts
├── dsp/            # Digital signal processing modules
├── simulation/     # Simulation scripts
├── requirements.txt
├── main_recorder.py
└── README.md
```

## Setup

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Create a Python virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

The project requires the following Python packages:

- **pylsl**: Lab Streaming Layer for real-time data streaming
- **numpy**: Numerical computing library
- **scipy**: Scientific computing library
- **neurokit2**: Neurophysiological signal processing
- **bleak**: Bluetooth Low Energy platform support

## IDE Recommendations

### 1. Google Antigravity (Agent-First IDE)

**Use this for: Project initialization and boilerplate generation**

Google Antigravity is a modified fork of VS Code designed for AI agents, ideal for:
- Creating folder structures
- Generating boilerplate code
- Setting up initial project scaffolding

**How to use:**
1. Open Antigravity and navigate to the `Biosignal_MVP` folder
2. Use the Manager View to delegate tasks to the AI agent
3. Example prompt: *"Create a Python environment structure for a biosignal application. I need a folder for acquisition, dsp, and simulation. Please create a `requirements.txt` file that includes `pylsl`, `numpy`, `scipy`, `neurokit2`, and `bleak`."*

**Benefits:**
- Saves 15-20 minutes of manual setup
- Automatic file and folder creation
- Intelligent dependency management

### 2. JetBrains PyCharm (The "Powerhouse")

**Use this for: Debugging and running the live system**

PyCharm excels at:
- Superior debugging capabilities
- Variable explorer (similar to MATLAB's workspace)
- Real-time execution monitoring
- Complex import handling

**How to use:**
1. Open the `Biosignal_MVP` project in PyCharm
2. Configure the Python Interpreter:
   - Go to `File > Settings > Project > Python Interpreter`
   - Select the `venv` created in the installation steps
3. Run `main_recorder.py` and use PyCharm's debugger to step through code
4. Use the variable explorer to monitor LSL streams and biosignal data

**When to use PyCharm:**
- Debugging LSL stream synchronization issues
- Inspecting heart rate data anomalies
- Live execution of recording scripts
- Troubleshooting complex imports like `pylsl`

### Recommended Workflow

1. **Project Initialization**: Use **Google Antigravity** to generate directory structure and initial scripts
2. **Live Coding & Debugging**: Open the same folder in **PyCharm** for real-time execution and debugging

**Note**: The Python code remains the same regardless of IDE choice. The only difference is *who* writes the boilerplate (Antigravity) and *where* you press "Run" (PyCharm).

## Usage

### Running the Main Recorder

```bash
python main_recorder.py
```

This script demonstrates:
- Creating LSL outlets for streaming biosignal data
- Finding available LSL streams on the network
- Recording data from LSL inlets
- Basic signal acquisition workflow

### Example: Creating a Custom Stream

```python
from pylsl import StreamInfo, StreamOutlet
import time
import numpy as np

# Create stream info
info = StreamInfo('MyECG', 'ECG', 1, 250, 'float32', 'myecg001')

# Create outlet
outlet = StreamOutlet(info)

# Stream data
while True:
    sample = [np.random.randn()]  # Your actual signal here
    outlet.push_sample(sample)
    time.sleep(1/250)  # 250 Hz
```

### Example: Recording from a Stream

```python
from pylsl import resolve_streams, StreamInlet
import numpy as np

# Find streams
streams = resolve_streams(timeout=5.0)

# Connect to first stream
inlet = StreamInlet(streams[0])

# Record data
samples = []
for _ in range(1000):  # Record 1000 samples
    sample, timestamp = inlet.pull_sample()
    samples.append(sample)

data = np.array(samples)
```

## Development

### Module Organization

- **acquisition/**: Scripts for acquiring biosignals from hardware devices (EEG, ECG, PPG, etc.)
- **dsp/**: Signal processing algorithms (filtering, feature extraction, etc.)
- **simulation/**: Generate synthetic biosignals for testing

### Adding New Modules

Each module should have its own `__init__.py` file and follow the existing structure.

## Troubleshooting

### LSL Stream Not Found

- Ensure the data source is running and creating an outlet
- Check network connectivity if streaming across devices
- Verify firewall settings allow LSL traffic

### Import Errors

- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check Python version compatibility (3.7+)

## License

This project is part of the _mediSig repository.
