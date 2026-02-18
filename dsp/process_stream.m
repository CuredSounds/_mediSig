% Biosignal_MVP: Real-time DSP Pipeline
% --------------------------------------------------
% This script connects to the 'Simulated_ECG' LSL stream and visualizes it
% using the DSP System Toolbox's Spectrum Analyzer.
%
% REQUIREMENTS:
% 1. MATLAB
% 2. DSP System Toolbox
% 3. LabStreamingLayer for MATLAB
%    (https://github.com/labstreaminglayer/liblsl-Matlab)
%    Ensure 'liblsl-Matlab' is in your MATLAB path.

% --- CONFIGURATION ---
lib = lsl_loadlib();

disp('Looking for ECG stream...');
result = {};
while isempty(result)
    result = lsl_resolve_byprop(lib, 'type', 'ECG');
end

disp('Stream found! Connecting...');
inlet = lsl_inlet(result{1});

% --- DSP SYSTEM TOOLBOX SETUP ---
% Create a Time Scope for time-domain visualization
timeScope = dsp.TimeScope(...
    'SampleRate', 100, ...
    'TimeSpan', 10, ...
    'BufferLength', 1000, ...
    'YLimits', [-1, 1], ...
    'Title', 'Real-Time ECG Signal');

% Create a Spectrum Analyzer for frequency-domain visualization
spectrumAnalyzer = dsp.SpectrumAnalyzer(...
    'SampleRate', 100, ...
    'Title', 'ECG Power Spectrum', ...
    'PlotAsTwoSidedSpectrum', false);

disp('Starting Real-Time Processing Loop...');
disp('Close the Time Scope window to stop.');

try
    while isvalid(timeScope)
        % Pull a chunk of data
        % [samples, timestamps] = inlet.pull_chunk();
        
        % For better real-time performance with DSP objects, dragging single samples
        % or small chunks is fine.
        [sample, TS] = inlet.pull_sample(0.01); % 10ms timeout
        
        if ~isempty(sample)
            % Process and Visualize
            % 1. Time Domain
            timeScope(sample(1));
            
            % 2. Frequency Domain
            spectrumAnalyzer(sample(1));
            
            % --- ADD YOUR CUSTOM DSP ALGORITHMS HERE ---
            % e.g., Filter = dsp.HighpassFilter...
            % filtered_sample = Filter(sample(1));
        end
        
        drawnow limitrate; 
    end
catch ME
    disp(['Processing stopped: ', ME.message]);
end

disp('Pipeline finished.');
