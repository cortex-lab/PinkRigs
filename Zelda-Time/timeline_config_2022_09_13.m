% Create a timeline object to be saved in hardware.mat
% Configured for ZELDA-TIMELINE
% Taken from Lilrig

% Instantiate the timeline object
timeline = hw.Timeline;

% Set sample rate
timeline.DaqSampleRate = 1000;

% Set up function for configuring inputs
daq_input = @(name, channelID, measurement, terminalConfig) ...
    struct('name', name,...
    'arrayColumn', -1,... % -1 is default indicating unused
    'daqChannelID', channelID,...
    'measurement', measurement,...
    'terminalConfig', terminalConfig, ...
    'axesScale', 1);

% Configure inputs
timeline.Inputs = [...
    daq_input('chrono', 'ai0', 'Voltage', 'SingleEnded')... % for reading back self timing wave
    daq_input('rotaryEncoder', 'ctr0', 'Position', [])...
    daq_input('photoDiode', 'ai12', 'Voltage', 'SingleEnded')...
    daq_input('photoDThorLabs', 'ai11', 'Voltage', 'SingleEnded')...
    daq_input('rewardEcho', 'ai6', 'Voltage', 'SingleEnded')...
    daq_input('acqLive', 'ai9', 'Voltage', 'SingleEnded')...
    daq_input('flipper', 'ai10', 'Voltage', 'SingleEnded')...
    daq_input('camSync', 'ai3', 'Voltage', 'SingleEnded')...  
    daq_input('audioOut', 'ai4', 'Voltage', 'SingleEnded')...
    daq_input('breathMonitor', 'ai5', 'Voltage', 'SingleEnded')...
    daq_input('microphoneOut', 'ai13', 'Voltage', 'SingleEnded')...    
   
    ];
    
% bu
% daq_input('sideCamStrobe', 'ai13', 'Voltage', 'SingleEnded')...
% daq_input('frontCamStrobe', 'ai14', 'Voltage', 'SingleEnded')...
% daq_input('eyeCamStrobe', 'ai15', 'Voltage', 'SingleEnded')...
    
% Activate all defined inputs
timeline.UseInputs = {timeline.Inputs.name};

% Configure outputs (each output is a specialized object)

% (chrono - required timeline self-referential clock)
chronoOutput = hw.TLOutputChrono;
chronoOutput.DaqChannelID = 'port0/line0';

% (acq live output - for external triggering)
acqLiveOutput = hw.TLOutputAcqLive;
acqLiveOutput.Name = 'acqLive'; % rename for legacy compatability
acqLiveOutput.DaqChannelID = 'port0/line1';

% (output to synchronize face camera)
camSyncOutput = hw.TLOutputCamSync;
camSyncOutput.Name = 'camSync'; % rename for legacy compatability
camSyncOutput.DaqChannelID = 'port0/line2';
camSyncOutput.PulseDuration = 0.2;
camSyncOutput.InitialDelay = 0.5;

% Package the outputs (VERY IMPORTANT: acq triggers illum, so illum must be
% set up BEFORE starting acqLive output)
timeline.Outputs = [chronoOutput,acqLiveOutput,camSyncOutput];

% Configure live "oscilliscope"
timeline.LivePlot = true;

% Clear out all temporary variables
clearvars -except timeline

% save to "hardware" file
% copy backup of hardware file?
rig = hostname;
hardwarefile_address = fullfile('\\zserver.cortexlab.net\Code\Rigging\config', upper(rig), 'hardware.mat');
copyfile(hardwarefile_address, ...
    [hardwarefile_address(1:end-4) '_' regexprep(regexprep(datestr(now), ' ', '_'),':','-') '.mat'])
save(hardwarefile_address,'timeline','-append')
disp('Saved ZELDA-TIMELINE config file')
