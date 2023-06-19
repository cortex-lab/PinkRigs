s = daq.createSession('ni');
s.addAnalogOutputChannel('Dev1', 'ao0', 'Voltage');
s.Rate = 10000;

rampUpDur = 0.05; % in ms
flatDur = 360.5; % in ms
rampDownDur = 0.35; % in ms
% create a waveform and pre-load it onto the board;
s.queueOutputData([linspace(0,1,rampUpDur*s.Rate)';ones(flatDur*s.Rate,1);linspace(1,0,rampDownDur*s.Rate)']*5);
%% after starting it
s.startBackground; 
