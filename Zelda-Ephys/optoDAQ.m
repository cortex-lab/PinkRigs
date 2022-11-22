s = daq.createSession('ni');
s.addAnalogOutputChannel('Dev1', 'ao0', 'Voltage');
% make the AO 'listen' to TTL on Dev1/PFI0 ('RisingEdge' by default)
s.addTriggerConnection('External', 'Dev1/PFI0', 'StartTrigger')
% not sure if this makes any difference (Inf works fine)
s.ExternalTriggerTimeout = Inf;
% allow it to fire for infinite number of triggers
s.TriggersPerRun = Inf;
% define update rate
s.Rate = 10000;
% create a waveform and pre-load it onto the board
s.queueOutputData([rand(20000, 1); 0]*5);
​
%% after starting it will wait for TTL to actually fire the output waveform
s.startBackground
%% to change parameters/waveforms need to stop the session
stop(s)
%% clean up when done
delete(s); 