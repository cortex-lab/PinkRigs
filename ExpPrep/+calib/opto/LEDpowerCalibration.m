
laserChannel = 'ao1';
powerMeterChannel = 'ai1'; 

s = daq.createSession('ni');
s.addAnalogOutputChannel('Dev1', laserChannel, 'Voltage');
s.addAnalogInputChannel('Dev1', powerMeterChannel, 'Voltage');
s.Rate = 10000;

%% run calibration in the loop
maxVoltage = 5;
voltages = linspace(0, maxVoltage, 10);
for iV = 1:numel(voltages)
    V = voltages(iV);
    s.queueOutputData(V*ones(1*s.Rate, 1)); % output the desired Voltage; 
    phdData{iV} = s.startForeground;
    aveData(iV) = mean(phdData{iV, iEmitter+1}(round(end/2):end));
    fprintf('emitter #%d, %g V : %g uW\n', iEmitter, V, aveData(iV, iEmitter+1)/2*maxRange)
end

s.outputSingleScan(0);

aveData = aveData/2*maxRange;
figure; 
plot(voltages, aveData);
xlabel('laser control [V]');
ylabel('Power at Emitter [\muW]');
legend({'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'})

%% lets save all the data in a file
sv.voltages = voltages;
sv.phdData = phdData;
sv.maxRange = maxRange;
sv.aveData = aveData; % already in microWatts
sv.laserChannel = laserChannel;
sv.laserID = laserID;
[file,path] = uiputfile('*.mat'); %select file name
save(fullfile(path, file), '-struct', 'sv');
