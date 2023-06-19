[~, rig] = system('hostname');
rig = rig(1:end-1);
obj.rig = rig; % to rewrite this, such that the rig is basically zelda time1? 
expPaths = dat.paths;         
obj.laserConfig = load([expPaths.rigConfig '\' 'LaserConfig.mat']);

powerMeterRange =150; % you need to set a range on the voltmeter and set an upper bound.
% read in the power from screen at max LED on. Give in milliWatts. 

%%
laserChannel = 'ao0';
powerMeterChannel = 'ai5'; 
s = daq.createSession('ni');
s.addAnalogOutputChannel('Dev1', laserChannel, 'Voltage');

ch = s.addAnalogInputChannel('Dev1', powerMeterChannel, 'Voltage');
ch.TerminalConfig = 'SingleEnded';
s.Rate = 10000; 

%% run calibration in the loop
maxVoltage = 5;
voltages = linspace(0, maxVoltage, 10);
for iV = 1:numel(voltages)
    V = voltages(iV);
    s.queueOutputData(V*ones(1*s.Rate, 1)); % output the desired Voltage; 
    phdData_voltage{iV} = s.startForeground();
    aveData_voltage(iV) = mean(phdData_voltage{iV}(round(end/2):end));
end

s.outputSingleScan(0); % to stop outputting voltage;
%%

%%
figure; 
% cacluate measured power from measured ao
aveData_power = aveData_voltage/2*powerMeterRange; 
plot(voltages, aveData_power);
xlabel('laser control [V]');
ylabel('Power [\mW]');



%% lets save all the data in a file
sv.voltages = voltages;
sv.phdData = phdData_voltage;
sv.maxRange = maxRange;
sv.aveData = aveData_power; % already in milliwatts (or whatever you set your range to); 
sv.laserChannel = laserChannel;
sv.laserID = laserID;
[file,path] = uiputfile('*.mat'); %select file name
save(fullfile(path, file), '-struct', 'sv');
