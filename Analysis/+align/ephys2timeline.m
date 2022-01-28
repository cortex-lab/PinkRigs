function ephys2timeline(subject, expDate, expNum)
if ~exist('kilosortOutput', 'var'); kilosortOutput = x.kilosortOutput; end
%% Load timeline and associated inputs
if ~exist(x.rawTimeline, 'file'); error('No timeline file exists for requested ephys session'); end
if ~exist('runWithoutSorting', 'var'); runWithoutSorting = 0; end
fprintf('Loading timeline... \n');
timeline = load(x.rawTimeline); timeline = timeline.Timeline;

inputNames = {timeline.hw.inputs.name}';
% Get acqLive signal
acqLiveTrace = timeline.rawDAQData(:,strcmp(inputNames, 'acqLive'));
acqLiveTrace = acqLiveTrace>(max(acqLiveTrace)/2);
% acqLiveSrtEnd = timeline.rawDAQTimestamps(find(diff(acqLiveTrace))+1);

% Get wheel position %%%%NEVER USED?
% wheelPosition = timeline.rawDAQData(:,strcmp(inputNames, 'rotaryEncoder'));
% wheelPosition(wheelPosition > 2^31) = wheelPosition(wheelPosition > 2^31) - 2^32;

%Get whether stim was flickering %%%%NEVER USED?
% if contains('stimScreen', inputNames); stimScreenFlicker = range(timeline.rawDAQData(:,strcmp('stimScreen', inputNames))) > 2; end

% Get flipper flips
flipperTrace = timeline.rawDAQData(:,strcmp(inputNames, 'flipper')) > 2;
flipperFlips = sort([strfind(flipperTrace', [0 1]), strfind(flipperTrace', [1 0])])'+1;
flipperFlipTimesTimeline = timeline.rawDAQTimestamps(flipperFlips)';

%% Load ephys data (single long recording)
fprintf('Loading ephys... \n');
% These are the digital channels going into the FPGA
if contains(x.rigName, {'lilrig-stim', 'zrig1'}); acqLiveSyncIdx = 2; flipperSyncIdx = 4;
elseif strcmpi(x.rigName, 'zatteo'); acqLiveSyncIdx = 1; flipperSyncIdx = 2;
end

%%
% Load clusters and sync/photodiode
sync = load([kilosortOutput '\sync.mat']); sync = sync.sync;

%%
% Read header information
if exist([kilosortOutput '\dat_params.txt'], 'file')
    headerFID = fopen([kilosortOutput '\dat_params.txt']);
else
    headerFID = fopen([kilosortOutput '\params.py']);
end
headerInfo = textscan(headerFID,'%s %s', 'delimiter',{' = '});
fclose(headerFID);
headerInfo = [headerInfo{1}'; headerInfo{2}'];
header = struct(headerInfo{:});
if exist([kilosortOutput '\dat_params.txt'], 'file')
    ephysSampleRate = str2double(header.apSampleRate);
else
    ephysSampleRate = str2double(header.sample_rate);
end

%%
% Load spike data
spikeTimes = double(readNPY([kilosortOutput '\spike_times.npy']))./ephysSampleRate;
spikeTemplates = readNPY([kilosortOutput '\spike_templates.npy']);
templates = readNPY([kilosortOutput '\templates.npy']);
channelPositions = readNPY([kilosortOutput '\channel_positions.npy']);
channelMap = readNPY([kilosortOutput '\channel_map.npy']); %#ok<NASGU>
winv = readNPY([kilosortOutput '\whitening_mat_inv.npy']);
templateAmplitudes = readNPY([kilosortOutput '\amplitudes.npy']);
if exist([kilosortOutput '\lfpPowerSpectra.mat'], 'file'); powerSpectra = load([kilosortOutput '\lfpPowerSpectra.mat']);
    if isfield(powerSpectra, 'powerSpectra')
        powerSpectra = mean(powerSpectra.powerSpectra,3);
    else
        powerSpectra = powerSpectra.lfpPowerSpectra;
        if ~isfield(powerSpectra, 'surfaceEst')
            kil.quickEstimateDepthKS2(powerSpectra, [kilosortOutput '\lfpPowerSpectra.mat']);
            uiwait(gcf);
            powerSpectra = load([kilosortOutput '\lfpPowerSpectra.mat']);
            powerSpectra = powerSpectra.lfpPowerSpectra;
        end
        
    end
else, powerSpectra = [];
end
%%
% Default channel map/positions are from end: make from surface
if ~contains(header.dat_path, 'imec')
    channelPositions(:,2) = 3840 - channelPositions(:,2); %% Hard coded for now
else
    for i = 1:4
        idx = (floor(channelPositions(:,1)/200)+1) == i;
        channelPositions(idx,2) = powerSpectra.surfaceEst(i) - channelPositions(idx,2); 
    end
end
if exist('flipperFlipTimesTimeline','var')
    % Get flipper experiment differences by long delays
    flipThresh = 1; % time between flips to define experiment gap (s)
    if isstruct(sync); flipTimes = sync(flipperSyncIdx).timestamps;
    else, flipTimes = (find(diff(sync)~=0)/ephysSampleRate)';
    end
    flipperStEnIdx = [[1;find(diff(flipTimes) > flipThresh)+1], [find(diff(flipTimes) > flipThresh); length(flipTimes)]];
    if size(flipperStEnIdx,1) == 1; experimentDurations = diff(flipTimes(flipperStEnIdx)); 
    else, experimentDurations = diff(flipTimes(flipperStEnIdx),[],2);
    end
    [~, currExpIdx] = min(abs(experimentDurations-timeline.rawDAQTimestamps(end)));
    flipperFlipTimesFPGA = flipTimes(flipperStEnIdx(currExpIdx,1):flipperStEnIdx(currExpIdx,2));
    
    % Check that number of flipper flips in timeline matches ephys
    numFlipsDiff = abs(diff([length(flipperFlipTimesFPGA) length(flipperFlipTimesTimeline)]));
    if numFlipsDiff>0 && numFlipsDiff<20 
        fprintf([x.subject ' ' x.expDate ': WARNING = Flipper flip times different in timeline/ephys \n']);
        if diff([length(flipperFlipTimesFPGA) length(flipperFlipTimesTimeline)])<20 && length(flipperFlipTimesFPGA) > 500
            fprintf([x.subject ' ' x.expDate ': Trying to account for missing flips.../ephys \n']);
            while length(flipperFlipTimesTimeline) > length(flipperFlipTimesFPGA)
                compareVect = [flipperFlipTimesFPGA-(flipperFlipTimesFPGA(1)) flipperFlipTimesTimeline(1:length(flipperFlipTimesFPGA))-flipperFlipTimesTimeline(1)];
                errPoint = find(abs(diff(diff(compareVect,[],2)))>0.005,1);
                flipperFlipTimesTimeline(errPoint+2) = [];
                flipperFlipTimesFPGA(errPoint-2:errPoint+2) = [];
                flipperFlipTimesTimeline(errPoint-2:errPoint+2) = [];
            end
            while length(flipperFlipTimesFPGA) < length(flipperFlipTimesTimeline)
                compareVect = [flipperFlipTimesTimeline-(flipperFlipTimesTimeline(1)) flipperFlipTimesFPGA(1:length(flipperFlipTimesTimeline))-flipperFlipTimesFPGA(1)];
                errPoint = find(abs(diff(diff(compareVect,[],2)))>0.005,1);
                flipperFlipTimesFPGA(errPoint+2) = [];
                flipperFlipTimesFPGA(errPoint-2:errPoint+2) = [];
                flipperFlipTimesTimeline(errPoint-2:errPoint+2) = [];
            end
            compareVect = [flipperFlipTimesFPGA-(flipperFlipTimesFPGA(1)) flipperFlipTimesTimeline-flipperFlipTimesTimeline(1)];
            if isempty(find(abs(diff(diff(compareVect,[],2)))>0.005,1)); fprintf('Success! \n');
                spikeTimesTimeline = interp1(flipperFlipTimesFPGA,flipperFlipTimesTimeline,spikeTimes,'linear','extrap');
            end
        end
    elseif numFlipsDiff==0, spikeTimesTimeline = interp1(flipperFlipTimesFPGA,flipperFlipTimesTimeline,spikeTimes,'linear','extrap');
    end
    
    % Get the spike/lfp times in timeline time (accounts for clock drifts)
end
%%
if ~exist('spikeTimesTimeline', 'var')
    warning([x.subject ' ' x.expDate ': WARNING = Could not sync flipper. Using AcqLive... \n']);
    % Get acqLive times for current experiment
    acqLiveTimes = sync(acqLiveSyncIdx).timestamps;
    acqLiveStEnIdx = [find(sync(acqLiveSyncIdx).values == 1) find(sync(acqLiveSyncIdx).values == 0)];
    experimentDurations = diff(acqLiveTimes(acqLiveStEnIdx),[],2);
    [~, currExpIdx] = min(abs(experimentDurations-timeline.rawDAQTimestamps(end)));
    acqLiveTimesFPGA = acqLiveTimes(acqLiveStEnIdx(currExpIdx,1):acqLiveStEnIdx(currExpIdx,2));
    acqLiveTimesTimeline = timeline.rawDAQTimestamps([find(acqLiveTrace,1),find(acqLiveTrace,1,'last')+1]);
    
    % Check that the experiment time is the same within threshold
    % (it should be almost exactly the same)
    if abs(diff(acqLiveTimesFPGA) - diff(acqLiveTimesTimeline)) > 1
        error([x.subject ' ' x.expDate ': acqLive duration different in timeline and ephys']);
    else, spikeTimesTimeline = interp1(acqLiveTimesFPGA,acqLiveTimesTimeline,spikeTimes,'linear','extrap');
    end
end

%%
% Get the depths of each template
[spikeAmps, ~, templateDepths, templateAmps, ~, templateDuration, waveforms] = ...
    kil.templatePositionsAmplitudes(templates,winv,channelPositions(:,2),spikeTemplates,templateAmplitudes);
% Eliminate spikes that were classified as not "good"
fprintf('Removing noise and MUA templates... \n');

if runWithoutSorting;  clusterGroups = tdfread([kilosortOutput '\cluster_group.tsv']);
else, clusterGroups = tdfread([kilosortOutput '\cluster_KSLabel.tsv']); clusterGroups.group = clusterGroups.KSLabel;
end
goodTemplatesList = clusterGroups.cluster_id(contains(num2cell(clusterGroups.group,2), 'good'));
goodTemplatesIdx = ismember(0:size(templates,1)-1,goodTemplatesList);

% Throw out all non-good template data
templateDepths = templateDepths(goodTemplatesIdx);
templateAmps = templateAmps(goodTemplatesIdx);
waveforms = waveforms(goodTemplatesIdx,:);
templates = templates(goodTemplatesIdx,:,:);
templateDuration = templateDuration(goodTemplatesIdx);
%%
% Throw out all non-good spike data
goodSpikeIdx = ismember(spikeTemplates,goodTemplatesList);
goodSpikeIdx(spikeTimesTimeline<-10 | spikeTimesTimeline>timeline.rawDAQTimestamps(end)+10) = 0;
spikeTemplates = spikeTemplates(goodSpikeIdx);
spikeTimesTimeline = spikeTimesTimeline(goodSpikeIdx);
spikeAmps = spikeAmps(goodSpikeIdx);

% Re-name the spike templates according to the remaining templates
% (and make 1-indexed from 0-indexed)
newSpikeIdx = nan(max(spikeTemplates)+1,1);
newSpikeIdx(goodTemplatesList+1) = 1:length(goodTemplatesList);
spikeTemplates = newSpikeIdx(spikeTemplates+1);

reducedTemplates = zeros(size(templates,1), size(templates,2)+1, 20, 'single');
for i = 1:size(templates,1)
    [~, maxSignals] = sort(max(abs(squeeze(templates(i,:,:)))), 'descend');
    reducedTemplates(i,1:end-1,:) = templates(i,:,maxSignals(1:20));
    reducedTemplates(i,end,:) = maxSignals(1:20);
end
[~, ephysFolder] = fileparts(kilosortOutput);
if strcmp(ephysFolder, 'kilosort'); ephysFolder = 'ephys'; end

spkTimesForClusters = arrayfun(@(x) single(spikeTimesTimeline(spikeTemplates==x)) ,1:length(templateAmps), 'uni', 0)';
spkAmpsForClusters = arrayfun(@(x) single(spikeAmps(spikeTemplates==x)) ,1:length(templateAmps), 'uni', 0)';
%%
eph.penetration.folder = ephysFolder;
eph.penetration.channelMap = {readNPY([kilosortOutput '\channel_positions.npy'])};
eph.penetration.lfpPowerSpectra = {powerSpectra};
eph.cluster.depths = templateDepths;
eph.cluster.amplitudes = templateAmps;
eph.cluster.duration = templateDuration;
eph.cluster.waveforms = waveforms;
eph.cluster.templates = reducedTemplates;
eph.cluster.spkTimes = spkTimesForClusters;
eph.cluster.spkAmplitudes = spkAmpsForClusters;
%%
fprintf('Finished loading experiment... \n');