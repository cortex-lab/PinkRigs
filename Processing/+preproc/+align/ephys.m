function [ephysRefTimesReord, timelineRefTimesReord, ephysPathReord, serialNumberReord] = ephys(varargin)
    %% Aligns the ephys data with its corresponding timeline file.
    % Note: This code is inspired by the code from kilotrode 
    % (https://github.com/cortex-lab/kilotrodeRig) and Pip's ephys2timeline script.
    %
    % Parameters:
    % -------------------
    % Classic PinkRigs inputs (optional).
    % ephysPath: cell of str
    %   Specific list of paths
    % toleranceThreshold: double
    %   Tolerance threshold in try2alignVectors    
    % timeline: struct
    %   Timeline structure
    % 
    %
    % Returns: 
    % -------------------
    % ephysRefTimesReord: vector
    %   Flipper switch times in ephys time
    % timelineRefTimesReord: vector
    %   Flipper switch times in timeline time
    % ephysPathReord: cell array
    %   Associated ephys paths.
    % serialNumberReord: cell array
    %   Serial number of the probe for each recording in ephysPathReord

    %% Get parameters
    % Parameters for processing (can be inputs in varargin{1})
    varargin = ['ephysPath', {[]}, varargin]; % for specific ephys folders (give full path)
    varargin = ['toleranceThreshold', {0.005}, varargin];
    varargin = ['timeline', {[]}, varargin]; % for timeline input
    params = csv.inputValidation(varargin{:});
    
    subject = params.subject{1};
    expDate = params.expDate{1};

    %% Get timeline flipper times
    % Get timeline
    if ~isfield(params,'dataTimeline')
        fprintf(1, 'Loading timeline\n');
        loadedData = csv.loadData(params, 'dataType','timeline');
        timeline = loadedData.dataTimeline{1};
    else
        timeline = params.dataTimeline{1};
    end
    % Detect sync events from timeline
    timelineFlipperTimes = timeproc.getChanEventTime(timeline,'flipper');

    %% Get all ephys flipper times
    
    ephysPath = params.ephysPath{1};
    
    % Get ephys folders
    % Will work only if the architecture is good.
    if isempty(ephysPath)
        % Just take them all, whatever the architecture..?
        ephysFiles = dir(fullfile(fileparts(params.expFolder{1}),'ephys','**','*.ap.*bin'));
        ephysFiles(contains({ephysFiles.folder},'.kilosort')) = [];
        if isempty(ephysFiles)
            error('No ephys file here: %s', fullfile(fileparts(params.expFolder{1}),'ephys'))
        else
            ephysPath = {ephysFiles.folder};
        end
    end
    ephysPath = unique(ephysPath); 
    
    % Get the sync for each recording
    ephysFlipperTimes = cell(1,numel(ephysPath));
    
    for ee = 1:numel(ephysPath)
        % Get meta data
        dataFile = dir(fullfile(ephysPath{ee},'*ap.*bin'));
        dataFile(contains({dataFile.folder},'.kilosort')) = [];
        metaS = readMetaData_spikeGLX(dataFile(1).name,dataFile(1).folder);
        
        % Load sync data
        syncDataFile = dir(fullfile(ephysPath{ee},'sync.mat'));
        if isempty(syncDataFile)
            fprintf('Couldn''t find the sync file for %s, %s. Computing it.\n', subject, expDate)
            try
                extractSync(fullfile(dataFile.folder,dataFile.name), str2double(metaS.nSavedChans))
            catch
                fprintf('Couldn''t extract the sync! Have a look?\n')
            end    
            ephysFlipperTimes{ee} = [];
            syncDataFile = dir(fullfile(ephysPath{ee},'sync.mat'));
        end
        if ~isempty(syncDataFile)
            syncData = load(fullfile(syncDataFile.folder,syncDataFile.name));
        else
            syncData.sync = [];
        end

        % Extract flips
        Fs = str2double(metaS.imSampRate);
        [c,~,ic] = unique(syncData.sync);
        if numel(c)>2
            warning('weird flipper, will try to process')
            syncData.sync = int16(medfilt1(single(syncData.sync),9));
            syncData.sync(1) = syncData.sync(2);
            tmp = diff(syncData.sync);
            [cd,~,icd] = unique(abs(tmp(tmp~=0)));
            h = histcounts(icd,numel(cd));
            if numel(h) == 1
                fprintf('All good, filtered it.\n')
            elseif numel(h) == 2 & any(h < 10)
                warning('looks like it''s just a shift, ignore?')
            else
                warning('looks more serious...')
                hc = histcounts(ic);
                [~,idx] = sort(hc,'descend');
                % first 2 should the real ones
                for ii = idx(3:end)
                    % find the closest one
                    idxEdge = diff(ic == ii) == 1;
                    ic(ic == ii) = mode(ic(idxEdge));
                end
                syncData.sync = c(ic);
            end
        end
        ephysFlipperTimes{ee} = (find(diff(syncData.sync)~=0)/Fs);
    end
    % Remove empty file refs
    ephysPath(cellfun(@(x) isempty(x),ephysFlipperTimes)) = [];
    ephysFlipperTimes(cellfun(@(x) isempty(x),ephysFlipperTimes)) = [];
    
    %%  Match up ephys and timeline events
    % Algorithm here is to go through each ephys available, figure out
    % whether the events in timeline align with any of those in the ephys. If
    % so, we have a conversion of events in that ephys into timeline
    
    ephysRefTimes = cell(1,numel(ephysPath));
    timelineRefTimes = cell(1,numel(ephysPath));
    
    for ee = 1:numel(ephysPath)
        ephysFlipperTimes_ee = ephysFlipperTimes{ee}';
        
        % Find the beginning and end of the experiments
        %%% This will work if there's no absurd time between ephys
        %%% start/end and timeline start/end.
        flipThresh = 5; % time between flips to define experiment gap (s)
        flipperStEnIdx = [[1;find(diff(ephysFlipperTimes_ee) > flipThresh)+1], ...
            [find(diff(ephysFlipperTimes_ee) > flipThresh); length(ephysFlipperTimes_ee)]];
        if size(flipperStEnIdx,1) == 1 
            experimentDurations = diff(ephysFlipperTimes_ee(flipperStEnIdx));
        else
            experimentDurations = diff(ephysFlipperTimes_ee(flipperStEnIdx),[],2);
        end
        % Get the current experiment (the one of which timeline duration
        % matches best)
        percMismatch = abs(experimentDurations-timeline.rawDAQTimestamps(end))./experimentDurations;
        [~, currExpIdx] = sort(percMismatch,'ascend');
        currExpIdx = currExpIdx(percMismatch(currExpIdx) < 1); % subselect only the experiments with a mismatch that's not too bad

        % Check that number of flipper flips in timeline matches ephys
        success = 0;
        % try with the others experiments that aren't too bad
        exp = 0;
        while exp <= numel(currExpIdx)
            exp = exp+1;
            try
                % Subselect the ephys flipper flip times
                ephysFlipperTimes_cut = ephysFlipperTimes_ee(flipperStEnIdx(currExpIdx(exp),1):flipperStEnIdx(currExpIdx(exp),2));
                [timelineFlipperTimes_corr, ephysFlipperTimes_cut] = ...
                    try2alignVectors(timelineFlipperTimes,ephysFlipperTimes_cut,params.toleranceThreshold{1},0);
                success = 1;
                exp = numel(currExpIdx)+1;
            catch
                success = 0;
            end
        end
                
        if success
            fprintf('Success! \n')
            ephysRefTimes{ee} = ephysFlipperTimes_cut;
            timelineRefTimes{ee} = timelineFlipperTimes_corr;
        else
            ephysRefTimes{ee} = [];
            timelineRefTimes{ee} = [];
        end
    end
    
    %% Select only the ones that were matched
    ephysPath(cellfun(@(x) isempty(x),timelineRefTimes)) = [];
    ephysRefTimes(cellfun(@(x) isempty(x),timelineRefTimes)) = [];
    timelineRefTimes(cellfun(@(x) isempty(x),timelineRefTimes)) = [];
  
    %% Reorder according to the probes

    probeInfo = csv.checkProbeUse(subject);
    expectedSerial = probeInfo.serialNumbers{1};    

    if ~isempty(ephysPath)
        % Get actual serial numbers
        ephysFiles = cellfun(@(x) dir(fullfile(x,'*.*bin')), ephysPath, 'uni', 0);
        metaData = arrayfun(@(x) readMetaData_spikeGLX(x{1}(1).name, x{1}(1).folder), ephysFiles, 'uni', 0);
        serialsFromMeta = cellfun(@(x) str2double(x.imDatPrb_sn), metaData);
        createTime = cell2mat(cellfun(@(x) datenum(x.fileCreateTime(end-7:end)), metaData, 'uni', 0));

        if strcmp(probeInfo.probeType{1},'Acute')
            % No check in acute recordings
            expectedSerial = serialsFromMeta;
        end

        % Check for unexpected serial numbers
        if ~isempty(expectedSerial)
            % Throw error if unexpected SN was found
            unexpectedSerial = ~ismember(serialsFromMeta,expectedSerial);
            if any(unexpectedSerial)
                error('Unrecognized probe %d.', serialsFromMeta(unexpectedSerial))
            end
        end
    else
        serialsFromMeta = nan*ones(1,max(1,numel(expectedSerial)));
        if strcmp(probeInfo.probeType{1},'Acute')
            % No check in acute recordings
            expectedSerial = serialsFromMeta;
        end
    end

    % Reorder them
    ephysPathReord = cell(numel(expectedSerial),1);
    ephysRefTimesReord = cell(numel(expectedSerial),1);
    timelineRefTimesReord = cell(numel(expectedSerial),1);
    for pp = 1:numel(expectedSerial)
        corresProbe = serialsFromMeta == expectedSerial(pp);
        if sum(corresProbe) > 1
            warning('Several corresponding probes. May be because the flippers were identical (due to arduino reinitialization)?')
            fprintf('Take the closest one in time...')
            [m,idx] = min(abs(createTime-datenum(timeline.startDateTimeStr(end-7:end))));
            fprintf('%d minutes away. \n',m*24*60)
            corresProbe(~ismember(1:numel(corresProbe),idx)) = 0;
        end
        if any(corresProbe)
            ephysPathReord(pp) = ephysPath(corresProbe);
            ephysRefTimesReord(pp) = ephysRefTimes(corresProbe);
            timelineRefTimesReord(pp) = timelineRefTimes(corresProbe);
        else
            ephysPathReord(pp) = {'error'};
            ephysRefTimesReord(pp) = {'error'};
            timelineRefTimesReord(pp) = {'error'};
        end
    end
    serialNumberReord = expectedSerial;