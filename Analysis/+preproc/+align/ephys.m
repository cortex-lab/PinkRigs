function [ephysRefTimes, timelineRefTimes, ephysPath] = ephys(expPath,varargin)
    %%% This function will align the flipper of the ephys data to the
    %%% flipper taken from the timeline.
    %%%
    %%% This code is inspired by the code from kilotrode
    %%% (https://github.com/cortex-lab/kilotrodeRig) and Pip's 
    %%% ephys2timeline script.
    
    %% Get parameters
    % Parameters for processing (can be inputs in varargin{1})
    params.ephysPath = []; % for specific ephys folders (give full path)
    params.toleranceThreshold = 0.005;
    
    if ~isempty(varargin)
        paramsIn = varargin{1};
        params = parseInputParams(params,paramsIn);
        
        if numel(varargin) > 1
            timeline = varargin{2};
        end
    end
    
    [subject, expDate, ~, server] = parseExpPath(expPath);
    
    %% Get timeline flipper times
    
    % Get timeline
    if ~exist('timeline','var')
        fprintf(1, 'Loading timeline\n');
        timeline = getTimeline(expPath);
    end
    % Detect sync events from timeline
    timelineFlipperTimes = timeproc.getChanEventTime(timeline,'flipper');

    %% Get all ephys flipper times
    
    ephysPath = params.ephysPath;
    
    % Get ephys folders
    % Will work only if the architecture is good.
    if isempty(ephysPath)
        % Just take them all, whatever the architecture..?
        ephysFiles = dir(fullfile(server,'Subjects',subject,expDate,'ephys','**','*.ap.bin'));
        if isempty(ephysFiles)
            error('No ephys file here: %s', fullfile(server,'Subjects',subject,expDate,'ephys'))
        else
            ephysPath = {ephysFiles.folder};
        end
    end
    
    % Get the sync for each recording
    ephysFlipperTimes = cell(1,numel(ephysPath));
    
    for ee = 1:numel(ephysPath)
        % Get syncData
        dataFile = dir(fullfile(ephysPath{ee},'*ap.bin'));
        metaS = readMetaData_spikeGLX(dataFile.name,dataFile.folder);
        
        % Load sync data
        syncDataFile = dir(fullfile(ephysPath{ee},'sync.mat'));
        if isempty(syncDataFile)
            fprintf('Couldn''t find the sync file for %s, %s. Computing it.\n', subject, expDate)
            extractSync(fullfile(dataFile.folder,dataFile.name), str2double(metaS.nSavedChans))
            ephysFlipperTimes{ee} = [];
            syncDataFile = dir(fullfile(ephysPath{ee},'sync.mat'));
        end
        syncData = load(fullfile(syncDataFile.folder,syncDataFile.name));
        
        % Extract flips
        Fs = str2double(metaS.imSampRate);
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
        flipThresh = 1; % time between flips to define experiment gap (s)
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
                    try2alignVectors(timelineFlipperTimes,ephysFlipperTimes_cut,params.toleranceThreshold,0);
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
  