function [ephysRefTimes, timelineRefTimes, ephysPath] = ephys_AVrigs(expPath,varargin)
    %%% This function will align the flipper of the ephys data to the
    %%% flipper taken from the timeline.
    %%%
    %%% This code is inspired by the code from kilotrode
    %%% (https://github.com/cortex-lab/kilotrodeRig) and Pip's 
    %%% ephys2timeline script.
    
    %% Get parameters
    % Parameters for processing (can be inputs in varargin{1})
    ephysPath = []; % for specific ephys folders (give full path)
    toleranceThreshold = 0.005;
    [subject, expDate, ~, server] = parseExpPath(expPath);
    
    % This is not ideal
    if ~isempty(varargin)
        params = varargin{1};
        
        if ~isempty(params) && isfield(params, 'alignType')
            ephysPath = params.ephysPath;
        end
        if ~isempty(params) && isfield(params, 'toleranceThreshold')
            toleranceThreshold = params.toleranceThreshold;
        end
        
        if numel(varargin) > 1
            timeline = varargin{2};
        end
    end
    
    %% Get timeline flipper times
    
    % Get timeline
    if ~exist('timeline','var')
        fprintf(1, 'loading timeline\n');
        timeline = getTimeline(expPath);
    end
    % Detect sync events from timeline
    timelineFlipperTimes = timepro.getChanEventTime(timeline,'flipper');

    %% Get all ephys flipper times
    
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
            extractSync(fullfile(dataFile.folder,dataFile.name), str2num(metaS.nSavedChans))
            ephysFlipperTimes{ee} = [];
            syncDataFile = dir(fullfile(ephysPath{ee},'sync.mat'));
        end
        syncData = load(fullfile(syncDataFile.folder,syncDataFile.name));
        
        % Extract flips
        Fs = str2num(metaS.imSampRate);
        tmp = abs(diff(syncData.sync));
        ephysFlipperTimes{ee} = find(tmp>=median(tmp(tmp>0)))/Fs; % there can be sometimes spiky noise that creates problems here
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
        [~, currExpIdx] = min(abs(experimentDurations-timeline.rawDAQTimestamps(end)));
        % Subselect the ephys flipper flip times
        ephysFlipperTimes_cut = ephysFlipperTimes_ee(flipperStEnIdx(currExpIdx,1):flipperStEnIdx(currExpIdx,2));
        
        % Check that number of flipper flips in timeline matches ephys
        success = 0;
        numFlipsDiff = abs(diff([length(ephysFlipperTimes_cut) length(timelineFlipperTimes)]));
        if numFlipsDiff>0 && numFlipsDiff<20
            fprintf([subject ' ' expDate ': WARNING = Flipper flip times different in timeline/ephys \n']);
            
            if diff([length(ephysFlipperTimes_cut) length(timelineFlipperTimes)])<20 && length(ephysFlipperTimes_cut) > 500
                fprintf([subject ' ' expDate ': Trying to account for missing flips.../ephys \n']);
                
                while length(timelineFlipperTimes) > length(ephysFlipperTimes_cut)
                    compareVect = [ephysFlipperTimes_cut-(ephysFlipperTimes_cut(1)) timelineFlipperTimes(1:length(ephysFlipperTimes_cut))-timelineFlipperTimes(1)];
                    errPoint = find(abs(diff(diff(compareVect,[],2)))>toleranceThreshold,1);
                    timelineFlipperTimes(errPoint+2) = [];
                    ephysFlipperTimes_cut(errPoint-2:errPoint+2) = [];
                    timelineFlipperTimes(errPoint-2:errPoint+2) = [];
                end
                while length(timelineFlipperTimes) < length(ephysFlipperTimes_cut)
                    compareVect = [timelineFlipperTimes-(timelineFlipperTimes(1)) ephysFlipperTimes_cut(1:length(timelineFlipperTimes))-ephysFlipperTimes_cut(1)];
                    errPoint = find(abs(diff(diff(compareVect,[],2)))>toleranceThreshold,1);
                    ephysFlipperTimes_cut(errPoint+2) = [];
                    ephysFlipperTimes_cut(errPoint-2:errPoint+2) = [];
                    timelineFlipperTimes(errPoint-2:errPoint+2) = [];
                end
                compareVect = [ephysFlipperTimes_cut-(ephysFlipperTimes_cut(1)) timelineFlipperTimes-timelineFlipperTimes(1)];
                if isempty(find(abs(diff(diff(compareVect,[],2)))>toleranceThreshold,1)); fprintf('Success! \n');
                    success = 1;
                end
            end
        elseif numFlipsDiff==0
            compareVect = [ephysFlipperTimes_cut-(ephysFlipperTimes_cut(1)) timelineFlipperTimes-timelineFlipperTimes(1)];
            if isempty(find(abs(diff(diff(compareVect,[],2)))>toleranceThreshold,1)); fprintf('Success! \n');
                success = 1;
            end
        end
        
        if success
            ephysRefTimes{ee} = ephysFlipperTimes_cut;
            timelineRefTimes{ee} = timelineFlipperTimes;
        else
            ephysRefTimes{ee} = [];
            timelineRefTimes{ee} = [];
        end
    end
    
    %% Select only the ones that were matched
    ephysPath(cellfun(@(x) isempty(x),timelineRefTimes)) = [];
    ephysRefTimes(cellfun(@(x) isempty(x),timelineRefTimes)) = [];
    timelineRefTimes(cellfun(@(x) isempty(x),timelineRefTimes)) = [];
  