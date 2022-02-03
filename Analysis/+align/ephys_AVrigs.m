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
        
        if isfield(params, 'alignType')
            ephysPath = params.ephysPath;
        end
        if isfield(params, 'toleranceThreshold')
            toleranceThreshold = params.toleranceThreshold;
        end
        
        if nargin > 1
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
    
    for e = 1:numel(ephysPath)
        % Get syncData
        %%% Recheck what's its name. Also check with Flora how that thing
        %%% is going to be computed and where it's going to be saved.
        syncDataFile = dir(fullfile(ephysPath{e},'*sync*'));
        
        if isempty(syncDataFile)
            warning('Couldn''t find the sync file for %s, %s. Skipping.', subject, expDate)
            ephysFlipperTimes{e} = [];
        else
            % Load sync data
            syncData = load(fullfile(syncDataFile.folder,syncDataFile.name));
            
            % Extract flips
            dataFile = dir(fullfile(ephysPath{e},'*ap.bin'));
            if exist(dataFile,'file')
                metaS = readMetaData_spikeGLX(dataFile.name,dataFile.folder);
                Fs = metaS.sRateHz;
            else
                warning('Couldn''t find metadata for %s, %s. Defining sampling rate as 30kHz.', subject, expDate)
                Fs = 30000;
            end
            tmp = abs(diff(syncData));
            ephysFlipperTimes{e} = find(tmp>=median(tmp(tmp>0)))/Fs; % there can be sometimes spiky noise that creates problems here 
        end
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
    
    for e = 1:numel(ephysPath)
        ephT = ephysFlipperTimes{e};
        
        % Check that number of flipper flips in timeline matches ephys
        success = 0;
        
        numFlipsDiff = abs(diff([length(ephT) length(timelineFlipperTimes)]));
        if numFlipsDiff>0 && numFlipsDiff<20
            fprintf([subject ' ' expDate ': WARNING = Flipper flip times different in timeline/ephys \n']);
            
            if diff([length(ephT) length(timelineFlipperTimes)])<20 && length(ephT) > 500
                fprintf([subject ' ' expDate ': Trying to account for missing flips.../ephys \n']);
                
                while length(timelineFlipperTimes) > length(ephT)
                    compareVect = [ephT-(ephT(1)) timelineFlipperTimes(1:length(ephT))-timelineFlipperTimes(1)];
                    errPoint = find(abs(diff(diff(compareVect,[],2)))>toleranceThreshold,1);
                    timelineFlipperTimes(errPoint+2) = [];
                    ephT(errPoint-2:errPoint+2) = [];
                    timelineFlipperTimes(errPoint-2:errPoint+2) = [];
                end
                while length(ephT) < length(timelineFlipperTimes)
                    compareVect = [timelineFlipperTimes-(timelineFlipperTimes(1)) ephT(1:length(timelineFlipperTimes))-ephT(1)];
                    errPoint = find(abs(diff(diff(compareVect,[],2)))>toleranceThreshold,1);
                    ephT(errPoint+2) = [];
                    ephT(errPoint-2:errPoint+2) = [];
                    timelineFlipperTimes(errPoint-2:errPoint+2) = [];
                end
                compareVect = [ephT-(ephT(1)) timelineFlipperTimes-timelineFlipperTimes(1)];
                if isempty(find(abs(diff(diff(compareVect,[],2)))>toleranceThreshold,1)); fprintf('Success! \n');
                    success = 1;
                end
            end
        elseif numFlipsDiff==0 
            success = 1;
        end
        
        if success
            ephysRefTimes{e} = ephT;
            timelineRefTimes{e} = timelineFlipperTimes;
        else
            ephysRefTimes{e} = [];
            timelineRefTimes{e} = [];
        end
    end
    
    %% Select only the ones that were matched
    ephysPath(cellfun(@(x) isempty(x),timelineRefTimes)) = [];
    ephysRefTimes(cellfun(@(x) isempty(x),timelineRefTimes)) = [];
    timelineRefTimes(cellfun(@(x) isempty(x),timelineRefTimes)) = [];
  