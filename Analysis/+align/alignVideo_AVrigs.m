function alignVideo_AVrigs(mouseName, thisDate, expNum, movieName, varargin)
    
    %%
    % Need to make it much more general, including:
    % - file names
    % - video types
    % - etc.
    
    %%
    timelineExpNums = expNum;
    tlSyncName = 'camSync';
    recompute = false;
    nFramesToLoad = 3000;
    if ~isempty(varargin)
        params = varargin{1};
        if isfield(params, 'recompute')
            recompute = params.recompute;
        end
        if isfield(params, 'nFramesToLoad')
            nFramesToLoad = params.nFramesToLoad;
        end
    end
    
    %%
    % should have a way to generalize (dat.expFilePath should be helpful,
    % but isn't working with the current split in servers...
    % Should insist a bit more though)
    server = '\\znas.cortexlab.net\Subjects\';
    if ~exist(fullfile(server,mouseName, thisDate, num2str(expNum)),'dir')
        server = '\\128.40.224.65\Subjects\';
    end
    movieDir = fullfile(server, mouseName, thisDate, num2str(expNum));
    intensFile = fullfile(server, mouseName, thisDate, num2str(expNum), ...
        [movieName '_avgIntensity.mat']);
    % have to deal with the "lastFrames" file. Pretty annoying.
    intensFile_lastFrames = fullfile(server, mouseName, thisDate, num2str(expNum), ...
        [movieName '_lastFrames_avgIntensity.mat']);
    
    if recompute && exist(intensFile, 'file')
        delete(intensFile);
    end
    if recompute && exist(intensFile_lastFrames, 'file')
        delete(intensFile_lastFrames);
    end
    
    if ~exist(intensFile, 'file')
        fprintf(1, 'computing average intensity of first/last frames...\n');
        ROI = avgMovieIntensity(movieDir, movieName, [], true, [], [], nFramesToLoad);
    end
    if ~exist(intensFile_lastFrames, 'file')
        ROI_lastFrames = avgMovieIntensity(movieDir, [movieName '_lastFrames'], [], true, [], [], []);
    end
    
    fprintf(1, 'loading avg intensity\n');
    load(intensFile);
    lf = load(intensFile_lastFrames);
    avgIntensity = [avgIntensity lf.avgIntensity];
    
    %% first detect the pulses in the avgIntensity trace
    
    expectedNumSyncs = numel(timelineExpNums)*2; % one at the beginning and end of each timeline file
    
    vidIntensThresh = [15 20];
    % [intensTimes, intensUp, intensDown] = schmittTimes(1:numel(avgIntensity), avgIntensity, vidIntensThresh);
    intensDown = find(diff(avgIntensity)>10); intensDown(find(diff(intensDown)<2)+1) = [];
    attemptNum = 1; loadAttemptNum = 1;
    while(numel(intensDown)~=expectedNumSyncs)
        % try some different approaches to get the right threshold
        % automatically...
        switch attemptNum
            case 1
                vidIntensThresh = min(avgIntensity)*[1.2 1.4];
            case 2
                intensMed = median(avgIntensity);
                intensMin = min(avgIntensity);
                vidIntensThresh = intensMin+(intensMed-intensMin)*[0.4 0.6];
            case 3
                vidIntensThresh = intensMin+(intensMed-intensMin)*[0.15 0.25];
            otherwise
                switch loadAttemptNum
                    case 1
                        fprintf(1, 'trying to load more frames...\n')
                        avgMovieIntensity(movieDir, movieName, [], true, ROI, [], 10000);
                        load(intensFile); avgIntensity = [avgIntensity lf.avgIntensity];
                        attemptNum = 0;
                    case 2
                        fprintf(1, 'trying to load all frames...\n')
                        avgMovieIntensity(movieDir, movieName, [], true, ROI);
                        load(intensFile); avgIntensity = [avgIntensity lf.avgIntensity];
                        attemptNum = 0;
                    otherwise
                        fprintf(1, 'cannot find a threshold that works. You tell me...\n');
                        figure; plot(avgIntensity);
                        keyboard
                end
                loadAttemptNum = loadAttemptNum+1;
        end
        
        [intensTimes, intensUp, intensDown] = schmittTimes(1:numel(avgIntensity), avgIntensity, vidIntensThresh);
        attemptNum = attemptNum +1;
    end
    assert(numel(intensDown)==expectedNumSyncs, 'could not find correct number of syncs');
    fprintf(1, 'found the sync pulses in the video\n');
    
    %% now get the timings
    % usually these are TTL and pretty much anything between 0 adn 5 will work
    % but here I use a small value because for the whisker camera I didn't have
    % TTL so this works for that too.
    % tlStrobeThresh = [0.08 0.15];
    tlStrobeThresh = [1 2];
    tlSyncThresh = [2 3];
    
    fprintf(1, 'loading timeline\n');
    load(fullfile(server, mouseName, thisDate, num2str(expNum), [thisDate '_' num2str(expNum) '_' mouseName '_Timeline.mat']));
    
    % find the timeline samples where cam sync pulses started (went
    % from 0 to 5V)
    tt = Timeline.rawDAQTimestamps;
    syncIndex = find(strcmp({Timeline.hw.inputs.name}, tlSyncName));
    tlSync = Timeline.rawDAQData(:,syncIndex);
    [~, tlSyncOnSamps, ~] = schmittTimes(1:numel(tlSync), tlSync, tlSyncThresh);
    
    vidSyncOnFrames = intensDown;
    
    A = importdata(fullfile(server, mouseName, thisDate, num2str(expNum), [movieName, '_times.txt']),'\t');
    timeFoundBetweenSyncs = A.data(vidSyncOnFrames(2),end)-A.data(vidSyncOnFrames(1),end);
    theoTimeBetweenSyncs = diff(tt(tlSyncOnSamps));
    
    timeDiscr = theoTimeBetweenSyncs-timeFoundBetweenSyncs;
    
    % time discrepancy could be due to a difference in timing (i.e., linear
    % scaling of time). Would be great to know.
    fprintf(1, 'time discrepancy was %ds \n', timeDiscr);
    
    % get the actual timestamps for the video in question.
    % take the metadata
    a = (tt(tlSyncOnSamps(2)) - tt(tlSyncOnSamps(1)))/(A.data(vidSyncOnFrames(2),end)-A.data(vidSyncOnFrames(1),end));
    b = tt(tlSyncOnSamps(1)) - a*A.data(vidSyncOnFrames(1),end);
    tVid = a*A.data(:,end) + b;
    vidFs = mean(diff(A.data(:,end)));
    
    saveName = fullfile(server, mouseName, thisDate, num2str(expNum), ...
        [movieName '_timeStamps.mat']);
    fprintf(1, 'saving to %s\n', saveName)
    
    save(saveName, 'tVid', 'vidFs');
end