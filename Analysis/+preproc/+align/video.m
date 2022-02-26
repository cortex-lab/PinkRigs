 function [tVid,numFramesMissed] = video(expPath, movieName, varargin)
    %%% This function will align the time frames of the input video to the
    %%% corresponding timeline. It will try to minimize the amount of time
    %%% and computing by loading first only the beginning and end of the
    %%% file. It will then iterate until it finds both flashes.
    %%% Additional arguments are some parameters, and the experiment's
    %%% timeline.
    %%%
    %%% This code is inspired by the code from kilotrode
    %%% (https://github.com/cortex-lab/kilotrodeRig).
    
    %% get path and parameters    
    % Parameters for processing (can be inputs in varargin{1})
    params.recomputeInt = false; % will recompute intensity file if true
    params.nFramesToLoad = 3000; % will start loading the first and 3000 of the movie
    params.adjustPercExpo = 1; % will adjust the timing of the first frame from its intensity
    params.plt = 1; % to plot the inter frame interval for sanity checks
    params.crashMissedFrames = 1; % will crash if any missed frame
    
    if ~isempty(varargin)
        paramsIn = varargin{1};
        params = parseInputParams(params,paramsIn);
        
        if numel(varargin) > 1
            timeline = varargin{2};
        end
    end
    
    %% Get files names
    
    % File with the movie intensity to detect the dark flashes
    intensFile = fullfile(expPath, ...
        [movieName '_avgIntensity.mat']);
    
    % File containing the last frames (due to vBox)
    intensFile_lastFrames = fullfile(expPath, ...
        [movieName '_lastFrames_avgIntensity.mat']);
    
    %% Load or (re)compute aligned times
    %% Get intensity files
    % This bit will compute and save the intensity of the movies.
    
    if params.recomputeInt
        % Delete intensity files
        if exist(intensFile, 'file')
            delete(intensFile);
        end
        if exist(intensFile_lastFrames, 'file')
            delete(intensFile_lastFrames);
        end
    end
    
    % Compute intensity file for the main file, will save it in folder
    if ~exist(intensFile, 'file')
        fprintf(1, 'computing average intensity of first/last frames...\n');
        vidproc.getAvgMovInt(expPath, movieName, params.nFramesToLoad);
    end
    
    % Compute intensity file for the lastFrames file, will save it in folder
    d = dir(fullfile(expPath, [movieName '_lastFrames.mj2'])); % to check if it's there and worth loading
    if ~isempty(d) && ~exist(intensFile_lastFrames, 'file')
        if d.bytes>100
            vidproc.getAvgMovInt(expPath, [movieName '_lastFrames'], []);
        end
    end
    
    % Load the average intensity
    fprintf(1, 'loading avg intensity\n');
    load(intensFile,'avgIntensity');
    
    % Load the lastFrames average intensity
    if ~isempty(d) && d.bytes>100
        lf = load(intensFile_lastFrames,'avgIntensity');
    else
        lf.avgIntensity = [];
    end
    if ~all(lf.avgIntensity == 0)
        % Happens when video is blank
        avgIntensity = [avgIntensity lf.avgIntensity];
    end
    
    %% Get the two dark flashes
    % This part will go through various methods to get the proper
    % thresholds. It's been working quite well.
    
    expectedNumSyncs = 2; % one at the beginning and end of each timeline file
    
    vidIntensThresh = [15 20];
    [~, ~, intensDown] = schmittTimes(1:numel(avgIntensity), avgIntensity, vidIntensThresh);
    
    attemptNum = 1; loadAttemptNum = 1;
    while(numel(intensDown)~=expectedNumSyncs)
        % Try some different approaches to get the right threshold
        % automatically...
        switch attemptNum
            case 1
                % vidIntensThresh = min(avgIntensity)*[1.2 1.4]; % if min
                % really is too small, can detect little wiggles
                intensMed = median(avgIntensity);
                intensMin = min(avgIntensity);
                vidIntensThresh = intensMin+(intensMed-intensMin)*[0.05 0.1];
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
                        vidproc.getAvgMovInt(expPath, movieName, 10000);
                        load(intensFile,'avgIntensity'); avgIntensity = [avgIntensity lf.avgIntensity];
                        attemptNum = 0;
                    case 2
                        fprintf(1, 'trying to load all frames...\n')
                        vidproc.getAvgMovInt(expPath, movieName);
                        load(intensFile,'avgIntensity'); avgIntensity = [avgIntensity lf.avgIntensity];
                        attemptNum = 0;
                    otherwise
                        fprintf(1, 'cannot find a threshold that works. You tell me...\n');
                        figure; plot(avgIntensity);
                        keyboard
                end
                loadAttemptNum = loadAttemptNum+1;
        end
        
        [~, ~, intensDown] = schmittTimes(1:numel(avgIntensity), avgIntensity, vidIntensThresh);
        attemptNum = attemptNum +1;
    end
    assert(numel(intensDown)==expectedNumSyncs, 'could not find correct number of syncs');
    fprintf(1, 'found the sync pulses in the video\n');
    
    vidSyncOnFrames = intensDown;
    
    %% Now get the timings
    % This part will find the timing from both the movie and timeline, and
    % align them.
    
    %% IN THE MOVIE
    % Real number of frames between sync
    numFramesFoundBetweenSyncs = diff(vidSyncOnFrames);
    
    % Get the frames times as saved by vBox
    A = importdata(fullfile(expPath, [movieName, '_times.txt']),'\t');
    
    % Inter frame interval
    % Note that a lot of jitter is introduced by matlab here
    IFI = diff(A.data(vidSyncOnFrames(1):vidSyncOnFrames(2),end));
    
    % Missed frames detection: will rely exclusively on Matlab times
    % could do differently for z4, but kept here will work on all rigs
    
    % First try to see of any IFI is bigger than expected
    % supposes there's not a majority of lost frames :)
    largeIFI = find(IFI>1.4*median(IFI));
    
    % Then check that have been compensated for in the next frame
    largeIFI_corrected = IFI(largeIFI((IFI(largeIFI)-median(IFI)+sum(IFI(largeIFI+1:2))-median(IFI) > 0.9*median(IFI))));
    numFramesMissed = largeIFI_corrected/median(IFI); % maybe won't be exactly that number??
    if isempty(numFramesMissed)
        numFramesMissed = 0;
    end
    
    if numFramesMissed && params.crashMissedFrames
        % Then error the whole thing to make sure you don't miss it
        error('missed frames: %d \n', numFramesMissed)
    else
        fprintf(1, 'missed frames: %d \n', numFramesMissed);
    end
    
    if params.plt
        % Plot and save a figure with the inter frame interval, for any
        % post-processing checks
        f = figure('visible','off'); hold all
        plot(IFI)
        axis tight
        plot([1 numel(IFI)], [median(IFI) median(IFI)],'k--')
        plot([1 numel(IFI)], 2*[median(IFI) median(IFI)],'k--')
        ylabel('Tnter-frame interval')
        xlabel('Frame')
        title(sprintf('Missed frames: %s',num2str(numFramesMissed)))
        
        % save
        saveas(f,fullfile(expPath, [movieName '_alignment.png']),'png')
    end
    
    %% IN TIMELINE
    % Load timeline if not an input
    if ~exist('Timeline','var')
        fprintf(1, 'Loading timeline\n');
        timeline = getTimeline(expPath);
    end
    tlTime = timeproc.extractChan(timeline,'time');
    
    % Find the timeline samples where cam sync pulses started
    tlSyncOnSamps = timeproc.getChanEventTime(timeline,'camSync');
    
    % If exist, find the strobe times for the camera
    camName = regexp(movieName,'[a-z]*Cam','match');
    strobeName = [camName{1} 'Strobe'];
    strobeSamps = timeproc.getChanEventTime(timeline,strobeName);
    if ~isempty(strobeSamps)
        % Take the strobes if exist
        numStrobesFoundBetweenSyncs = sum(strobeSamps>=tlSyncOnSamps(1) & strobeSamps<tlSyncOnSamps(2));
        numMissedFrames_wStrobes = numStrobesFoundBetweenSyncs - numFramesFoundBetweenSyncs;
        fprintf(1, 'missed frames with the strobes: %d \n', numMissedFrames_wStrobes);
    else
        numMissedFrames_wStrobes = nan;
    end
    
    if numFramesMissed && params.plt
        % Check which ones have been lost to further understand the issue
        % missedidx = find(diff(A.data(vidSyncOnFrames(1):vidSyncOnFrames(2),3))>1) + vidSyncOnFrames(1)-1;
        missedidx = largeIFI;
        figure;
        subplot(121)
        hold all
        tlSync = timeproc.extractChan(timeline,'camSync');
        plot(tlTime,tlSync)
        vline(tlSyncOnSamps)
        if ~isempty(strobeSamps)
            tlStrobe = timeproc.extractChan(timeline,strobeName);
            plot(tlTime,tlStrobe)
            vline(strobeSamps(find(strobeSamps>=tlSyncOnSamps(2),1)))
            vline(strobeSamps(find(strobeSamps<tlSyncOnSamps(2),1)))
        end
        
        subplot(122)
        hold all
        plot(avgIntensity)
        vline(intensDown)
        vline(numel(avgIntensity)-numel(lf.avgIntensity),'k--')
        vline(missedidx,'g--')
        fprintf(1, 'on the disk: %d frames / metadata %d frames \n',numel(avgIntensity),A.data(end,1)) % not sure what it means if these two things are different...
        
        % Check video around missed frames to see if we can see it
        vid = VideoReader(fullfile(expPath,[movieName '.mj2']));
        win = [missedidx-20,missedidx+20];
        tmp = read(vid,win);
        imtool3D(squeeze(tmp))
    end
    
    %% Align both
    % Get the actual timestamps for the video in question.
    % Try to realign a bit better with the percentage of exposition.
    % Could maybe use the strobes if they're here?
    
    if params.adjustPercExpo
        % Will adjust the first post-dark flash frame depending on its
        % intensity compared to the previous ones.
        percentExpo = (avgIntensity(vidSyncOnFrames(1))-avgIntensity(vidSyncOnFrames(1)-2))/(avgIntensity(vidSyncOnFrames(1)+2)-avgIntensity(vidSyncOnFrames(1)-2));
    else
        percentExpo = 0;
    end
    
    % Get offset and compression coefficients.
    vidFs = mean(diff(A.data(vidSyncOnFrames(1):vidSyncOnFrames(2),end))); % computed empirically...
    a = (tlSyncOnSamps(2) - tlSyncOnSamps(1))/(A.data(vidSyncOnFrames(2),end)-A.data(vidSyncOnFrames(1),end));
    b = tlSyncOnSamps(1) - a*(A.data(vidSyncOnFrames(1),end)) + percentExpo*vidFs;
    
    % Here I cannot use matlab's timing as they have a lot of 'fake'
    % jitter. So I just recompute the times. Note that first and last
    % frames should have the same timing.
    tVid = a*(A.data(vidSyncOnFrames(1),end) + ((1:size(A.data,1)) - vidSyncOnFrames(1))*vidFs) + b; % vidSyncOnFrames(1) is the one that has been properly aligned, so should be this one that is used?    
    
    %% Back up of other things we can compute to check... But doesn't work very well
    %     timeFoundBetweenSyncs = A.data(vidSyncOnFrames(2),end)-A.data(vidSyncOnFrames(1),end);
    %     theoTimeBetweenSyncs = diff(tt(tlSyncOnSamps));
    %     timeDiscr = theoTimeBetweenSyncs-timeFoundBetweenSyncs;
    %     % time discrepancy could be due to a difference in timing (i.e., linear
    %     % scaling of time). Would be great to know???
    %     fprintf(1, 'time discrepancy was %ds \n', timeDiscr);
    %
    %     % theoretical number of frames between sync (Frame count)
    %     % supposes that if there's a missed frame, that number (A.data(:,3))
    %     % will suddenly increase by more than 1
    %     numTheoFramesFoundBetweenSyncs_Count = A.data(vidSyncOnFrames(2),3)-A.data(vidSyncOnFrames(1),3);
    %
    %     % theoretical number of frames between sync (based on Frame rate)
    %     % pretty hard to compute!!! Because:
    %     % - if there's a lost frame, will lower the computed frame rate so we
    %     % won't see it
    %     % - if take median gives a large number of dropped frames, even when
    %     % there aren't...
    %     numTheoFramesFoundBetweenSyncs_Rate = timeFoundBetweenSyncs/mean(diff(A.data(:,end)));
end
