function [] = video_AVrigs(subject, expDate, expNum, movieName, varargin)
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
    % get experiment's path 
    expPath = getExpPath(subject, expDate, expNum);
    
    % parameters for processing
    recomputeInt = false; % will recompute intensity file if true
    nFramesToLoad = 3000; % will start loading the first and 3000 of the movie
    adjustFirstFrame = 1; % will adjust the timing of the first frame from its intensity
    
    if ~isempty(varargin)
        params = varargin{1};
        
        % to recompute the intensity files
        if isfield(params, 'recomputeInt')
            recomputeInt = params.recomputeInt;
        end
        if isfield(params, 'nFramesToLoad')
            nFramesToLoad = params.nFramesToLoad;
        end
        if isfield(params, 'adjustFirstFrame')
            adjustFirstFrame = params.adjustFirstFrame;
        end
        
        if numel(varargin)>1
            Timeline = varargin{2};
        end
    end
    
    %% get files names
    
    % file in which to save the new timestamps
    saveName = fullfile(expPath, ...
        [movieName '_timeStamps.mat']);
    
    % file with the movie intensity to detect the dark flashes
    intensFile = fullfile(expPath, ...
        [movieName '_avgIntensity.mat']);
    
    % file containing the last frames (due to vBox)
    intensFile_lastFrames = fullfile(expPath, ...
        [movieName '_lastFrames_avgIntensity.mat']);
    
    %% get intensity files
    % This bit will compute and save the intensity of the movies.
    
    if recomputeInt
        % delete intensity files
        if exist(intensFile, 'file')
            delete(intensFile);
        end
        if exist(intensFile_lastFrames, 'file')
            delete(intensFile_lastFrames);
        end
    end
    
    % compute intensity file for the main file, will save it in folder
    if ~exist(intensFile, 'file')
        fprintf(1, 'computing average intensity of first/last frames...\n');
        vidpro.getAvgMovInt(expPath, movieName, nFramesToLoad);
    end
    
    % compute intensity file for the lastFrames file, will save it in folder
    d = dir(fullfile(expPath, [movieName '_lastFrames.mj2'])); % to check if it's there and worth loading
    if ~exist(intensFile_lastFrames, 'file')
        if d.bytes>100
            vidpro.getAvgMovInt(expPath, [movieName '_lastFrames'], []);  
        end
    end
    
    % load the average intensity
    fprintf(1, 'loading avg intensity\n');
    load(intensFile);
    
    % load the lastFrames average intensity
    if d.bytes>100
        lf = load(intensFile_lastFrames);
    else
        lf.avgIntensity = [];
    end
    if ~all(lf.avgIntensity == 0)
        % happens when video is blank
        avgIntensity = [avgIntensity lf.avgIntensity];
    end
    
    %% get the two dark flashes
    % This part will go through various methods to get the proper
    % thresholds. It's been working quite well.
    
    expectedNumSyncs = 2; % one at the beginning and end of each timeline file
    
    vidIntensThresh = [15 20];
    [~, ~, intensDown] = schmittTimes(1:numel(avgIntensity), avgIntensity, vidIntensThresh);

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
                        vidpro.getAvgMovInt(expPath, movieName, 10000);
                        load(intensFile); avgIntensity = [avgIntensity lf.avgIntensity];
                        attemptNum = 0;
                    case 2
                        fprintf(1, 'trying to load all frames...\n')
                        vidpro.getAvgMovInt(expPath, movieName);
                        load(intensFile); avgIntensity = [avgIntensity lf.avgIntensity];
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
    
    %% now get the timings
    
    % load timeline if not an input
    if ~exist('Timeline','var')
        fprintf(1, 'loading timeline\n');
        Timeline = timepro.getTimeline(subject,expDate,expNum);
    end
    tt = timepro.extractChan(Timeline,'time');
    
    % find the timeline samples where cam sync pulses started
    tlSync = timepro.extractChan(Timeline,'camSync');
    tlSyncThresh = [2 3];
    [~, ~, tlSyncOnSamps] = schmittTimes(1:numel(tlSync), tlSync, tlSyncThresh);
    
    % get the frames times as saved by vBox
    A = importdata(fullfile(server, subject, expDate, num2str(expNum), [movieName, '_times.txt']),'\t');
        
    % real number of frames between sync
    numFramesFoundBetweenSyncs = diff(vidSyncOnFrames);
    
    IFI = diff(A.data(vidSyncOnFrames(1):vidSyncOnFrames(2),end));
    
    % find the strobe times for the camera
    % usually these are TTL and pretty much anything between 0 and 5 will work
    % but here I use a small value because for the whisker camera I didn't have
    % TTL so this works for that too.
    % tlStrobeThresh = [0.08 0.15];
    camName = regexp(vidName,'[a-z]*Cam','match');
    strobeName = [camName{1} 'Strobe'];
    tlStrobe = timepro.extractChan(Timeline,strobeName);  
    

    if ~isempty(tlStrobe)
        % take the strobes if exist
        tlStrobeThresh = [1 2];
        [~,strobeSamps,~] = schmittTimes(1:numel(tlStrobe), tlStrobe, tlStrobeThresh);
        numStrobesFoundBetweenSyncs = sum(strobeSamps>=tlSyncOnSamps(1) & strobeSamps<tlSyncOnSamps(2));
        framesMissed = numStrobesFoundBetweenSyncs-numFramesFoundBetweenSyncs;
    else
        % framesMissed = numFramesFoundBetweenSyncs - numTheoFramesFoundBetweenSyncs_Count;
        % maybe do it differently: try to see of any IFI is bigger than
        % expected
        largeIFI = find(IFI>1.4*median(IFI)); % supposes there's not a majority of lost frames :)
        % check that have been compensated for in the next frame
        framesMissed = IFI(largeIFI(find((IFI(largeIFI)-median(IFI)+sum(IFI(largeIFI+1:2))-median(IFI) > 0.9*median(IFI)))))/median(IFI); % maybe won't be exactly that number??
        if isempty(framesMissed)
            framesMissed = 0;
        end
    end
    
    f = figure('visible','off'); hold all
    plot(IFI)
    axis tight
    plot([1 numel(IFI)], [median(diff(A.data(:,end))) median(diff(A.data(:,end)))],'k--')
    plot([1 numel(IFI)], 2*[median(diff(A.data(:,end))) median(diff(A.data(:,end)))],'k--')
    ylabel('inter-frame interval')
    xlabel('frame')
    title(sprintf('Missed frames: %s',num2str(framesMissed)))
    saveas(f,fullfile(server, subject, expDate, num2str(expNum), [movieName '_alignment.png'],'png')

        
    fprintf(1, 'missed frames: %d \n', framesMissed);
    if framesMissed
        % check if we can find them
        % seems to be registered in the frame count of the metadata
        
        % check which ones have been lost
        missedidx = find(diff(A.data(vidSyncOnFrames(1):vidSyncOnFrames(2),3))>1) + vidSyncOnFrames(1)-1;
        figure;
        plot(tlStrobe)
        hold all
        plot(tlSync)
        vline(strobeSamps(find(strobeSamps>=tlSyncOnSamps(2),1)))
        vline(strobeSamps(find(strobeSamps<tlSyncOnSamps(2),1)))
        
        figure;
        plot(avgIntensity)
        hold all
        vline(intensDown)
        vline(numel(avgIntensity)-numel(lf.avgIntensity),'k--')
        vline(missedidx,'g--')
        fprintf(1, 'on the disk: %d frames / metadata %d frames \n',numel(avgIntensity),A.data(end,1)) % not sure what it means if these two things are different...
        
        figure; hold all
        plot(tlSync)
        vline(tlSyncOnSamps)
        
        % check video around missed frames to see if we can see it
        vid = VideoReader(fullfile(expPath,[movieName '.mj2']));
        win = [missedidx-20,missedidx+20];
        tmp = read(vid,win);
        imtool3D(squeeze(tmp))
        
        % then error the whole thing to make sure you don't miss it
        error(sprintf('missed frames: %d \n', framesMissed))
    end

    % get the actual timestamps for the video in question.
    % take the metadata
    % try to realign a bit better
    percentExpo = (avgIntensity(vidSyncOnFrames(1))-avgIntensity(vidSyncOnFrames(1)-2))/(avgIntensity(vidSyncOnFrames(1)+2)-avgIntensity(vidSyncOnFrames(1)-2));
    
    vidFs = mean(diff(A.data(:,end)));
    a = (tt(tlSyncOnSamps(2)) - tt(tlSyncOnSamps(1)))/(A.data(vidSyncOnFrames(2),end)-A.data(vidSyncOnFrames(1),end));
    b = tt(tlSyncOnSamps(1)) - a*(A.data(vidSyncOnFrames(1),end)) + percentExpo*vidFs;
    % tVid = a*A.data(:,end) + b; % some jitter is introduced by matlab here...
    tVid = a*(A.data(vidSyncOnFrames(1),end) + ((1:size(A.data,1)) - vidSyncOnFrames(1))*vidFs) + b; % vidSyncOnFrames(1) is the one that has been properly aligned, so should be this one that is used?
    
    fprintf(1, 'saving to %s\n', saveName)
    
    save(saveName, 'tVid', 'vidFs');
    
    
        
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
