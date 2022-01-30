function [] = alignVideo_AVrigs(mouseName, thisDate, expNum, cam, varargin)
    
    %%
    % Need to make it much more general, including:
    % - file names
    % - video types
    % - etc.
    
    %%
    movieName = [thisDate '_' num2str(expNum) '_' mouseName '_' cam];
    
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
        server = '\\zinu.cortexlab.net\Subjects\';
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
    d = dir(fullfile(movieDir, [movieName '_lastFrames.mj2']));
    if ~exist(intensFile_lastFrames, 'file')
        if d.bytes>100
            ROI_lastFrames = avgMovieIntensity(movieDir, [movieName '_lastFrames'], [], true, [], [], []);  
        end
    end
    
    fprintf(1, 'loading avg intensity\n');
    load(intensFile);
    if d.bytes>100
        lf = load(intensFile_lastFrames);
    else
        lf.avgIntensity = [];
    end
    if ~all(lf.avgIntensity == 0)
        avgIntensity = [avgIntensity lf.avgIntensity];
    end
    
    %% first detect the pulses in the avgIntensity trace
    
    expectedNumSyncs = numel(timelineExpNums)*2; % one at the beginning and end of each timeline file
    
    vidIntensThresh = [15 20];
    [intensTimes, intensUp, intensDown] = schmittTimes(1:numel(avgIntensity), avgIntensity, vidIntensThresh);
    % intensDown = find(diff(avgIntensity)>10); intensDown(find(diff(intensDown)<2)+1) = []; %% that's intensUp...
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
                        avgMovieIntensity(movieDir, movieName, [], true, [], [], 10000);
                        load(intensFile); avgIntensity = [avgIntensity lf.avgIntensity];
                        attemptNum = 0;
                    case 2
                        fprintf(1, 'trying to load all frames...\n')
                        avgMovieIntensity(movieDir, movieName, [], true, []);
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
    strobeName = [cam 'Strobe'];
    tlStrobeThresh = [1 2];
    tlSyncThresh = [2 3];
    
    fprintf(1, 'loading timeline\n');
    load(fullfile(server, mouseName, thisDate, num2str(expNum), [thisDate '_' num2str(expNum) '_' mouseName '_Timeline.mat']));
    
    % find the timeline samples where cam sync pulses started (went
    % from 0 to 5V)
    tt = Timeline.rawDAQTimestamps;
    syncIndex = find(strcmp({Timeline.hw.inputs.name}, tlSyncName));
    tlSync = Timeline.rawDAQData(:,syncIndex);
    [~, ~, tlSyncOnSamps] = schmittTimes(1:numel(tlSync), tlSync, tlSyncThresh);
    
    vidSyncOnFrames = intensDown;
    
    A = importdata(fullfile(server, mouseName, thisDate, num2str(expNum), [movieName, '_times.txt']),'\t');
    timeFoundBetweenSyncs = A.data(vidSyncOnFrames(2),end)-A.data(vidSyncOnFrames(1),end);
    theoTimeBetweenSyncs = diff(tt(tlSyncOnSamps));
    timeDiscr = theoTimeBetweenSyncs-timeFoundBetweenSyncs;
    % time discrepancy could be due to a difference in timing (i.e., linear
    % scaling of time). Would be great to know???
    fprintf(1, 'time discrepancy was %ds \n', timeDiscr);
        
    % real number of frames between sync
    numFramesFoundBetweenSyncs = diff(vidSyncOnFrames);
    
    % theoretical number of frames between sync (Frame count)
    % supposes that if there's a missed frame, that number (A.data(:,3))
    % will suddenly increase by more than 1 
    numTheoFramesFoundBetweenSyncs_Count = A.data(vidSyncOnFrames(2),3)-A.data(vidSyncOnFrames(1),3);
    
    % theoretical number of frames between sync (based on Frame rate)
    % pretty hard to compute!!! Because:
    % - if there's a lost frame, will lower the computed frame rate so we
    % won't see it
    % - if take median gives a large number of dropped frames, even when
    % there aren't...
    numTheoFramesFoundBetweenSyncs_Rate = timeFoundBetweenSyncs/mean(diff(A.data(:,end)));
    
    IFI = diff(A.data(vidSyncOnFrames(1):vidSyncOnFrames(2),end));
    
    % find the strobe times for the camera
    strobeIndex = find(strcmp({Timeline.hw.inputs.name}, strobeName));
    if ~isempty(strobeIndex)
        tlStrobe = Timeline.rawDAQData(:,strobeIndex);
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
        
        %     figure;
        %     scatter(IFI(largeIFI)-median(IFI),IFI(largeIFI+1)-median(IFI));
    end
    
    f = figure('visible','off'); hold all
    plot(IFI)
    axis tight
    plot([1 numel(IFI)], [median(diff(A.data(:,end))) median(diff(A.data(:,end)))],'k--')
    plot([1 numel(IFI)], 2*[median(diff(A.data(:,end))) median(diff(A.data(:,end)))],'k--')
    ylabel('inter-frame interval')
    xlabel('frame')
    title(sprintf('Missed frames: %s',num2str(framesMissed)))
    saveas(f,fullfile(server, mouseName, thisDate, num2str(expNum), [movieName '_alignment.png']));

        
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
        vid = VideoReader(fullfile(movieDir,[movieName '.mj2']));
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
    
    saveName = fullfile(server, mouseName, thisDate, num2str(expNum), ...
        [movieName '_timeStamps.mat']);
    fprintf(1, 'saving to %s\n', saveName)
    
    save(saveName, 'tVid', 'vidFs');
end
