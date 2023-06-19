function movie = getVideoAtTime(expPath,t,vidName,win)
    %% Extract a chunk of video at time 't', centered on window 'win'
    %
    % Parameters:
    % -------------------
    % expPath: str
    %   Path of the experiment
    % t: double
    %   Time in timeline time
    % vidName: str
    %   Which video to look at
    % win: vector
    %   Window to look at
    %
    % Returns: 
    % -------------------
    % movie: array
    %   Associated movie
    
    %% Convert time into video time
    alignmentFile = dir(fullfile(expPath,'*alignment.mat'));
    
    if isempty(alignmentFile)
        error('No alignment file for this experiment.')
    else
        % Load alignment file
        alignment = load(fullfile(alignmentFile.folder,alignmentFile.name),'video');
        if isempty(alignment.video)
            error('No video')
        end
    end
    
    goodVidIdx = find(~strcmp({alignment.video.missedFrames},'error'));
    if isempty(goodVidIdx)
        error('All videos are corrupted.')
    end
    
    vidIdx = strcmp({alignment.video.name},vidName);
    if isempty(vidIdx) || strcmp({alignment.video(vidIdx).missedFrames},'error')
        vidName = alignment.video(goodVidIdx(1)).name;
        fprintf('Couldn''t find what you ask (inexistant or corrupted). Loading video %s.\n', vidName)
        vidIdx = strcmp({alignment.video.name},vidName);
    end
        
    timeWin = find(alignment.video(vidIdx).frameTimes > t+win(1) & ... 
        alignment.video(vidIdx).frameTimes < t+win(2));
    if isempty(timeWin)
        error('Time point outside of what has been recorded.')
    else
        timeWin = [timeWin(1) timeWin(end)];
    end
    
    %% Get video file 
    [subject,expDate,expNum] = parseExpPath(expPath);
    vidFiles = dir(fullfile(expPath,[expDate '_' expNum  '_' subject '_*Cam.mj2']));
    if isempty(vidFiles)
        error('No video for this experiment.')
    else
        if exist('vidName','var')
            vidIdx = contains({vidFiles.name},vidName);
            vidFile = fullfile(vidFiles(vidIdx).folder,vidFiles(vidIdx).name);
            if ~exist(vidFile,'file')
                error('This video doesn''t exist.')
            end
        end
    end
    try
        vid = VideoReader(vidFile);
    catch
        error('This video is corrupted.')
    end

    %% Read specific window
    
    movie = read(vid,timeWin);
    
    
   