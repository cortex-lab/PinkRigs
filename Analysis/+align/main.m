function main(p,varargin)
    %%% This function will run the main alignment code, and save the
    %%% results in the expPath folder.
    
    %% Get missing parameters and list of mice to check
    if isfield(p, 'recompute')
        p.recompute = {'none'};
    end
    
    if nargin > 1
        mouse2checkList = varargin{1};
    else
        % Get mouse list from main csv
        mainCSVLoc = getCSVLocation('main');
        mouseList = readtable(mainCSVLoc);
        mouse2checkList = mouseList.Subject;
    end
    
    %% Check which experiments aren't aligned
    for m = 1:numel(mouse2checkList)
        % Loop through subjects
        subject = mouse2checkList{m};
        
        % Get list of exp for this mouse
        expList = getMouseExpList(subject);
        
        for e = 1:numel(expList)
            % Loop through csv to look for experiments that weren't
            % aligned, or all if p.recompute isn't none.
            % Can also amend the csv to say whether this one has been
            % aligned or not.
            
            % Get exp info
            expInfo = expList(e,:);
            expPath = expInfo.path;
            
            % Define savepath
            savePath = fullfile(expPath,'alignment.mat');
            
            % Load it if exists
            if exist(savePath,'file')
                load(savePath,'alignment');
            else
                alignmentOld = struct();
            end
            
            if ~(strcmp(p.recompute,'none') && isfield(alignmentOld,'ephys') && ...
                    isfield(alignmentOld,'block') && ...
                    isfield(alignmentOld,'video') && ...
                    isfield(alignmentOld,'mic'))
                % Go through it only if has something missing and don't
                % want to recompute
                
                %% Align spike times to timeline and save results in experiment folder
                %  This function will load the timeline flipper from the experiment and
                %  check this against all ephys files recorded on the same date. If it
                %  locates a matching section of ephys data, it will extract the
                %  corresponding spike data and save it into the experiment folder. This
                %  function will not run if there is no ephys/kilosort data
                
                if contains(p.recompute,'all') || contains(p.recompute,'ephys') || ~isfield(alignmentOld,'ephys')
                    % align it
                else
                    % just load it
                    alignment.ephys = alignmentOld.ephys;
                end
                
                %% Align the block timings to timeline
                %  This function will load the timeline and block for that experiment and
                %  align one with another using 1) the wheel or 2) the photodiode.
                %  It will output two time series, and one can use these time series to
                %  compute the events times in timeline time from times in block time using
                %  "block2tl".
                
                if contains(p.recompute,'all') || contains(p.recompute,'block') || ~isfield(alignmentOld,'block')
                    [blRefTimes, tlRefTimes] = align.block_AVrigs(expPath);
                else
                    % just load it
                    alignment.block = alignmentOld.block;
                end
                
                % save it
                alignment.block.blRefTimes = blRefTimes;
                alignment.block.tlRefTimes = tlRefTimes;
                
                %% Align the video frame times to timeline
                %  This function will align all cameras' frame times with the experiment's
                %  timeline.
                %  The resulting times for these alignments will be saved in a structure
                %  'vids' that contains all cameras.
                
                if contains(p.recompute,'all') || contains(p.recompute,'video') || ~isfield(alignmentOld,'video')
                    
                    % Define a few parameters (optional) -- should maybe live somewhere else?
                    pVid.recomputeInt = false; % will recompute intensity file if true
                    pVid.nFramesToLoad = 3000; % will start loading the first and 3000 of the movie
                    pVid.adjustPercExpo = 1; % will adjust the timing of the first frame from its intensity
                    pVid.plt = 1; % to plot the inter frame interval for sanity checks
                    pVid.crashMissedFrames = 1; % will crash if any missed frame
                    
                    % Get cameras' names
                    vids = dir(fullfile(expPath,'*Cam.mj2')); % there should be 3: side, front, eye
                    
                    % Align each of them
                    for v = 1:numel(vids)
                        [~,vidName,~]=fileparts(vids(v).name);
                        try
                            [vids(v).frameTimes, vids(v).missedFrames] = align.video_AVrigs(expPath, vidName, pVid);
                        catch me
                            % case when it's corrupted
                            vids(v).frameTimes = [];
                            vids(v).missedFrames = [];
                            warning('Corrupted video %s: threw an error (%s)',vidName,me.message)
                        end
                    end
                else
                    % just load it
                    alignment.video = alignmentOld.video;
                end
                
                % save it
                alignment.video = vids;
                
                %% Align microphone to timeline
                %  This function will take the output of the 192kHz microphone and align it
                %  to the low frequency microphone that records directly into the timeline
                %  channel. Saved as a 1Hz version of the envelope of both.
                
                if contains(p.recompute,'all') || contains(p.recompute,'mic') || ~isfield(alignmentOld,'mic')
                    % align it
                else
                    % just load it
                    alignment.mic = alignmentOld.mic;
                end
                
                %% Save the alignment structure
                
                save(savePath,'alignment')
            end
        end
    end
    
