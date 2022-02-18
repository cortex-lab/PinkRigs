function main(varargin)
    %%% This function will run the main alignment code, and save the
    %%% results in the expPath folder.
    %%% Inputs can be a set of parameters (input 1) and/or a list of
    %%% experiments (input 2), given by their paths.
    
    %% Get parameters and list of mice to check
    % Parameters for processing (can be inputs in varargin{1})
    recompute = {'none'};
    paramsVid = []; % will take default
    
    % This is not ideal
    if ~isempty(varargin)
        params = varargin{1};
        
        if ~isempty(params) && isfield(params, 'recompute')
            recompute = params.recompute;
        end
        if ~isempty(params) && isfield(params, 'paramsVid')
            paramsVid = params.paramsVid;
        end
        
        if numel(varargin) > 1
            if istable(varargin{2})
                % already in the right format, with all the info
                exp2checkList = varargin{2};
            else
                % format is just a cell with paths, go fetch info
                expPath2checkList = varargin{2};
                exp2checkList = getExpInfoFromPath(expPath2checkList);
            end
        end
    end
    
    if ~exist('exp2checkList', 'var')
        % Will get all the exp for the active mice.
        exp2checkList = getAllExp2Check();
    end
    
    %% --------------------------------------------------------
    %% Will compute the 'alignment' file for each experiment.
    
    for ee = 1:size(exp2checkList,1)
        
        % Can also amend the csv to say whether this one has been
        % aligned or not.
        
        % Get exp info
        expInfo = exp2checkList(ee,:);
        expPath = expInfo.path{1};
        
        %% Get the path of the alignment file and fetch it if exists
        % Define savepath for the alignment results
        savePath = fullfile(expPath,'alignment.mat');
        
        % Load it if exists
        if exist(savePath,'file')
            load(savePath,'alignment');
            alignmentOld = alignment; clear alignment
        else
            alignmentOld = struct();
        end
        
        % monitors if anything has changed
        change = 0;
        
        %% Align spike times to timeline and save results in experiment folder
        %  This function will load the timeline flipper from the experiment and
        %  check this against all ephys files recorded on the same date. If it
        %  locates a matching section of ephys data, it will save the reference
        %  flipper times for both the ephys and the timeline.
        %  It will output two time series, and one can use these time series to
        %  compute the events times in timeline time from times in block time using
        %  "event2timeline".
        
        if contains(recompute,'all') || contains(recompute,'ephys') || ~isfield(alignmentOld,'ephys')
            if expInfo.ephys
                % Align it
                [ephysFlipperTimes, timelineFlipperTimes, ephysPath] = preproc.align.ephys(expPath);
                
                % Save it
                for p = 1:numel(ephysPath)
                    alignment.ephys(p).originTimes = ephysFlipperTimes{p};
                    alignment.ephys(p).timelineTimes = timelineFlipperTimes{p};
                    alignment.ephys(p).ephysPath = ephysPath{p}; % can have several probes
                end
            else
                alignment.ephys = [];
            end
            change = 1;
        else
            % Just load it
            alignment.ephys = alignmentOld.ephys;
        end
        
        %% Align the block timings to timeline
        %  This function will load the timeline and block for that experiment and
        %  align one with another using 1) the wheel or 2) the photodiode.
        %  It will output two time series, and one can use these time series to
        %  compute the events times in timeline time from times in block time using
        %  "event2timeline".
        
        if contains(recompute,'all') || contains(recompute,'block') || ~isfield(alignmentOld,'block')
            [blockRefTimes, timelineRefTimes] = preproc.align.block(expPath);
            
            % save it
            alignment.block.originTimes = blockRefTimes;
            alignment.block.timelineTimes = timelineRefTimes;
            change = 1;
        else
            % Just load it
            alignment.block = alignmentOld.block;
        end
        
        %% Align the video frame times to timeline
        %  This function will align all cameras' frame times with the experiment's
        %  timeline.
        %  The resulting times for these alignments will be saved in a structure
        %  'vids' that contains all cameras.
        
        if contains(recompute,'all') || contains(recompute,'video') || ~isfield(alignmentOld,'video')
            
            % Get cameras' names
            vids = dir(fullfile(expPath,'*Cam.mj2')); % there should be 3: side, front, eye
            f = fieldnames(vids);
            vids = rmfield(vids,f(~ismember(f,'name')));
            
            % Align each of them
            for v = 1:numel(vids)
                [~,vidName,~] = fileparts(vids(v).name);
                try
                    [vids(v).frameTimes, vids(v).missedFrames] = preproc.align.video(expPath, vidName, paramsVid);
                catch me
                    % case when it's corrupted
                    vids(v).frameTimes = [];
                    vids(v).missedFrames = [];
                    warning('Corrupted video %s: threw an error (%s)',vidName,me.message)
                end
            end
            
            % save it
            alignment.video = vids;
            change = 1;
        else
            % Just load it
            alignment.video = alignmentOld.video;
        end
        
        %% Align microphone to timeline
        %  This function will take the output of the 192kHz microphone and align it
        %  to the low frequency microphone that records directly into the timeline
        %  channel. Saved as a 1Hz version of the envelope of both.
        
        if contains(recompute,'all') || contains(recompute,'mic') || ~isfield(alignmentOld,'mic')
            % Align it
            if expInfo.micDat > 0
                %%% TODO
            else
                alignment.mic = nan;
            end
            change = 1;
        else
            % Just load it
            alignment.mic = alignmentOld.mic;
        end
        
        %% Save the alignment structure
        
        if change
            save(savePath,'alignment')
        else
            % Could save over it anyway even if nothing has changed,
            % but I felt like keeping the date of last modifications of
            % the files might be useful (e.g., for debugging).
        end
        
        %% Amend the csv 
        % to say that alignment has been computed, and which
        % ephys corresponds to that experiment.
        
        %%% TO DO
    end

    
