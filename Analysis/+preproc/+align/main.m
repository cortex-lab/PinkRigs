function main(varargin)
    %%% This function will run the main alignment code, and save the
    %%% results in the expPath folder.
    %%% Inputs can be a set of parameters (input 1) and/or a list of
    %%% experiments (input 2), given by their paths.
    
    %% Get parameters and list of mice to check
    % Parameters for processing (can be inputs in varargin{1})
    params.recompute = {'none'};
    params.paramsVid = []; % will take default
    
    % This is not ideal
    if ~isempty(varargin)
        paramsIn = varargin{1};
        params = parseInputParams(params,paramsIn);
        
        if numel(varargin) > 1
            if istable(varargin{2})
                % Already in the right format, with all the info
                exp2checkList = varargin{2};
            else
                % Format is just a cell with paths, go fetch info
                expPath2checkList = varargin{2};
                exp2checkList = getExpInfoFromPath(expPath2checkList);
            end
        end
    end
    
    if ~exist('exp2checkList', 'var')
        % Will get all the exp for the active mice.
        exp2checkList = queryExp();
    end
    
    %% --------------------------------------------------------
    %% Will compute the 'alignment' file for each experiment.
    
    for ee = 1:size(exp2checkList,1)
        
        % Can also amend the csv to say whether this one has been
        % aligned or not.
        
        % Get exp info
        expInfo = exp2checkList(ee,:);
        expPath = expInfo.expFolder{1};
        
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
        
        fprintf(1, '*** Aligning experiment %s... ***\n', expPath);
        
        %% Align spike times to timeline and save results in experiment folder
        %  This function will load the timeline flipper from the experiment and
        %  check this against all ephys files recorded on the same date. If it
        %  locates a matching section of ephys data, it will save the reference
        %  flipper times for both the ephys and the timeline.
        %  It will output two time series, and one can use these time series to
        %  compute the events times in timeline time from times in block time using
        %  "event2timeline".
        
        if contains(params.recompute,'all') || contains(params.recompute,'ephys') || ~isfield(alignmentOld,'ephys')
            ephysFolder = fullfile(fileparts(expPath),'ephys');
            if exist(ephysFolder,'dir')
                try
                    % Align it
                    fprintf(1, '* Aligning ephys... *\n');
                    [ephysFlipperTimes, timelineFlipperTimes, ephysPath] = preproc.align.ephys(expPath);
                    fprintf(1, '* Ephys alignment done. *\n');
                    
                    % Save it
                    if isempty(ephysPath)
                        % Couldn't find matching ephys for that experiment.
                        alignment.ephys = nan;
                    else
                        % Found a (set of) matching ephys for that exp.
                        for p = 1:numel(ephysPath)
                            alignment.ephys(p).originTimes = ephysFlipperTimes{p};
                            alignment.ephys(p).timelineTimes = timelineFlipperTimes{p};
                            alignment.ephys(p).ephysPath = ephysPath{p}; % can have several probes
                        end
                    end
                    
                    % Remove any error file
                    if exist(fullfile(expPath, 'AlignEphysError.json'),'file')
                        delete(fullfile(expPath, 'AlignEphysError.json'))
                    end
                catch me
                    warning('Couldn''t align ephys: threw an error (%s)',me.message)
                    alignment.ephys = 'error';
                    
                    % Save error message locally
                    saveErrMess(me.message,fullfile(expPath, 'AlignEphysError.json'))
                end
            else
                % Case where the ephys fodler did not exist. It's either
                % because it's not supposed to exist, or wasn't copied.
                alignment.ephys = nan;
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
        
        if contains(params.recompute,'all') || contains(params.recompute,'block') || ~isfield(alignmentOld,'block')
            % Note that block file should always exist.
            try
                fprintf(1, '* Aligning block... *\n');
                [blockRefTimes, timelineRefTimes] = preproc.align.block(expPath);
                fprintf(1, '* Block alignment done. *\n');
                
                % Save it
                alignment.block.originTimes = blockRefTimes;
                alignment.block.timelineTimes = timelineRefTimes;
                
                % Remove any error file
                if exist(fullfile(expPath, 'AlignBlockError.json'),'file')
                    delete(fullfile(expPath, 'AlignBlockError.json'))
                end
            catch me
                warning('Couldn''t align block: threw an error (%s)',me.message)
                alignment.block = 'error';
                
                % Save error message locally
                saveErrMess(me.message,fullfile(expPath, 'AlignBlockError.json'))
            end
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
        
        if contains(params.recompute,'all') || contains(params.recompute,'video') || ~isfield(alignmentOld,'video')
            fprintf(1, '* Aligning videos... *\n');
            % Get cameras' names
            vids = dir(fullfile(expPath,'*Cam.mj2')); % there should be 3: side, front, eye
            f = fieldnames(vids);
            vids = rmfield(vids,f(~ismember(f,'name')));
            
            % Align each of them
            for v = 1:numel(vids)
                [~,vidName,~] = fileparts(vids(v).name);
                try
                    [vids(v).frameTimes, vids(v).missedFrames] = preproc.align.video(expPath, vidName, params.paramsVid);
                    
                    % Remove any error file
                    if exist(fullfile(expPath, sprintf('AlignVideoError_%s.json',vidName)),'file')
                        delete(fullfile(expPath, sprintf('AlignVideoError_%s.json',vidName)))
                    end
                catch me
                    warning('Couldn''t align video %s: threw an error (%s)',vidName,me.message)
                    
                    if strcmp(me.message,'Failed to initialize internal resources.')
                        % Very likely that video is corrupted. Make it a
                        % nan because there's not much we can do for now.
                        vids(v).frameTimes = nan;
                        vids(v).missedFrames = nan;
                    else
                        % Another error occured. Save it.
                        vids(v).frameTimes = 'error';
                        vids(v).missedFrames = 'error';
                    end
                    
                    % Save error message locally
                    saveErrMess(me.message,fullfile(expPath, sprintf('AlignVideoError_%s.json',vidName)))
                end
            end
            fprintf(1, '* Video alignment done. *\n');
            
            % Save it
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
        
        if contains(params.recompute,'all') || contains(params.recompute,'mic') || ~isfield(alignmentOld,'mic')
            % Align it
            if expInfo.micDat > 0
                try                    
                    fprintf(1, '* Aligning mic... *\n');
                    %%% TODO
                    error('Haven''t found or coded a way to align file yet.') % for now
                    fprintf(1, '* Mic alignment done. *\n');
                    
                    % Remove any error file
                    if exist(fullfile(expPath, 'AlignMicError.json'),'file')
                        delete(fullfile(expPath, 'AlignMicError.json'))
                    end
                catch me
                    warning('Couldn''t align mic: threw an error (%s)',me.message)
                    alignment.mic = 'error';
                    
                    % Save error message locally
                    saveErrMess(me.message,fullfile(expPath, 'AlignMicError.json'))
                end
            else
                % Mic data wasn't there.
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
        
        %% Update the csv
        
        if change
            [subject, expDate, expNum] = parseExpPath(expPath);
            csv.updateRecord(subject, expDate, expNum)
        end
    end

    
