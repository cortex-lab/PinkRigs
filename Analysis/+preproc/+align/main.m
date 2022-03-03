function main(varargin)
    %%% This function will run the main alignment code, and save the
    %%% results in the expPath folder.
    %%% Inputs can be a set of parameters (input 1) and/or a list of
    %%% experiments (input 2), given by their paths.
    
    %% Get parameters and list of mice to check
    % Parameters for processing (can be inputs in varargin{1})
    params.recompute = {'none'};
    params.paramsVid.videoNames = {'frontCam','sideCam','eyeCam'}; % will take default for the rest

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
        
        % Get exp info
        expInfo = exp2checkList(ee,:);
        expPath = expInfo.expFolder{1};
        
        % Define savepath for the alignment results
        [subject, expDate, expNum] = parseExpPath(expPath);
        savePath = fullfile(expPath,[expDate '_' expNum '_' subject '_alignment.mat']);
        if exist(savePath,'file')
            % To check if anything's missing (and that the csv hasn't seen
            % for some reason)
            varListInFile = who('-file', savePath);
        else
            varListInFile = {};
        end
        
        % Get preproc status
        alignStatus = parseStatusCode(expInfo.alignBlkFrontSideEyeMicEphys{1});
        
        if ~(strcmp(params.recompute,'none') && strcmp(expInfo.alignBlkFrontSideEyeMicEphys{1},'1,1,1,1,1,1')) % If good already
            %% If all isn't good...
                        
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
            
            if contains(params.recompute,'all') || contains(params.recompute,'ephys') || ...
                    strcmp(alignStatus.ephys,'0') || ~ismember('ephys',varListInFile)
                
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
                            ephys = nan;
                        else
                            % Found a (set of) matching ephys for that exp.
                            ephys = struct();
                            for p = 1:numel(ephysPath)
                                ephys(p).originTimes = ephysFlipperTimes{p};
                                ephys(p).timelineTimes = timelineFlipperTimes{p};
                                ephys(p).ephysPath = ephysPath{p}; % can have several probes
                            end
                        end
                        
                        % Remove any error file
                        if exist(fullfile(expPath, 'AlignEphysError.json'),'file')
                            delete(fullfile(expPath, 'AlignEphysError.json'))
                        end
                    catch me
                        warning('Couldn''t align ephys: threw an error (%s)',me.message)
                        ephys = 'error';
                        
                        % Save error message locally
                        saveErrMess(me.message,fullfile(expPath, 'AlignEphysError.json'))
                    end
                else
                    % Case where the ephys fodler did not exist. It's either
                    % because it's not supposed to exist, or wasn't copied.
                    ephys = nan;
                end
                                
                change = 1;
                
                % Save it
                if exist(savePath,'file')
                    save(savePath,'ephys','-append')
                else
                    save(savePath,'ephys')
                end
            end
            
            %% Align the block timings to timeline
            %  This function will load the timeline and block for that experiment and
            %  align one with another using 1) the wheel or 2) the photodiode.
            %  It will output two time series, and one can use these time series to
            %  compute the events times in timeline time from times in block time using
            %  "event2timeline".
            
            if contains(params.recompute,'all') || contains(params.recompute,'block') || ...
                    strcmp(alignStatus.block,'0') || ~ismember('block',varListInFile)
                
                % Note that block file should always exist.
                try
                    fprintf(1, '* Aligning block... *\n');
                    [blockRefTimes, timelineRefTimes] = preproc.align.block(expPath);
                    fprintf(1, '* Block alignment done. *\n');
                    
                    % Save it
                    block.originTimes = blockRefTimes;
                    block.timelineTimes = timelineRefTimes;
                    
                    % Remove any error file
                    if exist(fullfile(expPath, 'AlignBlockError.json'),'file')
                        delete(fullfile(expPath, 'AlignBlockError.json'))
                    end
                catch me
                    warning('Couldn''t align block: threw an error (%s)',me.message)
                    block = 'error';
                    
                    % Save error message locally
                    saveErrMess(me.message,fullfile(expPath, 'AlignBlockError.json'))
                end
                                
                change = 1;
                
                % Save it
                if exist(savePath,'file')
                    save(savePath,'block','-append')
                else
                    save(savePath,'block')
                end
            end
            
            %% Align the video frame times to timeline
            %  This function will align all cameras' frame times with the experiment's
            %  timeline.
            %  The resulting times for these alignments will be saved in a structure
            %  'vids' that contains all cameras.
            
            if contains(params.recompute,'all') || contains(params.recompute,'video') || ...
                    strcmp(alignStatus.frontCam,'0') || strcmp(alignStatus.eyeCam,'0') || strcmp(alignStatus.sideCam,'0') || ...
                    ~ismember('video',varListInFile) %  Won't check for every cam here
                
                fprintf(1, '* Aligning videos... *\n');
                
                % Align each of them
                video = struct();
                for v = 1:numel(params.paramsVid.videoNames)
                    vidName = params.paramsVid.videoNames{v};
                    d = dir(fullfile(expPath,['*' vidName '.mj2']));
                    video(v).name = vidName;
                    if ~isempty(d)
                        try
                            [video(v).frameTimes, video(v).missedFrames] = preproc.align.video(expPath, d.name(1:end-4), params.paramsVid);
                            
                            % Remove any error file
                            if exist(fullfile(expPath, sprintf('AlignVideoError_%s.json',vidName)),'file')
                                delete(fullfile(expPath, sprintf('AlignVideoError_%s.json',vidName)))
                            end
                        catch me
                            warning('Couldn''t align video %s: threw an error (%s)',vidName,me.message)
                            
                            if strcmp(me.message,'Failed to initialize internal resources.')
                                % Very likely that video is corrupted. Make it a
                                % nan because there's not much we can do for now.
                                video(v).frameTimes = nan;
                                video(v).missedFrames = nan;
                            else
                                % Another error occured. Save it.
                                video(v).frameTimes = 'error';
                                video(v).missedFrames = 'error';
                            end
                            
                            % Save error message locally
                            saveErrMess(me.message,fullfile(expPath, sprintf('AlignVideoError_%s.json',vidName)))
                        end
                    else
                        % Couldn't find the file.
                        video(v).frameTimes = nan;
                        video(v).missedFrames = nan;
                    end
                end
                fprintf(1, '* Video alignment done. *\n');
                                
                change = 1;
                
                % Save it
                if exist(savePath,'file')
                    save(savePath,'video','-append')
                else
                    save(savePath,'video')
                end
            end
            
            %% Align microphone to timeline
            %  This function will take the output of the 192kHz microphone and align it
            %  to the low frequency microphone that records directly into the timeline
            %  channel. Saved as a 1Hz version of the envelope of both.
            
            if contains(params.recompute,'all') || contains(params.recompute,'mic') || ...
                    strcmp(alignStatus.mic,'0') || ~ismember('mic',varListInFile)
                
                % Align it
                if str2double(expInfo.micDat{1}) > 0
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
                        mic = 'error';
                        
                        % Save error message locally
                        saveErrMess(me.message,fullfile(expPath, 'AlignMicError.json'))
                    end
                else
                    % Mic data wasn't there.
                    mic = nan;
                end
                
                change = 1;
                
                % Save it
                if exist(savePath,'file')
                    save(savePath,'mic','-append')
                else
                    save(savePath,'mic')
                end
            end
            
            %% Update the csv
            
            if change
                csv.updateRecord(subject, expDate, expNum);
            end
        end
    end
        
        
