function main(varargin)
    %%% This function will run the main alignment code, and save the
    %%% results in the expFolder.
    %%% Inputs can be a set of parameters (input 1) and/or a list of
    %%% experiments (input 2), given by their paths.
    
    %% Get parameters and list of mice to check
    % Parameters for processing (can be inputs in varargin)
    varargin = ['recompute', 'none', varargin];
    varargin = ['videoNames', {{{'frontCam';'sideCam';'eyeCam'}}}, varargin];
    varargin = ['process', 'all', varargin];
    params = csv.inputValidation(varargin{:});
    exp2checkList = csv.queryExp(params);    
    %% --------------------------------------------------------

    %% Will compute the 'alignment' file for each experiment.
    for ee = 1:size(exp2checkList,1)
        % Get expInfo for current experiment (passed to sub functions)
        expInfo = exp2checkList(ee,:);

        % Assign variables from exp2checkList to ease of use later
        expDate = exp2checkList.expDate{ee, 1};
        expNum = exp2checkList.expNum{ee, 1};
        subject = exp2checkList.subject{ee, 1};
        expFolder = exp2checkList.expFolder{ee, 1};
        recompute = exp2checkList.recompute{ee, 1};
        process = exp2checkList.process{ee, 1};
        videoNames = exp2checkList.videoNames{ee, 1};
        
        % Define savepath for the alignment results
        pathStub = fullfile(expFolder, [expDate '_' expNum '_' subject]);
        savePath = [pathStub '_alignment.mat'];
        if exist(savePath,'file')
            % To check if anything's missing (and that the csv hasn't seen
            % for some reason)
            varListInFile = who('-file', savePath);
        else
            varListInFile = {};
        end
        
        % Get preproc status
        alignStatus = csv.parseStatusCode(expInfo.alignBlkFrontSideEyeMicEphys);
        
        %If there is no timeline. All alignment is NaN
        if strcmp(expInfo.timeline, '0')
            if ~all(structfun(@(x) strcmpi(x,'nan'), alignStatus))
                [block, mic, ephys] = deal(nan);
                video = struct('name', expInfo.videoNames, ...
                    'frameTimes', num2cell(nan*ones(3,1)),...
                    'missedFrames', num2cell(nan*ones(3,1)));
                save(savePath,'block', 'video', 'ephys', 'mic');
                csv.updateRecord('subject', subject, ...
                    'expDate', expDate,...
                    'expNum', expNum);
            end
            fprintf(1, '*** WARNING: Skipping alignment as no timeline: %s... ***\n', expFolder);
            continue;
        end
        alignStatus = structfun(@(x) strcmp(x,'0'), alignStatus,'uni',0);
        alignStatus.video = 0;

        %Anonymous funciton to decide whether something should be processed
        shouldProcess = @(x,y) (...
            any(contains(recompute,{'all';x}, 'IgnoreCase',true))...
            || alignStatus.(x)...
            || ~ismember(y, varListInFile))...
            && contains(process,{'all';x});

        if ~(contains('none', recompute) && strcmp(expInfo.alignBlkFrontSideEyeMicEphys,'1,1,1,1,1,1')) % If good already
            %% If all isn't good...
                        
            % monitors if anything has changed
            change = 0;
            
            fprintf(1, '*** Aligning experiment %s... ***\n', expFolder);
            
            %% Align spike times to timeline and save results in experiment folder
            %  This function will load the timeline flipper from the experiment and
            %  check this against all ephys files recorded on the same date. If it
            %  locates a matching section of ephys data, it will save the reference
            %  flipper times for both the ephys and the timeline.
            %  It will output two time series, and one can use these time series to
            %  compute the events times in timeline time from times in block time using
            %  "event2timeline".
            
            if shouldProcess('ephys', 'ephys')
                
                ephysFolder = fullfile(fileparts(expFolder),'ephys');
                if exist(ephysFolder,'dir')
                    try
                        % Align it
                        fprintf(1, '* Aligning ephys... *\n');
                        [ephysFlipperTimes, timelineFlipperTimes, ephysPath, serialNumber] = preproc.align.ephys(expInfo);
                        fprintf(1, '* Ephys alignment done. *\n');
                        
                        % Save it
                        % Found a (set of) matching ephys for that exp.
                        ephys = struct();
                        for p = 1:numel(ephysPath) % can have several probes
                            ephys(p).originTimes = ephysFlipperTimes{p};
                            ephys(p).timelineTimes = timelineFlipperTimes{p};
                            ephys(p).ephysPath = ephysPath{p}; 
                            ephys(p).serialNumber = serialNumber(p); 
                        end
                        
                        % Remove any error file
                        if exist(fullfile(expFolder, 'AlignEphysError.json'),'file')
                            delete(fullfile(expFolder, 'AlignEphysError.json'))
                        end
                    catch me
                        warning('Couldn''t align ephys: threw an error (%s)',me.message)
                        ephys = 'error';
                        
                        % Save error message locally
                        saveErrMess(me.message,fullfile(expFolder, 'AlignEphysError.json'))
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
            
            if shouldProcess('block', 'block')

                % Note that block file should always exist.
                try
                    block = struct;
                    fprintf(1, '* Aligning block... *\n');
                    if contains(expInfo.expDef, 'spontaneousActivity')
                        % expDefs that aren't expected to have alignement
                        block = nan;
                    else
                        [blockRefTimes, timelineRefTimes] = preproc.align.block(expInfo);
                        block.originTimes = blockRefTimes;
                        block.timelineTimes = timelineRefTimes;
                    end
                    fprintf(1, '* Block alignment done. *\n');
                    
                    % Remove any error file
                    if exist(fullfile(expFolder, 'AlignBlockError.json'),'file')
                        delete(fullfile(expFolder, 'AlignBlockError.json'))
                    end
                catch me
                    warning('Couldn''t align block: threw an error (%s)', me.message)
                    block = 'error';
                    
                    % Save error message locally
                    saveErrMess(me.message,fullfile(expFolder, 'AlignBlockError.json'))
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
            
            if any(cellfun(@(x)shouldProcess(x, 'video'), [videoNames; 'video']))                               
                fprintf(1, '* Aligning videos... *\n');
                
                if contains('video', varListInFile)
                    video = load(savePath, 'video');
                    video = video.video;
                else
                    video = struct();
                end

                if ~isempty(video) && ~any(contains({'video'; 'all'}, recompute))
                    vids2Process = videoNames(contains(recompute, videoNames,  'IgnoreCase', 1));
                else, vids2Process = videoNames;
                end

                % Align each of them
                for v = 1:numel(vids2Process)
                    vidName = vids2Process{v};
                    expInfo.vidName = vidName;
                    expInfo.vidInfo{1} = dir(fullfile(expFolder,['*' vidName '.mj2']));
                    video(v).name = vidName;
                    if ~isempty(expInfo.vidInfo)
                        try
                            [video(v).frameTimes, video(v).missedFrames] = preproc.align.video(expInfo);
                            
                            % Remove any error file
                            if exist(fullfile(expFolder, sprintf('AlignVideoError_%s.json',vidName)),'file')
                                delete(fullfile(expFolder, sprintf('AlignVideoError_%s.json',vidName)))
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
                            saveErrMess(me.message,fullfile(expFolder, sprintf('AlignVideoError_%s.json',vidName)))
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
            
            if shouldProcess('mic', 'mic')

                % Align it
                if str2double(expInfo.micDat) > 0
                    try
                        fprintf(1, '* Aligning mic... *\n');
                        %%% TODO
                        error('Haven''t found or coded a way to align file yet.') % for now
                        fprintf(1, '* Mic alignment done. *\n');
                        
                        % Remove any error file
                        if exist(fullfile(expFolder, 'AlignMicError.json'),'file')
                            delete(fullfile(expFolder, 'AlignMicError.json'))
                        end
                    catch me
                        warning('Couldn''t align mic: threw an error (%s)',me.message)
                        mic = 'error';
                        
                        % Save error message locally
                        saveErrMess(me.message,fullfile(expFolder, 'AlignMicError.json'))
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
                csv.updateRecord('subject', subject, 'expDate', expDate, 'expNum', expNum);
            end
        end
    end
        
        
