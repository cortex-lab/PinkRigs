function main(varargin)
    %%% This function will run the main alignment code, and save the
    %%% results in the expFolder.
    %%% Inputs can be a set of parameters (input 1) and/or a list of
    %%% experiments (input 2), given by their paths.
    
    %% Get parameters and list of mice to check
    % Parameters for processing (can be inputs in varargin{1})
    varargin = ['recompute', 'none', varargin];
    varargin = ['videoNames', {{{'frontCam';'sideCam';'eyeCam'}}}, varargin];
    varargin = ['process', 'all', varargin];
    params = csv.inputValidation(varargin{:});
    exp2checkList = csv.queryExp(params);    
    %% --------------------------------------------------------

    %% Will compute the 'alignment' file for each experiment.
    varargin = varargin(cellfun(@(x) ~istable(x), varargin));
    for ee = 1:size(exp2checkList,1)
        % Get exp info
        expInfo = csv.inputValidation(varargin{:}, exp2checkList(ee,:));
        expFolder = expInfo.expFolder{1};
        recompute = params.recompute{1};
        process = params.process{1};
        
        % Define savepath for the alignment results
        pathStub = fullfile(expFolder, [expInfo.expDate{1} '_' expInfo.expNum{1} '_' expInfo.subject{1}]);
        savePath = [pathStub '_alignment.mat'];
        if exist(savePath,'file')
            % To check if anything's missing (and that the csv hasn't seen
            % for some reason)
            varListInFile = who('-file', savePath);
        else
            varListInFile = {};
        end
        
        % Get preproc status
        alignStatus = csv.parseStatusCode(expInfo.alignBlkFrontSideEyeMicEphys{1});
        alignStatus = structfun(@(x) strcmp(x,'0'), alignStatus,'uni',0);
        
        %Anonymous funciton to decide whether something should be processed
        shouldProcess = @(x,y) (contains(recompute,{'all';x}) || alignStatus.(x)...
            || ~ismember(y, varListInFile)) && contains(process,{'all';x});

        if ~(strcmp(params.recompute{1},'none') && strcmp(expInfo.alignBlkFrontSideEyeMicEphys{1},'1,1,1,1,1,1')) % If good already
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
                        [ephysFlipperTimes, timelineFlipperTimes, ephysPath] = preproc.align.ephys(expInfo);
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
                    [blockRefTimes, timelineRefTimes] = preproc.align.block(expInfo);
                    fprintf(1, '* Block alignment done. *\n');
                    
                    % Save it
                    block.originTimes = blockRefTimes;
                    block.timelineTimes = timelineRefTimes;
                    
                    % Remove any error file
                    if exist(fullfile(expFolder, 'AlignBlockError.json'),'file')
                        delete(fullfile(expFolder, 'AlignBlockError.json'))
                    end
                catch me
                    warning('Couldn''t align block: threw an error (%s)',me.message)
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
            
            if any(cellfun(@(x)shouldProcess(x, 'video'), params.videoNames{1}))                               
                fprintf(1, '* Aligning videos... *\n');
                
                % Align each of them
                video = struct();
                for v = 1:numel(params.videoNames{1})
                    vidName = params.videoNames{1}{v};
                    expInfo.vidName{1} = vidName;
                    expInfo.vidInfo{1} = dir(fullfile(expFolder,['*' vidName '.mj2']));
                    video(v).name = vidName;
                    if ~isempty(expInfo.vidInfo{1})
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
                if str2double(expInfo.micDat{1}) > 0
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
                csv.updateRecord(expInfo.subject{1}, expInfo.expDate{1}, expInfo.expNum{1});
            end
        end
    end
        
        
