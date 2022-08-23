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
        if strcmp(recompute, 'all') && exist(savePath,'file')
           delete(savePath) 
        end
        
        % Get align status
        alignFields = contains(expInfo.Properties.VariableNames, 'align');
        notAlignedYet = struct();
        for i = expInfo.Properties.VariableNames(alignFields)
            notAlignedYet.(lower(i{1}(6:end))) = expInfo.(i{1});
        end

        %If there is no timeline. All alignment is NaN
        if strcmp(expInfo.existTimeline, '0')
            if ~all(structfun(@(x) strcmpi(x,'nan'), notAlignedYet))
                [block, ephys] = deal(nan);
                save(savePath,'block', 'ephys');
                csv.updateRecord('subject', subject, ...
                    'expDate', expDate,...
                    'expNum', expNum);
            end
            fprintf(1, '*** WARNING: Skipping alignment as no timeline: %s... ***\n', expFolder);
            continue;
        end
        notAlignedYet = structfun(@(x) contains(x,'0'), notAlignedYet,'uni',0);
        notAlignedYet.video = 0;

        % Anonymous function to decide whether something should be processed
        shouldProcess = @(x) (...
            any(contains(recompute,{'all';x}, 'IgnoreCase',true))...
            || notAlignedYet.(lower(x)))...
            && contains(process,{'all';x});

        if ~(contains('none', recompute) && ~any(structfun(@(x)all(x==1), notAlignedYet))) % If good already
            %% If all isn't good...
                        
            % Monitors if anything has changed
            change = 0;
            
            % Loads timeline once
            expInfo = csv.loadData(expInfo, 'dataType','timeline');
            
            fprintf(1, '*** Aligning experiment %s... ***\n', expFolder);
            
            %% Align spike times to timeline and save results in experiment folder
            %  This function will load the timeline flipper from the experiment and
            %  check this against all ephys files recorded on the same date. If it
            %  locates a matching section of ephys data, it will save the reference
            %  flipper times for both the ephys and the timeline.
            %  It will output two time series, and one can use these time series to
            %  compute the events times in timeline time from times in block time using
            %  "event2timeline".
            
            if shouldProcess('ephys')
                try
                    % Align it
                    fprintf(1, '* Aligning ephys... *\n');
                    [ephysFlipperTimes, timelineFlipperTimes, ephysPath, serialNumber] = preproc.align.ephys(expInfo);
                    fprintf(1, '* Ephys alignment done. *\n');
                    
                    % Save it
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
                    msgText = getReport(me);
                    warning('Couldn''t align ephys: threw an error (%s)',msgText)
                    ephys = 'error';
                    
                    % Save error message locally
                    saveErrMess(msgText,fullfile(expFolder, 'AlignEphysError.json'))
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
            
            if shouldProcess('block')
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
                    msgText = getReport(me);
                    warning('Couldn''t align block: threw an error (%s)', msgText)
                    block = 'error';
                    
                    % Save error message locally
                    saveErrMess(msgText,fullfile(expFolder, 'AlignBlockError.json'))
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
            
            if any(cellfun(@(x)shouldProcess(x), [videoNames; 'video']))                               
                fprintf(1, '* Aligning videos... *\n');
                
                vids2Process = videoNames(cellfun(@(x) shouldProcess(x), videoNames));

                % Align each of them
                for v = 1:numel(vids2Process)

                    vidName = vids2Process{v};
                    expInfo.vidName = vidName;
                    expInfo.vidInfo{1} = dir(fullfile(expFolder,['*' vidName '.mj2']));
                    
                    videoONEFolder = fullfile(expFolder,'ONE_preproc',vidName);
                    initONEFolder(videoONEFolder,'times')

                    fprintf(1, 'Aligning video %s... \n',vidName);
                    try
                        [frameTimes, missedFrames,nFirstFrames] = preproc.align.video(expInfo);
                        
                        if missedFrames > 0
                           error('Missed frames: %d. Recheck.', missedFrames) 
                        end
                        
                        stub = [expDate '_' expNum '_' subject '_' vidName];
                        saveONEFormat(frameTimes(1:nFirstFrames)',videoONEFolder,'camera','times','npy',stub);
                        
                        % Remove any error file
                        if exist(fullfile(videoONEFolder, sprintf('AlignVideoError_%s.json',vidName)),'file')
                            delete(fullfile(videoONEFolder, sprintf('AlignVideoError_%s.json',vidName)))
                        end
                    catch me
                        msgText = getReport(me);
                        warning('Could not align video %s: threw an error (%s)',vidName,msgText)
                        
                        % Save error message locally
                        saveErrMess(msgText,fullfile(videoONEFolder, sprintf('AlignVideoError_%s.json',vidName)))
                    end
                end
                fprintf(1, '* Video alignment done. *\n');
                                
                change = 1;
            end
            
            %% Align microphone to timeline
            %  This function will take the output of the 192kHz microphone and align it
            %  to the low frequency microphone that records directly into the timeline
            %  channel. Saved as a 1Hz version of the envelope of both.
            
            if shouldProcess('mic')
                
                micONEFolder = fullfile(expFolder,'ONE_preproc','mic');
                initONEFolder(micONEFolder)
                
                % Align it
                try
                    fprintf(1, '* Aligning mic... *\n');
                    %%% TODO -- ERRORS FOR NOW
                    error('Have not found or coded a way to align file yet.') % for now
                    fprintf(1, '* Mic alignment done. *\n');
                    
                    % Remove any error file
                    if exist(fullfile(micONEFolder, 'AlignMicError.json'),'file')
                        delete(fullfile(micONEFolder, 'AlignMicError.json'))
                    end
                catch me
                    msgText = getReport(me);
                    warning('Could not align mic: threw an error (%s)',msgText)
                    
                    % Save error message locally
                    saveErrMess(msgText,fullfile(micONEFolder, 'AlignMicError.json'))
                end
                
                change = 1;
            end
            
            %% Update the csv
            
            if change
                csv.updateRecord('subject', subject, 'expDate', expDate, 'expNum', expNum);
            end
        end
    end
        
        
