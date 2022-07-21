function extractExpData(varargin)
    %%% This function will extract all the important information from the
    %%% experiment's timeline and block, load the spikes, align everything,
    %%% and save a preprocessed version of the data in the exp folder.
    %%% First argument is a set of params, second a list of experiments,
    %%% either in a table or cell with paths format.
    
    %% Get parameters
    varargin = ['recompute', 'none', varargin];
    varargin = ['process', 'all', varargin];
    params = csv.inputValidation(varargin{:});
    exp2checkList = csv.queryExp(params);
    
    %% --------------------------------------------------------
    %% Will compute the 'preprocData' file for each experiment.   
    for ee = 1:size(exp2checkList,1)
        % Get exp info
        expInfo = exp2checkList(ee,:);

        % Assign variables from exp2checkList to ease of use later
        expDate = exp2checkList.expDate{ee,1};
        expNum = exp2checkList.expNum{ee,1};
        expDef = exp2checkList.expDef{ee,1};
        subject = exp2checkList.subject{ee,1};
        expFolder = exp2checkList.expFolder{ee,1};
        recompute = exp2checkList.recompute{ee,1};
        process = exp2checkList.process{ee,1};
        
        % Define savepath for the preproc results
        pathStub = fullfile(expFolder, [expDate '_' expNum '_' subject]);
        savePath = [pathStub '_preprocData.mat'];

        if exist(savePath,'file')
            % To check if anything's missing (and that the csv hasn't seen
            % for some reason)
            varListInFile = who('-file', savePath);
        else
            varListInFile = {};
        end
        
        % Get preproc status
        preprocStatus = csv.parseStatusCode(expInfo.preProcSpkEV);
        preprocStatus = structfun(@(x) strcmp(x,'0'), preprocStatus,'uni',0);

        %Anonymous function to decide whether something should be processed
        shouldProcess = @(x) (contains(recompute,{'all';x}) || preprocStatus.(x)...
            || ~ismember(x, varListInFile)) && contains(process,{'all';x});

        if ~(strcmp(recompute,'none') && strcmp(expInfo.preProcSpkEV,'1,1')) 
            %% If all isn't good...
                        
            % monitors if anything has changed
            change = 0;
            
            fprintf(1, '*** Preprocessing experiment %s... ***\n', expFolder);

            % get alignment file location
            alignmentFile = strrep(savePath, 'preprocData', 'alignment');

            if exist(alignmentFile, 'file')
                %% Load alignment file
                
                %% Extract important info from timeline or block
                % If need be, use preproc.align.event2timeline(eventTimes,alignment.block.originTimes,alignment.block.timelineTimes)
                
                if shouldProcess('ev')
                    alignment = load(alignmentFile, 'block');
                    
                    eventsONEFolder = fullfile(expFolder,'ONE_preproc','events');
                    if exist(eventsONEFolder,'dir')
                        rmdir(eventsONEFolder,'s')
                    end
                    mkdir(eventsONEFolder);
                    
                    try
                        fprintf(1, '* Extracting events... *\n');
                        
                        % Get Block and Timeline
                        loadedData = csv.loadData(expInfo, 'loadTag', 'timelineblock');
                        timeline = loadedData.timelineData{1};
                        block = loadedData.blockData{1};
                        
                        % Get the appropriate ref for the exp def
                        expDefRef = preproc.getExpDefRef(expDef);
                        
                        % Call specific preprocessing function
                        ev = preproc.expDef.(expDefRef)(timeline,block,alignment.block);
                        
                        % Remove any error file
                        if exist(fullfile(eventsONEFolder, 'GetEvError.json'),'file')
                            delete(fullfile(eventsONEFolder, 'GetEvError.json'))
                        end
                        
                        fprintf(1, '* Events extraction done. *\n');

                        stub = [expDate '_' expNum '_' subject];
                        saveONEFormat(ev,eventsONEFolder,'_av_trials','table','pqt',stub);
                        
                    catch me
                        msgText = getReport(me);
                        warning('%s \n Could not get events (ev): threw an error\n %s',me.identifier, msgText)
                        ev = 'error';
                        
                        % Save error message locally
                        saveErrMess(me.message,fullfile(eventsONEFolder, 'GetEvError.json'))
                    end
                    
                    change = 1;
                    
                    % Save it
                    if exist(savePath,'file')
                        save(savePath,'ev','-append')
                    else
                        save(savePath,'ev')
                    end
                end    
                    
                %% Extract spikes and clusters info (depth, etc.)
                
                if shouldProcess('spk')
                    alignment = load(alignmentFile,'ephys','block');
                    if isfield(alignment,'ephys')
                        if isstruct(alignment.ephys)
                            fprintf (1, '* Extracting spikes... *\n');
                            
                            if ~exist('block','var')
                                loadedData = csv.loadData(expInfo, 'loadTag', 'block');
                                block = loadedData.blockData{1};
                            end
                            
                            spk = cell(1,numel(alignment.ephys));
                            for probeNum = 1:numel(alignment.ephys)
                                
                                probeONEFolder = fullfile(expFolder,'ONE_preproc',sprintf('probe%d',probeNum-1));
                                if exist(probeONEFolder,'dir')
                                    % Need to do differently if only
                                    % reprocessing the q metrics?
                                    rmdir(probeONEFolder,'s')
                                end
                                mkdir(probeONEFolder);
                                
                                if ~isnan(alignment.ephys(probeNum).ephysPath)
                                    try
                                        % -----------
                                        % Keep this one for now
                                        % Get spikes times & cluster info
                                        spk{probeNum} = preproc.getSpikeData(alignment.ephys(probeNum).ephysPath);
                                        
                                        % Align them
                                        spk{probeNum}.spikes.time = preproc.align.event2Timeline(spk{probeNum}.spikes.time, ...
                                            alignment.ephys(probeNum).originTimes,alignment.ephys(probeNum).timelineTimes);
                                        
                                        % Subselect the ones that are within this experiment
                                        expLength = block.duration;
                                        spk2keep = (spk{probeNum}.spikes.time>0) & (spk{probeNum}.spikes.time<expLength);
                                        spk{probeNum}.spikes.time = spk{probeNum}.spikes.time(spk2keep);
                                        spk{probeNum}.spikes.cluster = spk{probeNum}.spikes.cluster(spk2keep);
                                        spk{probeNum}.spikes.xpos = spk{probeNum}.spikes.xpos(spk2keep);
                                        spk{probeNum}.spikes.depth = spk{probeNum}.spikes.depth(spk2keep);
                                        spk{probeNum}.spikes.tempScalingAmp = spk{probeNum}.spikes.tempScalingAmp(spk2keep);
                                        [~,shankID] = min(abs(spk{probeNum}.spikes.xpos - repmat([0 200 400 600], [numel(spk{probeNum}.spikes.xpos),1])),[],2);
                                        shankID = shankID-1;
                                        
                                        % Recompute spike numbers
                                        for c = 1:numel(spk{probeNum}.clusters)
                                            spk{probeNum}.clusters(c).Spknum = sum(spk{probeNum}.spikes.cluster == spk{probeNum}.clusters(c).ID);
                                        end
                                        
                                        % Get probe info
                                        spk{probeNum}.probe.serialNumber = alignment.ephys(probeNum).serialNumber;
                                        % -----------
                                        
                                        % -----------
                                        % Save IBL style -- 
                                        % Get the spike and cluster info
                                        spkONE{probeNum} = preproc.getSpikeDataONE(alignment.ephys(probeNum).ephysPath);
                                        
                                        % Align them
                                        spkONE{probeNum}.spikes.times = preproc.align.event2Timeline(spkONE{probeNum}.spikes.times, ...
                                            alignment.ephys(probeNum).originTimes,alignment.ephys(probeNum).timelineTimes);
                                        
                                        % Subselect the ones that are within this experiment
                                        expLength = block.duration;
                                        spk2keep = (spkONE{probeNum}.spikes.times>0) & (spkONE{probeNum}.spikes.times<expLength);
                                        spkONE{probeNum}.spikes.times = spkONE{probeNum}.spikes.times(spk2keep);
                                        spkONE{probeNum}.spikes.templates = spkONE{probeNum}.spikes.templates(spk2keep);
                                        spkONE{probeNum}.spikes.amps = spkONE{probeNum}.spikes.amps(spk2keep);
                                        spkONE{probeNum}.spikes.depths = spkONE{probeNum}.spikes.depths(spk2keep);
                                        spkONE{probeNum}.spikes.av_xpos = spkONE{probeNum}.spikes.av_xpos(spk2keep);
                                        spkONE{probeNum}.spikes.av_shankIDs = spkONE{probeNum}.spikes.av_shankIDs(spk2keep);
                                        
                                        % go get qmetrics??
                                        % TODO
                                        
                                        stub = [expDate '_' expNum '_' subject '_' ...
                                            sprintf('probe%d-%d',probeNum-1,alignment.ephys(probeNum).serialNumber)];
                                        fieldsSpk = fieldnames(spkONE{probeNum});
                                        for ff = 1:numel(fieldsSpk)
                                            obj = fieldsSpk{ff};
                                            fieldsObj = fieldnames(spkONE{probeNum}.(obj));
                                            for fff = 1:numel(fieldsObj)
                                                attr = fieldsObj{fff};
                                                attr = regexprep(attr,'av_','_av_');
                                                saveONEFormat(spkONE{probeNum}.(obj).(attr), ...
                                                    probeONEFolder,obj,attr,'npy',stub);
                                            end
                                        end
                                        
                                        % -----------
                                        
                                        % Remove any error file
                                        if exist(fullfile(probeONEFolder, 'GetSpkError.json'),'file')
                                            delete(fullfile(probeONEFolder, 'GetSpkError.json'))
                                        end
                                        
                                        fprintf('Block duration: %d / last spike: %d\n', block.duration, max(spk{1}.spikes.time))
                                    catch me
                                        warning(me.identifier,'Couldn''t get spikes (spk) for probe %d: threw an error (%s)',probeNum,me.message)
                                        spk{probeNum} = 'error';
                                        
                                        % Save error message locally
                                        saveErrMess(me.message,fullfile(probeONEFolder, 'GetSpkError.json'))
                                    end
                                else
                                    spk{probeNum} = nan;
                                    
                                    saveErrMess(sprintf('Couldn''t find ephys.'), ...
                                        fullfile(probeONEFolder, 'GetSpkError.json'))
                                end
                            end
                            
                            fprintf(1, '* Spikes extraction done. *\n');
                        elseif ischar(alignment.ephys) && strcmp(alignment.ephys,'error')
                            spk = 'error';
                            
                            saveErrMess(sprintf('Error in alignment.'), ...
                                fullfile(probeONEFolder, 'GetSpkError.json'))
                        elseif isa(alignment.ephys,'double') && isnan(alignment.ephys)
                            % case when no ephys folder?
                            % do nothing?
                            
                            spk = nan;
                        else
                            error('Unknown format for ephys alignment variable.')
                        end
                    else
                        error('Ephys wasn''t aligned.')
                    end
                              
                    change = 1;
                    
                    % Save it
                    if exist(savePath,'file')
                        save(savePath,'spk','-append')
                    else
                        save(savePath,'spk')
                    end
                end
                
                %% Update csv
                
                if change
                    csv.updateRecord('subject', subject, ...
                        'expDate', expDate,...
                        'expNum', expNum);
                end
            else
                fprintf('Alignment for exp. %s does not exist. Skipping.\n', expFolder)
            end
        end
        clear block timeline
    end