function extractExpData(varargin)
    %% Extracts all experimental data and saves it in ONE format. 
    % This function will extract all the important information from the
    % experiment's timeline and block, load the spikes, align everything,
    % and save a preprocessed version of the data in the exp folder.
    
    % Parameters
    % ------------------
    % Classic PinkRigs inputs (optional).
    % recompute: cell of str
    %   What to recompute. Can contain:
    %       'none', 'all', 'ephys', etc.
    % process: cell of str
    %   What to process, similarly to 'recompute'. Can contain:
    % KSversion: str
    %    'kilosort2', 'PyKS', or 'kilosort4'
   
    
    %% Get parameters
    varargin = ['recompute', 'none', varargin];
    varargin = ['process', 'all', varargin];
    varargin = ['KSversion', 'PyKS', varargin];
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
        KSversion = exp2checkList.KSversion{ee,1};
        csv.getOldPipMice;
       
        % Special case for old mice from Coen&Sit paper
        if contains(subject, oldPipMice); oldPipTag = 1; else, oldPipTag = 0; end

        % Get the alignment file
        pathStub = fullfile(expFolder, [expDate '_' expNum '_' subject]);
        alignmentFile = [pathStub '_alignment.mat'];

        %%% temporary--delete the old preproc files
        oldPreprocFile = regexprep(alignmentFile,'alignment','preproc');
        if exist(oldPreprocFile, 'file')
            delete(oldPreprocFile);
        end
        oldPreprocFolder = fullfile(expFolder,'preproc');
        if exist(oldPreprocFolder, 'dir')
            rmdir(oldPreprocFolder,'s');
        end
        %%%
        
        % Get extraction status
        notExtracted.events = any(contains(exp2checkList.extractEvents{ee,1},'0'));
        notExtracted.spikes = any(contains(exp2checkList.extractSpikes{ee,1},'0'));
        
        % Anonymous function to decide whether something should be processed
        shouldProcess = @(x) (contains(recompute,{'all';x}) || ...
            notExtracted.(x)) && contains(process,{'all';x});

        if ~(strcmp(recompute,'none') && all(structfun(@(x) x~=1, notExtracted)))
            %% If all isn't good...
                        
            % monitors if anything has changed
            change = 0;
            
            fprintf(1, '*** Preprocessing experiment %s (%d/%d)... ***\n', expFolder,ee,size(exp2checkList,1));

            if exist(alignmentFile, 'file') || oldPipTag
                %% Extract important info from timeline or block
                % If need be, use preproc.align.event2timeline(eventTimes,alignment.block.originTimes,alignment.block.timelineTimes)
                
                if shouldProcess('events')
                    if strcmp(expInfo.alignBlock, '1') || oldPipTag
                        if ~oldPipTag
                            alignment = load(alignmentFile, 'block');
                        else
                            alignment.block = nan;
                        end

                        % Get the events ONE folder
                        eventsONEFolder = fullfile(expFolder,'ONE_preproc','events');
                        initONEFolder(eventsONEFolder)
                        
                        try
                            fprintf(1, '* Extracting events... *\n');
                            
                            % Get Block and Timeline
                            loadedData = csv.loadData(expInfo, 'dataType', {{'timeline'; 'block'}});
                            block = loadedData.dataBlock{1};
                            if ~oldPipTag
                                timeline = loadedData.dataTimeline{1};
                            else
                                %Deal with Coen&Sit mice
                                timeline = nan;
                                pipParams = load(strrep(alignmentFile, 'alignment', 'parameters'));
                                galvoLog = strrep(alignmentFile, 'alignment', 'galvoLog');
                                if exist(galvoLog, 'file')
                                    block.galvoLog = load(galvoLog);
                                else
                                    block.galvoLog = 0;
                                end
                                if isstruct(galvoLog) && length(fieldnames(galvoLog))<2
                                    block.galvoLog = 0;
                                end
                                % Needed because there was an issue with old recordings where the 1st trial
                                % timings were wrong.
                                [block, block.galvoLog] = removeFirstTrialFromBlock(block, block.galvoLog);

                                % Need to account for name changes
                                block = standardPipMiceBlkNames(block, pipParams.parameters);
                            end
                            block.expInfo = expInfo; % need to pass down expInfo for opening the optoLog

                            
                            % Get the appropriate ref for the exp def
                            expDefRef = preproc.getExpDefRef(expDef);
                            
                            % Call specific preprocessing function
                            events = preproc.expDef.(expDefRef)(timeline,block,alignment.block);
                            
                            stub = [expDate '_' expNum '_' subject];
                            if any(structfun(@(x) size(x,3)>1, events))
                                % Remove later. Sanity Check.
                                if ~contains(expDefRef, 'sparseNoise'); keyboard; end
                                saveFields = fields(events);
                                cellfun(@(x) saveONEFormat(events.(x),eventsONEFolder, ...
                                    '_av_trials', x, 'npy', stub), saveFields, 'uni',0);
                            else
                                saveONEFormat(events,eventsONEFolder,'_av_trials','table','pqt',stub);
                            end
                             
                            % Remove any error file
                            if exist(fullfile(eventsONEFolder, 'GetEvError.json'),'file')
                                delete(fullfile(eventsONEFolder, 'GetEvError.json'))
                            end
                            
                            fprintf(1, '* Events extraction done. *\n');
                            
                        catch me
                            msgText = getReport(me);
                            warning('%s \n Could not get events (ev): threw an error\n %s',me.identifier, msgText)
                            
                            % Save error message locally
                            saveErrMess(msgText,fullfile(eventsONEFolder, 'GetEvError.json'))
                        end
                        
                        change = 1;
                    else
                        % Do nothing
                        fprintf('No successful alignment for block, skipping. \n')
                    end
                end
                    
                %% Extract spikes and clusters info (depth, etc.)
                
                if shouldProcess('spikes')
                    if contains(expInfo.alignEphys, '1') 
                        fprintf (1, '* Extracting spikes... *\n');
                        
                        alignment = load(alignmentFile, 'ephys');
                        
                        if ~exist('block','var')
                            loadedData = csv.loadData(expInfo, 'dataType', 'block');
                            block = loadedData.dataBlock{1};
                        end

                        for probeNum = 1:numel(alignment.ephys)
                            
                            % Get the probe's ONE folder
                            probeONEFolder = fullfile(expFolder,'ONE_preproc',sprintf('probe%d',probeNum-1));

                            switch KSversion
                                case 'kilosort2'
                                    KSFolder = fullfile(alignment.ephys(probeNum).ephysPath,'kilosort2');
                                case 'PyKS'
                                    KSFolder = fullfile(alignment.ephys(probeNum).ephysPath,'PyKS','output');
                                case 'kilosort4'
                                    KSFolder = fullfile(alignment.ephys(probeNum).ephysPath,'kilosort4');
                            end
                            IBLFormatFolder = fullfile(KSFolder,'ibl_format');

                            stub = [expDate '_' expNum '_' subject '_' ...
                                sprintf('probe%d-%d',probeNum-1,alignment.ephys(probeNum).serialNumber)];

                            try
                                if shouldProcess('spikes') && exist(IBLFormatFolder,'dir')
                                    % Initialize the folder (do it here
                                    % because want to delete it only when
                                    % processing the spikes)
                                    initONEFolder(probeONEFolder) %%% will have to see what to do when recomputing the qMetrics only

                                    % Get the spike and cluster info
                                    spkONE = preproc.getSpikeDataONE(KSFolder);
                                    
                                    % Write a json file in target ONE that
                                    % contains the string of the IBL
                                    % format file 
                                    saveErrMess(IBLFormatFolder,fullfile(probeONEFolder, sprintf('_av_rawephys.path.%s.json',stub)))

                                    % Align them
                                    spkONE.spikes.times = preproc.align.event2Timeline(spkONE.spikes.times, ...
                                        alignment.ephys(probeNum).originTimes,alignment.ephys(probeNum).timelineTimes);

                                    % Subselect the ones that are within this experiment
                                    expLength = block.duration;
                                    spk2keep = (spkONE.spikes.times>0) & (spkONE.spikes.times<expLength);
                                    spkONE.spikes.times = spkONE.spikes.times(spk2keep);
                                    spkONE.spikes.templates = spkONE.spikes.templates(spk2keep);
                                    spkONE.spikes.clusters = spkONE.spikes.clusters(spk2keep);
                                    spkONE.spikes.amps = spkONE.spikes.amps(spk2keep);
                                    spkONE.spikes.depths = spkONE.spikes.depths(spk2keep);
                                    spkONE.spikes.av_xpos = spkONE.spikes.av_xpos(spk2keep);
                                    spkONE.spikes.av_shankIDs = spkONE.spikes.av_shankIDs(spk2keep);

                                    fieldsSpk = fieldnames(spkONE);
                                    for ff = 1:numel(fieldsSpk)
                                        obj = fieldsSpk{ff};
                                        fieldsObj = fieldnames(spkONE.(obj));
                                        for fff = 1:numel(fieldsObj)
                                            attr = fieldsObj{fff};
                                            % note that the regexprep is because a field cannot start with '_'...
                                            saveONEFormat(spkONE.(obj).(attr), ...
                                                probeONEFolder,obj,regexprep(attr,'av_','_av_'),'npy',stub);
                                        end
                                    end
                                    fprintf('Block duration: %d / last spike: %d\n', block.duration, max(spkONE.spikes.times))

                                    % Get IBL quality metrics
                                    qMetrics = preproc.getQMetrics(KSFolder,'IBL');
                                    % The qMetrics don't get calculated for some trash units, but we need to keep
                                    % the dimensions consistent of course...
                                    qMetrics = removevars(qMetrics,{'cluster_id_1'}); % remove a useless variable
                                    cname = setdiff(spkONE.clusters.av_IDs,qMetrics.cluster_id);

                                    added_array = qMetrics(1,:);
                                    added_array{1,added_array.Properties.VariableNames(1:end-1)} = nan;
                                    added_array{1,'ks2_label'} = {'noise'};

                                    for i=1:numel(cname)
                                        added_array(1,'cluster_id') = {cname(i)};
                                        qMetrics = [qMetrics;added_array];
                                    end
                                    qMetrics = sortrows(qMetrics,'cluster_id');
                                    saveONEFormat(qMetrics, ...
                                        probeONEFolder,'clusters','_av_qualityMetrics','pqt',stub);

                                    % Get Bombcell metrics
                                    if exist(fullfile(KSFolder,'qMetrics','templates._bc_qMetrics.parquet'),'file')
                                        qMetrics = preproc.getQMetrics(KSFolder,'bombcell');
                                        qMetrics.clusterID = qMetrics.clusterID - 1; % base 0 indexing
                                        saveONEFormat(qMetrics, ...
                                            probeONEFolder,'clusters','_bc_qualityMetrics','pqt',stub);
                                    else
                                        error('Quality metrics haven''t been computed...')
                                    end
                                end

                                % Remove any error file
                                if exist(fullfile(probeONEFolder, 'GetSpkError.json'),'file')
                                    delete(fullfile(probeONEFolder, 'GetSpkError.json'))
                                end
                                
                            catch me
                                msgText = getReport(me);
                                warning(me.identifier,'Couldn''t get spikes (spk) for probe %d: threw an error (%s)',probeNum,msgText)
                                                                
                                % Save error message locally
                                saveErrMess(msgText,fullfile(probeONEFolder, 'GetSpkError.json'))
                            end
                        end
                        
                        fprintf(1, '* Spikes extraction done. *\n');
                    else
                        % Do nothing
                        fprintf('No successful alignment or spikesorting for ephys, skipping. \n')
                    end
                              
                    change = 1;
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