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
        
        % Get the alignment file
        pathStub = fullfile(expFolder, [expDate '_' expNum '_' subject]);
        alignmentFile = [pathStub '_alignment.mat'];

        %%% temporary--delete the old preproc files
        oldPreprocFile = regexprep(alignmentFile,'alignment','preproc');
        if exist(oldPreprocFile, 'file')
            delete(oldPreprocFile);
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
            
            fprintf(1, '*** Preprocessing experiment %s... ***\n', expFolder);

            if exist(alignmentFile, 'file')
                %% Extract important info from timeline or block
                % If need be, use preproc.align.event2timeline(eventTimes,alignment.block.originTimes,alignment.block.timelineTimes)
                
                if shouldProcess('events')
                    if strcmp(expInfo.alignBlock, '1')
                        alignment = load(alignmentFile, 'block');
                        
                        % Get the events ONE folder
                        eventsONEFolder = fullfile(expFolder,'ONE_preproc','events');
                        initONEFolder(eventsONEFolder)
                        
                        try
                            fprintf(1, '* Extracting events... *\n');
                            
                            % Get Block and Timeline
                            loadedData = csv.loadData(expInfo, 'dataType', {{'timeline'; 'block'}});
                            timeline = loadedData.dataTimeline{1};
                            block = loadedData.dataBlock{1};
                            
                            % Get the appropriate ref for the exp def
                            expDefRef = preproc.getExpDefRef(expDef);
                            
                            % Call specific preprocessing function
                            events = preproc.expDef.(expDefRef)(timeline,block,alignment.block);
                            
                            stub = [expDate '_' expNum '_' subject];
                            saveONEFormat(events,eventsONEFolder,'_av_trials','table','pqt',stub);
                            
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
                            initONEFolder(probeONEFolder) %%% will have to see what to do when recomputing the qMetrics only
                            
                            try
                                % Get the spike and cluster info
                                spkONE = preproc.getSpikeDataONE(alignment.ephys(probeNum).ephysPath);
                                
                                % Align them
                                spkONE.spikes.times = preproc.align.event2Timeline(spkONE.spikes.times, ...
                                    alignment.ephys(probeNum).originTimes,alignment.ephys(probeNum).timelineTimes);
                                
                                % Subselect the ones that are within this experiment
                                expLength = block.duration;
                                spk2keep = (spkONE.spikes.times>0) & (spkONE.spikes.times<expLength);
                                spkONE.spikes.times = spkONE.spikes.times(spk2keep);
                                spkONE.spikes.templates = spkONE.spikes.templates(spk2keep);
                                spkONE.spikes.amps = spkONE.spikes.amps(spk2keep);
                                spkONE.spikes.depths = spkONE.spikes.depths(spk2keep);
                                spkONE.spikes.av_xpos = spkONE.spikes.av_xpos(spk2keep);
                                spkONE.spikes.av_shankIDs = spkONE.spikes.av_shankIDs(spk2keep);
                                
                                % go get qmetrics??
                                % TODO
                                
                                stub = [expDate '_' expNum '_' subject '_' ...
                                    sprintf('probe%d-%d',probeNum-1,alignment.ephys(probeNum).serialNumber)];
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
                                
                                % Remove any error file
                                if exist(fullfile(probeONEFolder, 'GetSpkError.json'),'file')
                                    delete(fullfile(probeONEFolder, 'GetSpkError.json'))
                                end
                                
                                fprintf('Block duration: %d / last spike: %d\n', block.duration, max(spkONE.spikes.times))
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
                        fprintf('No successful alignment for ephys, skipping. \n')
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