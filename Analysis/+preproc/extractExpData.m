function extractExpData(varargin)
    %%% This function will extract all the important information from the
    %%% experiment's timeline and block, load the spikes, align everything,
    %%% and save a preprocessed version of the data in the exp folder.
    %%% First argument is a set of params, second a list of experiments,
    %%% either in a table or cell with paths format.
    
    %% Get parameters
    varargin = ['recompute', {'none'}, varargin];
    params = csv.inputValidation(varargin{:});
    exp2checkList = csv.queryExp(params);
    
    %% --------------------------------------------------------
    %% Will compute the 'preprocData' file for each experiment.   
    
    for ee = 1:size(exp2checkList,1)
        
        % Get exp info
        expInfo = csv.inputValidation(varargin{:}, exp2checkList(ee,:));
        expFolder = expInfo.expFolder{1};
        recompute = params.recompute{1};
        
        % Define savepath for the preproc results
        pathStub = fullfile(expFolder, [expInfo.expDate{1} '_' expInfo.expNum{1} '_' expInfo.subject{1}]);
        savePath = [pathStub '_preprocData.mat'];

        if exist(savePath,'file')
            % To check if anything's missing (and that the csv hasn't seen
            % for some reason)
            varListInFile = who('-file', savePath);
        else
            varListInFile = {};
        end
        
        % Get preproc status
        preprocStatus = parseStatusCode(expInfo.preProcSpkEV);
        preprocStatus = structfun(@(x) strcmp(x,'0'), preprocStatus,'uni',0);

        if ~(strcmp(recompute,'none') && strcmp(expInfo.preProcSpkEV{1},'1,1')) 
            %% If all isn't good...
                        
            % monitors if anything has changed
            change = 0;
            
            fprintf(1, '*** Preprocessing experiment %s... ***\n', expFolder);

            % get alignment file location
            alignmentFile = strrep(savePath, 'preprocData', 'alignment');

            if exist(alignmentFile, 'file')
                %% Load alignment file
                alignment = load(alignmentFile,'ephys','block');
                
                %% Extract important info from timeline or block
                % If need be, use preproc.align.event2timeline(eventTimes,alignment.block.originTimes,alignment.block.timelineTimes)
                
                if contains(recompute,{'all';'ev'}) || preprocStatus.ev || ~ismember('ev',varListInFile)
                         
                    try
                        fprintf(1, '* Extracting events... *\n');
                        
                        % Get Block and Timeline
                        loadedData = csv.loadData(expInfo, 'loadTag', 'timelineblock');
                        timeline = loadedData.timelineData{1};
                        block = loadedData.blockData{1};
                        
                        % Get the appropriate ref for the exp def
                        expDef = expInfo.expDef{1}{1};
                        expDefRef = preproc.getExpDefRef(expDef);
                        
                        % Call specific preprocessing function
                        ev = preproc.expDef.(expDefRef)(timeline,block,alignment.block);
                        
                        % Remove any error file
                        if exist(fullfile(expFolder, 'GetEvError.json'),'file')
                            delete(fullfile(expFolder, 'GetEvError.json'))
                        end
                        
                        fprintf(1, '* Events extraction done. *\n');
                    catch me
                        warning(me.identifier,'Couldn''t get events (ev): threw an error (%s)',me.message)
                        ev = 'error';
                        
                        % Save error message locally
                        saveErrMess(me.message,fullfile(expFolder, 'GetEvError.json'))
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
                
                if contains(recompute,{'all';'spk'}) || preprocStatus.spk || ~ismember('spk',varListInFile)
                    
                    if isstruct(alignment.ephys)
                        try
                            fprintf(1, '* Extracting spikes... *\n');
                            
                            if ~exist('block','var')
                                loadedData = csv.loadData(expInfo, 'loadTag', 'block');
                                block = loadedData.blockData{1};
                            end
                            
                            spk = cell(1,numel(alignment.ephys));
                            for probeNum = 1:numel(alignment.ephys)
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
                                
                                % Recompute spike numbers
                                for c = 1:numel(spk{probeNum}.clusters)
                                    spk{probeNum}.clusters(c).Spknum = sum(spk{probeNum}.spikes.cluster == spk{probeNum}.clusters(c).ID);
                                end
                            end
                            fprintf('Block duration: %d / last spike: %d\n', block.duration, max(spk{1}.spikes.time))
                            
                            % Remove any error file
                            if exist(fullfile(expFolder, 'GetSpkError.json'),'file')
                                delete(fullfile(expFolder, 'GetSpkError.json'))
                            end
                            
                            fprintf(1, '* Spikes extraction done. *\n');
                        catch me
                            warning(me.identifier,'Couldn''t get spikes (spk): threw an error (%s)',me.message)
                            spk = 'error';
                            
                            % Save error message locally
                            saveErrMess(me.message,fullfile(expFolder, 'GetSpkError.json'))
                        end
                    elseif ischar(alignment.ephys) && strcmp(alignment.ephys,'error')
                        spk = 'error';
                    elseif isnan(alignment.ephys)
                        spk = nan;
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
                    [subject, expDate, expNum] = parseExpPath(expFolder);
                    csv.updateRecord(subject, expDate, expNum);
                end
            else
                fprintf('Alignment for exp. %s does not exist. Skipping.\n', expFolder)
            end
        end
        clear block timeline
    end