function extractExpData(varargin)
    %%% This function will extract all the important information from the
    %%% experiment's timeline and block, load the spikes, align everything,
    %%% and save a preprocessed version of the data in the exp folder.
    %%% First argument is a set of params, second a list of experiments,
    %%% either in a table or cell with paths format.
    
    %% Get parameters
    % Parameters for processing (can be inputs in varargin{1})
    params.recompute = {'none'};
    
    if ~isempty(varargin)
        paramsIn = varargin{1};
        params = parseInputParams(params,paramsIn);
        
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
        exp2checkList = queryExp();
    end
    
    %% --------------------------------------------------------
    %% Will compute the 'preprocData' file for each experiment.   
    
    for ee = 1:size(exp2checkList,1)
        
        % Get exp info
        expInfo = exp2checkList(ee,:);
        expPath = expInfo.expFolder{1};
        
        % Define savepath for the preproc results
        [subject, expDate, expNum] = parseExpPath(expPath);
        savePath = fullfile(expPath,[expDate '_' expNum '_' subject '_preprocData.mat']);
        if exist(savePath,'file')
            % To check if anything's missing (and that the csv hasn't seen
            % for some reason)
            varListInFile = who('-file', savePath);
        else
            varListInFile = {};
        end
        
        % Get preproc status
        preprocStatus = parseStatusCode(expInfo.preProcSpkEV);
        
        if ~(strcmp(params.recompute,'none') && strcmp(expInfo.preProcSpkEV{1},'1,1')) 
            %% If all isn't good...
                        
            % monitors if anything has changed
            change = 0;
            
            fprintf(1, '*** Preprocessing experiment %s... ***\n', expPath);

            % get alignment file location
            alignmentFile = dir(fullfile(expInfo.expFolder{1},'*alignment.mat'));

            if ~isempty(alignmentFile)
                %% Load alignment file
                alignment = load(fullfile(alignmentFile.folder,alignmentFile.name),'ephys','block');
                
                %% Extract important info from timeline or block
                % If need be, use preproc.align.event2timeline(eventTimes,alignment.block.originTimes,alignment.block.timelineTimes)
                
                if contains(params.recompute,'all') || contains(params.recompute,'ev') || ...
                        strcmp(preprocStatus.ev,'0') || ~ismember('ev',varListInFile)
                         
                    try
                        fprintf(1, '* Extracting events... *\n');
                        
                        % Get Block and Timeline
                        timeline = getTimeline(expPath);
                        block = getBlock(expPath);
                        
                        % Get the appropriate ref for the exp def
                        expDef = expInfo.expDef{1};
                        expDefRef = preproc.getExpDefRef(expDef);
                        
                        % Call specific preprocessing function
                        ev = preproc.expDef.(expDefRef)(timeline,block,alignment.block);
                        
                        % Remove any error file
                        if exist(fullfile(expPath, 'GetEvError.json'),'file')
                            delete(fullfile(expPath, 'GetEvError.json'))
                        end
                        
                        fprintf(1, '* Events extraction done. *\n');
                    catch me
                        warning(me.identifier,'Couldn''t get events (ev): threw an error (%s)',me.message)
                        ev = 'error';
                        
                        % Save error message locally
                        saveErrMess(me.message,fullfile(expPath, 'GetEvError.json'))
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
                
                if contains(params.recompute,'all') || contains(params.recompute,'spk') || ...
                        strcmp(preprocStatus.spk,'0') || ~ismember('spk',varListInFile)
                    
                    if isstruct(alignment.ephys)
                        try
                            fprintf(1, '* Extracting spikes... *\n');
                            
                            if ~exist('block','var')
                                block = getBlock(expPath);
                            end

                            spk = cell(1,numel(alignment.ephys));
                            for probeNum = 1:numel(alignment.ephys)
                                % Get spikes times & cluster info
                                spk{probeNum} = preproc.getSpikeData(alignment.ephys(probeNum).ephysPath);
                                
                                % Align them
                                for clu = 1:numel(spk{probeNum})
                                    spk{probeNum}(clu).spikeTimes = preproc.align.event2Timeline(spk{probeNum}(clu).spikeTimes, ...
                                        alignment.ephys(probeNum).originTimes,alignment.ephys(probeNum).timelineTimes);
                                    
                                    % Subselect the ones that are within this experiment
                                    expLength = block.duration;
                                    spk2keep = (spk{probeNum}(clu).spikeTimes>0) & (spk{probeNum}(clu).spikeTimes<expLength);
                                    spk{probeNum}(clu).spikeTimes = spk{probeNum}(clu).spikeTimes(spk2keep);
                                end
                            end
                            
                            % Remove any error file
                            if exist(fullfile(expPath, 'GetSpkError.json'),'file')
                                delete(fullfile(expPath, 'GetSpkError.json'))
                            end
                            
                            fprintf(1, '* Spikes extraction done. *\n');
                        catch me
                            warning(me.identifier,'Couldn''t get spikes (spk): threw an error (%s)',me.message)
                            spk = 'error';
                            
                            % Save error message locally
                            saveErrMess(me.message,fullfile(expPath, 'GetSpkError.json'))
                        end
                    elseif isstring(alignment.ephys) && strcmp(alignment.ephys,'error')
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
                    [subject, expDate, expNum] = parseExpPath(expPath);
                    csv.updateRecord(subject, expDate, expNum);
                end
            else
                fprintf('Alignment for exp. %s does not exist. Skipping.\n', expPath)
            end
        end
    end