function extractExpData(varargin)
    %%% This function will extract all the important information from the
    %%% experiment's timeline and block, load the spikes, align everything,
    %%% and save a preprocessed version of the data in the exp folder.
    %%% First argument is a set of params, second a list of experiments,
    %%% either in a table or cell with paths format.
    
    %% Get parameters
    % Parameters for processing (can be inputs in varargin{1})
    params.recompute = 0;
    
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
        exp2checkList = getAllExp2Check();
    end
    
    %% --------------------------------------------------------
    %% Will compute the 'preprocData' file for each experiment.   
    
    for ee = 1:size(exp2checkList,1)
        
        % Can also amend the csv to say whether this one has been
        % aligned or not.
        
        % Get exp info
        %%% Here could also extract other info? Has it been aligned? Which
        %%% ephys goes with that one? Etc.
        expInfo = exp2checkList(ee,:);
        expPath = expInfo.expFolder{1};
        
        [subject, expDate, expNum] = parseExpPath(expPath);
        savePath = fullfile(expPath,[expDate '_' expNum '_' subject '_preprocData.mat']);
        if exist(fullfile(expPath,'preprocData.mat'))
            movefile(fullfile(expPath,'preprocData.mat'),savePath)
        end
        
        fprintf(1, '*** Preprocessing experiment %s... ***\n', expPath);
        
        if ~exist(savePath,'file') || params.recompute
            % get alignment file location
            alignmentFile = dir(fullfile(expInfo.expFolder{1},'*alignment.mat'));

            if ~isempty(alignmentFile)
                %% Load alignment file
                load(fullfile(alignmentFile.folder,alignmentFile.name), 'alignment');
                
                %% Extract important info from timeline or block
                % If need be, use preproc.align.event2timeline(eventTimes,alignment.block.originTimes,alignment.block.timelineTimes)
                
                % Get Block and Timeline
                timeline = getTimeline(expPath);
                block = getBlock(expPath);
                
                try
                    fprintf(1, '* Extracting events... *\n');
                    % Get the appropriate ref for the exp def
                    expDef = expInfo.expDef{1};
                    expDefRef = preproc.getExpDefRef(expDef);
                    
                    % Call specific preprocessing function
                    ev = preproc.expDef.(expDefRef)(timeline,block,alignment);
                    fprintf(1, '* Events extraction done. *\n');
                catch me
                    warning(me.identifier,'Couldn''t get events (ev): threw an error (%s)',me.message)
                    ev = 'error';
                    
                    % Save error message locally
                    saveErrMess(me.message,fullfile(expPath, 'GetEvError.json'))
                end
                
                    
                %% Extract spikes and clusters info (depth, etc.)
                if isstruct(alignment.ephys)
                    try
                        fprintf(1, '* Extracting spikes... *\n');
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
                
                %% Save all
                save(savePath,'spk','ev')
            else
                fprintf('Alignment for exp. %s does not exist. Skipping.\n', expPath)
            end
            
            %% Update csv
            [subject, expDate, expNum] = parseExpPath(expPath);
            csv.updateRecord(subject, expDate, expNum);
        end
    end