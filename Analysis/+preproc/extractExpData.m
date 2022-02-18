function extractExpData(varargin)
    %%% This function will extract all the important information from the
    %%% experiment's timeline and block, load the spikes, align everything,
    %%% and save a preprocessed version of the data in the exp folder.
    %%% First argument is a set of params, second a list of experiments,
    %%% either in a table or cell with paths format.
    
    %% Get parameters
    % Parameters for processing (can be inputs in varargin{1})
    recompute = 0;
    
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
    %% Will compute the 'alignment' file for each experiment.   
    
    for ee = 1:size(exp2checkList,1)
        
        % Can also amend the csv to say whether this one has been
        % aligned or not.
        
        % Get exp info
        %%% Here could also extract other info? Has it been aligned? Which
        %%% ephys goes with that one? Etc.
        expInfo = exp2checkList(ee,:);
        expPath = expInfo.path{1};
        
        savePath = fullfile(expPath,'preprocData.m');
        
        if ~exist(savePath,'file') || recompute
            % get alignment file location
            alignmentFile = fullfile(expInfo.path{1},'alignment.mat');
            
            if exist(alignmentFile, 'file')
                %% Load alignment file
                load(alignmentFile, 'alignment');
                
                %% Extract spikes and clusters info (depth, etc.)
                if ~isempty(alignment.ephys)
                    % Get spikes
                    
                    % Align them
                    for probeNum = 1:numel(alignment.ephys)
                        spikeTimesAligned{probeNum} = preproc.align.event2timeline(spikeTimes, ...
                            alignment.ephys(probeNum).originTimes,alignment.ephys(probeNum).timelineTimes);
                        
                        % Subselect the ones that are within this experiment
                    end
                end
                
                %% Extract important info from timeline or block
                % If need be, use preproc.align.event2timeline(eventTimes,alignment.block.originTimes,alignment.block.timelineTimes)
                
                % Get Block and Timeline
                timeline = getTimeline(expPath);
                block = getBlock(expPath);
    
                % Extract exp def that was used
                expDef = expInfo.expDef{1};
                if contains(expDef,'imageWorld')
                    expDefRef = 'imageWorld';
                else
                    %%% TODO fill in that part with you own expDefs...
                end
                
                % Call specific preprocessing function
                ev = preproc.expDef.(expDefRef)(timeline,block,alignment);
                
                %% Save all
                save(savePath,'sp','ev')
                
            else
                frprintf('Alignment for exp. %s does not exist. Skipping.\n', expPath)
            end
        end
    end