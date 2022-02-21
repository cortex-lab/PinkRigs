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
        expPath = expInfo.path{1};
        
        savePath = fullfile(expPath,'preprocData.mat');
        
        if ~exist(savePath,'file') || params.recompute
            % get alignment file location
            alignmentFile = fullfile(expInfo.path{1},'alignment.mat');
            
            if exist(alignmentFile, 'file')
                %% Load alignment file
                load(alignmentFile, 'alignment');
                
                %% Extract important info from timeline or block
                % If need be, use preproc.align.event2timeline(eventTimes,alignment.block.originTimes,alignment.block.timelineTimes)
                
                % Get Block and Timeline
                timeline = getTimeline(expPath);
                block = getBlock(expPath);
                
                % Extract exp def that was used
                expDef = expInfo.expDef{1};
                if contains(expDef,'imageWorld')
                    expDefRef = 'imageWorld';
                elseif contains(expDef,'spontaneousActivity')
                    expDefRef = 'spontaneous';
                elseif contains(expDef,'sparseNoise')
                    expDefRef = 'sparseNoise';
                else
                    %%% TODO fill in that part with you own expDefs...
                end
                
                % Call specific preprocessing function
                ev = preproc.expDef.(expDefRef)(timeline,block,alignment);
                
                %% Extract spikes and clusters info (depth, etc.)
                if ~isempty(alignment.ephys)
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
                else
                    spk = [];
                end
                
                %% Save all
                save(savePath,'spk','ev')
            else
                frprintf('Alignment for exp. %s does not exist. Skipping.\n', expPath)
            end
        end
    end