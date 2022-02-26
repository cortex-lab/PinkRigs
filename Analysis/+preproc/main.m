function main(varargin)
    %%% This function will run the alignment and preprocessing codes to get
    %%% the final preprocessed data for each experiment.
    
    %% Get parameters and list of mice to check
    % Parameters for processing (can be inputs in varargin{1})
    params.paramsAlign = [];
    params.paramsExtract = [];
    params.days2Check = inf; % back in time from today
    params.mice2Check = 'active';

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
        exp2checkList = queryExp(params);
    end
    
    %% --------------------------------------------------------
    %% Compute and save alignment
    
    preproc.align.main(params.paramsAlign, exp2checkList)
    
    %% Get and save processed chunk of data
    
    preproc.extractExpData(params.paramsExtract, exp2checkList)