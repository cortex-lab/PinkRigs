function main(varargin)
    %%% This function will run the alignment and preprocessing codes to get
    %%% the final preprocessed data for each experiment.
    
    %% Get parameters and list of mice to check
    % Parameters for processing (can be inputs in varargin{1})
    paramsAlign = [];
    days2Check = inf;
    mice2Check = 'active';
    
    % This is not ideal
    if ~isempty(varargin)
        params = varargin{1};
        
        if ~isempty(params) && isfield(params, 'days2Check')
            days2Check = params.days2Check;
        end
        if ~isempty(params) && isfield(params, 'mice2Check')
            mice2Check = params.mice2Check;
        end
        if ~isempty(params) && isfield(params, 'paramsAlign')
            paramsAlign = params.paramsAlign;
        end
        
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
        p.days2Check = days2Check;
        p.mice2Check = mice2Check;
        exp2checkList = getAllExp2Check(p);
    end
    
    %% --------------------------------------------------------
    %% Compute and save alignment
    
    preproc.align.main(paramsAlign, exp2checkList)
    
    %% Get and save processed chunk of data
    
    preproc.extractExpData(paramsExtract, exp2checkList)