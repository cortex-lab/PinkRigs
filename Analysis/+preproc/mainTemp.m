function mainTemp(varargin)
    %%% This function will run the alignment and preprocessing codes to get
    %%% the final preprocessed data for each experiment.
    
    %% Get parameters and list of mice to check
    % Parameters for processing (can be inputs in varargin{1})

    if istable(varargin{1})
        exp2checkList = varargin{1};
    else
        varargin = ['expDate', {3}, varargin];
        exp2checkList = csv.queryExp(varargin{:});
    end
    params.paramsAlign = [];
    params.paramsExtract = [];
    
    %% --------------------------------------------------------
    %% Compute and save alignment
    preproc.align.main(params.paramsAlign, exp2checkList)
    
    %% Get and save processed chunk of data
    
    preproc.extractExpData(params.paramsExtract, exp2checkList)