function main(varargin)
    %%% This function will run the alignment and preprocessing codes to get
    %%% the final preprocessed data for each experiment.
    
    %% Get parameters and list of mice to check
    % Parameters for processing (can be inputs in varargin{1})
    recompute = {'none'};
    paramsVid = []; % will take default
    
    % This is not ideal
    if ~isempty(varargin)
        params = varargin{1};
        
        if ~isempty(params) && isfield(params, 'recompute')
            recompute = params.recompute;
        end
        if ~isempty(params) && isfield(params, 'paramsVid')
            paramsVid = params.paramsVid;
        end
        
        if numel(varargin) > 1
            expPath2checkList = varargin{2};
            
            % Check that they are in the main csvs and fetch exp info
            %%% Maybe put in a function, getExpInfoFromPath ?
            
            %%% Should update csv ffirst?
        end
    end