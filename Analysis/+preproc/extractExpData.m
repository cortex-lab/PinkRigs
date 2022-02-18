function extractExpData(varargin)
    %%% This function will extract all the important information from the
    %%% experiment's timeline and block, load the spikes, align everything,
    %%% and save a preprocessed version of the data in the exp folder.
    
    %% Get parameters
    % Parameters for processing (can be inputs in varargin{1})
    alignType = [];
    
    % This is not ideal
    if ~isempty(varargin)
        params = varargin{1};
        
        if ~isempty(params) && isfield(params, 'alignType')
            alignType = params.alignType;
        end
        
        if numel(varargin) > 1
            timeline = varargin{2};
        end
    end