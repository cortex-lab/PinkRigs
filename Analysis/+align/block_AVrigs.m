function block_AVrigs(subject, expDate, expNum, varargin)
    %%% This function aligns a block file with its corresponding timeline
    %%% file. 
    
    %% get path and parameters
    % get experiment's path
    expPath = getExpPath(subject, expDate, expNum);
    
    % parameters for processing (can be inputs in varargin{1})
    % empty for now?
    
    % this is not ideal 
    if ~isempty(varargin)
        params = varargin{1};

        if numel(varargin)>1
            Timeline = varargin{2};
        end
    end
    
    %% get photodiode and wheel data
    
    if ~exist('Timeline','var')
        fprintf(1, 'loading timeline\n');
        Timeline = timepro.getTimeline(subject,expDate,expNum);
    end
    
    