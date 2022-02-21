function chan = extractChan(timeline,chanName,varargin)
    %%% Will extract specific channel from timeline.
    %%% Additional argument to display or not warning when not found.
    
    if nargin < 3
        dispWaring = 1;
    else
        dispWaring = varargin{1};
    end
    
    
    if strcmp(chanName,'time')
        % extract the time channel
        chan = timeline.rawDAQTimestamps;
    else
        chanIndex = find(strcmp({timeline.hw.inputs.name}, chanName));
        if ~isempty(chanIndex)
            chan = timeline.rawDAQData(:,chanIndex);
        else
            chan = [];
            if dispWaring
                warning('Couldn''t find channel %s in timeline.',chanName)
            end
        end
    end