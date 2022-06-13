function chan = extractChan(timeline,chanName,varargin)
    %%% Will extract specific channel from timeline.
    %%% Additional argument to display or not warning when not found.

    if nargin < 3
        dispWaring = 1;
    else
        dispWaring = varargin{1};
    end
    
    
    if strcmpi(chanName,'time')
        % extract the time channel
        chan = timeline.rawDAQTimestamps;
    else
        chanIndex = find(strcmpi({timeline.hw.inputs.name}, chanName));

        %To deal with changing name from soundOutput to audioOut
        if isempty(chanIndex) && strcmpi(chanName, 'audioOut')
            fprintf('No "audioOut... will try soundOutput \n')
            chanIndex = find(strcmpi({timeline.hw.inputs.name}, 'soundOutput'));
        end

        if ~isempty(chanIndex)
            chan = timeline.rawDAQData(:,chanIndex);
        else
            chan = [];
            if dispWaring
                warning('Couldn''t find channel %s in timeline.',chanName)
            end
        end
    end