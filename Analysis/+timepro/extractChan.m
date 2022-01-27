function chan = extractChan(Timeline,chanName)
    %%% Will extract specific channel from timeline.
    
    if strcmp(chanName,'time')
        % extract the time channel
        chan = Timeline.rawDAQTimestamps;
    else
        syncIndex = find(strcmp({Timeline.hw.inputs.name}, chanName));
        if ~isempty(syncIndex)
            chan = Timeline.rawDAQData(:,syncIndex);
        else
            chan = [];
            warning('Couldn''t find channel %s in timeline.',chanName)
        end
    end