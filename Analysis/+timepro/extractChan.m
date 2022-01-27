function chan = extractChan(timeline,chanName)
    %%% Will extract specific channel from timeline.
    
    if strcmp(chanName,'time')
        % extract the time channel
        chan = timeline.rawDAQTimestamps;
    else
        syncIndex = find(strcmp({timeline.hw.inputs.name}, chanName));
        if ~isempty(syncIndex)
            chan = Timeline.rawDAQData(:,syncIndex);
        else
            chan = [];
            warning('Couldn''t find channel %s in timeline.',chanName)
        end
    end