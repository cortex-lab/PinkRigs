function [timelineRefTimes, chanName] = extractBestPhotodiode(timeline,block)
    %% Extracts the best photodiode times (closest to block flips) 
    %
    % Parameters:
    % -------------------
    % timeline: struct
    %   Timeline struct
    % block: struct
    %   Block stuct
    %
    % Returns: 
    % -------------------
    % timelineRefTimes: vector of photodiode flip times
    %   Contains the event times
    %
    % chanName: name of the "best" photodiode channel
    %   Contains the event times

    
    %% Extract channel
    numberOfblockFlips = length(block.stimWindowUpdateTimes);
    timelineRefTimes = {};
    chanName = {};
    if any(strcmpi({timeline.hw.inputs.name}', 'photoDiode'))
        timelineRefTimes = [timelineRefTimes; timeproc.getChanEventTime(timeline,'photoDiode')];
        chanName = [chanName; 'photoDiode'];
    end
    if any(strcmpi({timeline.hw.inputs.name}', 'photoDRaw'))
        timelineRefTimes = [timelineRefTimes; timeproc.getChanEventTime(timeline,'photoDRaw')];
        chanName = [chanName; 'photoDRaw'];
    end
    if any(strcmpi({timeline.hw.inputs.name}', 'photoDThorLabs'))
        timelineRefTimes = [timelineRefTimes; timeproc.getChanEventTime(timeline,'photoDThorLabs')];
        chanName = [chanName; 'photoDThorLabs'];
    end

    [~, bestIdx] = min(abs(cellfun(@length, timelineRefTimes)-numberOfblockFlips));
    timelineRefTimes = timelineRefTimes{bestIdx}(:)';
    chanName = chanName{bestIdx};
end
    