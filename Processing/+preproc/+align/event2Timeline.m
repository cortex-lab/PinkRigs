function events_inTimelineTime = event2Timeline(events_inOriTime, originTime_ref, timelineTime_ref)
    %% Aligns events to timeline time.
    %
    % Parameters:
    % -------------------
    % events_inOriTime: vector
    %   Timings of events in their original time.
    % originTime_ref: vector
    %   Timings of reference points in their original time.
    % timelineTime_ref: vector
    %   Timings of reference points in timeline time.

    %
    % Returns: 
    % -------------------
    % events_inTimelineTime: vector
    %   Timings of events in timeline time.
    
    events_inTimelineTime = interp1(originTime_ref, timelineTime_ref, events_inOriTime, 'linear', nan);
    nanIdx = isnan(events_inTimelineTime);
    
    if any(nanIdx)
        refOffsets = timelineTime_ref(:)-originTime_ref(:);
        offSetsPerPoint = interp1(originTime_ref, refOffsets, events_inOriTime, 'nearest', 'extrap');
        
        events_inTimelineTime(nanIdx) = events_inOriTime(nanIdx)+offSetsPerPoint(nanIdx);
    end