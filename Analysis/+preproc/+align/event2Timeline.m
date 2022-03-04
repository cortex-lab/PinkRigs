function events_inTimelineTime = event2Timeline(events_inOriTime, originTime_ref, timelineTime_ref)
    %%% This function will compute the times of events in timeline ref.
    
    events_inTimelineTime = interp1(originTime_ref, timelineTime_ref, events_inOriTime, 'linear', nan);
%     events_inTimelineTime = interp1(originTime_ref, timelineTime_ref, events_inOriTime, 'linear', 'extrap');
    nanIdx = isnan(events_inTimelineTime);
    
    if any(nanIdx)
        refOffsets = timelineTime_ref(:)-originTime_ref(:);
        offSetsPerPoint = interp1(originTime_ref, refOffsets, events_inOriTime, 'nearest', 'extrap');
        
        events_inTimelineTime(nanIdx) = events_inOriTime(nanIdx)+offSetsPerPoint(nanIdx);
    end