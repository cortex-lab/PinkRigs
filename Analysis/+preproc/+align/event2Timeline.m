function events_inTimelineTime = event2Timeline(events_inOriTime, originTime_ref, timelineTime_ref)
    %%% This function will compute the times of events in timeline ref.
    
    events_inTimelineTime = interp1(originTime_ref, timelineTime_ref, events_inOriTime, 'linear', 'extrap');