function events_tlTime = block2tl(events_blTime, blTime_ref, tlTime_ref)
    %%% This function will compute the times of events in timeline ref.
    
    events_tlTime = interp1(blTime_ref, tlTime_ref, events_blTime, 'linear', 'extrap');