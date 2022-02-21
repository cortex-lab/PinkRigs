function ev = AVprotocol(timeline, block, alignment)
    %%% This function will fetch all important information from the AV
    %%% protocols, during postactive or during training.
    
    %% Extract photodiode onsets in timeline
    % might need to clean a bit?
    
    ev.visStimOnsetTime = timeproc.getChanEventTime(timeline,'photoDiode');
    
    %% Extract sounds onsets
    % might need to clean a bit?
    
    ev.audStimOnsetTime = timeproc.getChanEventTime(timeline,'audio');
    
    %% Extract reward onsets
    
    ev.rewardOnsetTimes = timeproc.getChanEventTime(timeline,'reward');
    
    %% Extract trial info