function s = expDef_imageWorld(timeline, block, alignment)
    %%% This function will fetch all important information from the expDef
    %%% imageWorld.
    
    %% Extract photodiode onsets in timeline
    %%% Note that here it will correspond to the real photodiode onsets,
    %%% which have the best precision (because photoiode is used for the
    %%% alignment). 
    
    blockOnsetTimes = block.stimWindowUpdateTimes;    
    s.imageOnsetTime = align.event2timeline(blockOnsetTimes,alignment.block.originTimes,alignment.block.timelineTimes);
            
    %% Get image ID
    
    s.imageID = block.events.numValues;