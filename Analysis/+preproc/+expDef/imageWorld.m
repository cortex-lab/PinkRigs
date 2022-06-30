function ev = imageWorld(timeline, block, alignmentBlock)
    %%% This function will fetch all important information from the expDef
    %%% imageWorld.
    
    %% Extract photodiode onsets in timeline
    %%% Note that here it will correspond to the real photodiode onsets,
    %%% which have the best precision (because photodiode is used for the
    %%% alignment). 
  
    ev.imageOnsetTime = preproc.align.event2Timeline(block.events.stimulusOnTimes(block.events.stimulusOnValues), ...
        alignmentBlock.originTimes,alignmentBlock.timelineTimes);
            
    ev.imageOffsetTime = preproc.align.event2Timeline(block.events.stimulusOnTimes(~block.events.stimulusOnValues), ...
        alignmentBlock.originTimes,alignmentBlock.timelineTimes);
    
    %% Get image ID
    
    ev.imageID = block.events.numValues;