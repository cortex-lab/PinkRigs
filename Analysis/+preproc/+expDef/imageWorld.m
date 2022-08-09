function ev = imageWorld(timeline, block, alignmentBlock)
    %%% This function will fetch all important information from the expDef
    %%% imageWorld.
    
    %% Extract photodiode onsets in timeline
    %%% Note that here it will correspond to the real photodiode onsets,
    %%% which have the best precision (because photodiode is used for the
    %%% alignment). 
  
    ev.imageOnsetTimes = preproc.align.event2Timeline(block.events.stimulusOnTimes(block.events.stimulusOnValues)', ...
        alignmentBlock.originTimes,alignmentBlock.timelineTimes);
            
    ev.imageOffsetTimes = preproc.align.event2Timeline(block.events.stimulusOnTimes(~block.events.stimulusOnValues)', ...
        alignmentBlock.originTimes,alignmentBlock.timelineTimes);
    
    %% Get image ID
    
    ev.imageIDs = block.events.numValues';

    %% Make them the same size

    trialNum = min([numel(ev.imageIDs), numel(ev.imageOnsetTimes), numel(ev.imageOffsetTimes)]);
    ev.imageOnsetTimes = ev.imageOnsetTimes(1:trialNum);
    ev.imageOffsetTimes = ev.imageOffsetTimes(1:trialNum);
    ev.imageIDs = ev.imageIDs(1:trialNum);