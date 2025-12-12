function ev = imageWorld(timeline, block, alignmentBlock)
    %% Fetches all important information from the imageWorld protocols
    %
    % Parameters:
    % -------------------
    % timeline: struct
    %   Timeline structure.
    % block: struct
    %   Block structure
    % alignmentBlock: struct
    %   Alignement structure, containing fields "originTimes" and
    %   "timelineTimes"
    %
    % Returns: 
    % -------------------
    % ev: struct
    %   Structure containing all relevant events information. 
    %   All fields should have the form [nxm] where n is the number of trials.
    %       imageOnsetTimes: image onset times
    %       imageOffsetTimes: image offset times
    %       imageIDs: image IDs
    
    %% Extract photodiode onsets in timeline
    %%% Note that here it will correspond to the real photodiode onsets,
    %%% which have the best precision (because photodiode is used for the
    %%% alignment). 
  
    ev.imageOnsetTimes = preproc.align.event2Timeline(block.events.stimulusOnTimes(block.events.stimulusOnValues)', ...
        alignmentBlock.originTimes,alignmentBlock.timelineTimes);

    if any(ev.imageOnsetTimes < 0)
        error('negative image onset times?')
    end

    ev.imageOffsetTimes = preproc.align.event2Timeline(block.events.stimulusOnTimes(~block.events.stimulusOnValues)', ...
        alignmentBlock.originTimes,alignmentBlock.timelineTimes);

    if any(ev.imageOffsetTimes > max(alignmentBlock.timelineTimes))
        disp('image offset overshoot? delete last one?')
%         ev.imageOnsetTimes(ev.imageOffsetTimes > max(alignmentBlock.timelineTimes)) = [];
%         ev.imageOffsetTimes(ev.imageOffsetTimes > max(alignmentBlock.timelineTimes)) = [];
    end
    
    %% Get image ID
    
    ev.imageIDs = block.events.numValues';

    %% Make them the same size

    trialNum = min([numel(ev.imageIDs), numel(ev.imageOnsetTimes), numel(ev.imageOffsetTimes)]);
    ev.imageOnsetTimes = ev.imageOnsetTimes(1:trialNum);
    ev.imageOffsetTimes = ev.imageOffsetTimes(1:trialNum);
    ev.imageIDs = ev.imageIDs(1:trialNum);

end