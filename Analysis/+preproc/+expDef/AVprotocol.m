function ev = AVprotocol(timeline, block, alignmentBlock)
    %%% This function will fetch all important information from the AV
    %%% protocols, during postactive or during training.
    
    %     %% Extract photodiode onsets in timeline
    %     % might need to clean a bit?
    %
    %     ev.visStimOnsetTime = timeproc.getChanEventTime(timeline,'photoDiode');
    %
    %     %% Extract sounds onsets
    %     % might need to clean a bit?
    %
    %     ev.audStimOnsetTime = timeproc.getChanEventTime(timeline,'audio');
    %
    %     %% Extract reward onsets
    %
    %     ev.rewardOnsetTimes = timeproc.getChanEventTime(timeline,'reward');
    %
    %     %% Extract trial info
    
    % QUICK AND DIRTY FOR NOW -- NEED TO CHANGE AND RECOMPUTE
    % get stim info
    visTrial = block.events.viscontrastValues > 0;
    visTrialsLoc = block.events.visazimuthValues; visTrialsLoc(~visTrial) = nan;
    audTrial = block.events.audamplitudeValues > 0;
    audTrialsLoc = block.events.audazimuthValues; audTrialsLoc(~audTrial) = nan;
    rewTrials = block.outputs.rewardValues>0;
    
    % get stim onsets
    t = timeproc.extractChan(timeline,'time');
    pho = timeproc.extractChan(timeline,'photoDiode');
    aud = timeproc.extractChan(timeline,'audioOut');
    
    tmp = block.stimWindowUpdateTimes(2:2:end);
    [~, phoFlips, ~] = schmittTimes(t, pho, [6 10]);
    if numel(tmp)>numel(phoFlips)
        tmp(1) = []; % empirical...
    end

    stimOnsetRaw = preproc.align.event2Timeline(block.events.visstimONTimes, ...
        alignmentBlock.originTimes,alignmentBlock.timelineTimes);
    
    % get it from photodiode?
    [~, audOnset, ~] = schmittTimes(t, aud, max(aud)*[0.5 0.8]);
    audOnset(find(diff(audOnset)<0.5)+1) = [];
    % audOnset = t(find(diff(aud)>1)+1);
    % audOnset(find(diff(audOnset)<1)+1) = [];
    [~, visOnset, ~] = schmittTimes(t, pho, max(pho)*[0.5 0.8]);
    visOnset(1) = []; % looks like something else
    visOnset(find(diff(visOnset)<0.5)+1) = [];
    nTrials = numel(block.events.stimuliOnTimes);
    visOnsetAll = nan(1,nTrials);
    audOnsetAll = nan(1,nTrials);
    idxv = 0;
    idxa = 0;
    for tt = 1:nTrials
        if visTrial(tt)
            idxv = idxv+1;
            visOnsetAll(tt) = visOnset(idxv);
        end
        if audTrial(tt)
            idxa = idxa+1;
            audOnsetAll(tt) = audOnset(idxa);
        end
    end
    stimOnset = min([visOnsetAll; audOnsetAll]); % NOT IDEAL AT ALL, TAKES THE FIRST ONE THAT COMES // COULD ALSO TAKE stimOnsetRaw
    stimOnset(audTrial) = audOnset; % force it to be aligned on auditory onsets for now, because that's what we're interested in...
    stimOnset(isnan(stimOnset)) = stimOnsetRaw(isnan(stimOnset));
    
    %% save it in ev
    ev.stimOnset = stimOnset;
    ev.trialInfo.visTrial = visTrial;
    ev.trialInfo.visTrialsLoc = visTrialsLoc;
    ev.trialInfo.audTrial = audTrial;
    ev.trialInfo.audTrialsLoc = audTrialsLoc;
    ev.trialInfo.rewTrials = rewTrials;