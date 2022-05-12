function ev = AVPassive(timeline, block, alignmentBlock)
%%% This function will fetch all important information from the AV
    %%% protocols, during postactive or during training.

    % Returns:  
    % ev.is_blankTrial      %logical: indicating "blank" trials
    % ev.is_visualTrial     %logical: indicating "visual" trials
    % ev.is_auditoryTrial   %logical: indicating "auditory" trials
    % ev.is_coherentTrial   %logical: indicating "coherent" trials
    % ev.is_conflictTrial   %logical: indicating "conflict" trials
    % ev.is_rewardTrial   %logical: indicating "reward click" trials
    % 
    % ev.block_trialOnOff   %nx2 matrix: [starttime endtime]
    % ev.block_stimOn       %nx1 matrix: start times for stimulus period
    % 
    % ev.timeline_rewardOn  %nx1 matrix: reward times 
    % ev.timeline_audOnOff  %nx8 matrix: [on off] times for aud stimuli (each click)
    % ev.timeline_visOnOff  %nx8 matrix: [on off] times for vis stimuli (each flash)
    % 
    % ev.stim_audAmplitude        %nx1 matrix: aud amplitude
    % ev.stim_visContrast         %nx1 matrix: vis contrast
    % ev.stim_audAzimuth          %nx1 matrix: aud azimuth presented
    % ev.stim_visAzimuth          %nx1 matrix: vis azimuth presented   


    % get stim info
    visTrial = block.events.viscontrastValues(1:numel(block.events.endTrialValues)) > 0;
    visTrialsLoc = block.events.visazimuthValues(1:numel(block.events.endTrialValues)); visTrialsLoc(~visTrial) = nan;
    audTrial = block.events.audamplitudeValues(1:numel(block.events.endTrialValues)) > 0;
    audTrialsLoc = block.events.audazimuthValues(1:numel(block.events.endTrialValues)); audTrialsLoc(~audTrial) = nan;
    rewTrials = block.outputs.rewardValues(1:numel(block.events.endTrialValues))>0;
    
    % get trial types, Dr Coen scheme 
    blankTrials  = ((isnan(visTrialsLoc)) & (isnan(audTrialsLoc)) & (rewTrials==0));
    visOnlyTrials =(visTrial==1) & (audTrial==0);
    audOnlyTrials =(visTrial==0) & (audTrial==1);
    coherentTrials = (visTrial==1) & (audTrial==1) & (visTrialsLoc==audTrialsLoc);
    conflictTrials = (visTrial==1) & (audTrial==1) & (visTrialsLoc~=audTrialsLoc);
    
    % get timing for blanks 
    stimOnsetRaw = preproc.align.event2Timeline(block.events.visstimONTimes, ...
        alignmentBlock.originTimes,alignmentBlock.timelineTimes);
    
    % get timings of trials
    trialStart = preproc.align.event2Timeline(block.events.newTrialTimes, ...
        alignmentBlock.originTimes,alignmentBlock.timelineTimes);
    
    trialEnd = preproc.align.event2Timeline(block.events.endTrialTimes, ...
        alignmentBlock.originTimes,alignmentBlock.timelineTimes);
    
    %% visual stimulus timings 
    
    % get all screen flips 
    photoDiodeFlipTimes = timeproc.getChanEventTime(timeline, 'photoDiode'); 
    
    % sort by trial
    p = block.paramsValues(1); 
    numClicks = numel((p.clickDuration/2):1/p.clickRate:p.stimDuration); 
    visOnOff = sortClicksByTrial(photoDiodeFlipTimes,trialStart,trialEnd,numClicks*2,0);

    
    
    %% auditory click times 
    audTrace = timeproc.extractChan(timeline,'audioOut');
    timelineTime = timeproc.extractChan(timeline,'time');
    audTrace = [0;diff(detrend(audTrace))];
    [~, thresh] = kmeans(audTrace(1:5:end),5);
    timelineClickOn = timelineTime(strfind((audTrace>max(thresh)*0.2)', [0 1]));
    timelineClickOff = timelineTime(strfind((audTrace<min(thresh)*0.2)', [0 1]));
    ClickTimes = sort([timelineClickOn timelineClickOff]);
    
    audOnOff = sortClicksByTrial(ClickTimes,trialStart,trialEnd,numClicks*2,0); 
    
    
    %% reward times 
    reward = timeproc.getChanEventTime(timeline,'rewardEcho'); 
    [rewardAll,~] = sortClicksByTrial(reward,trialStart,trialEnd,1,0);
    
    %% save it in ev
 
    ev.is_blankTrial = blankTrials';    
    ev.is_visualTrial = visOnlyTrials'; 
    ev.is_auditoryTrial = audOnlyTrials'; 
    ev.is_coherentTrial = coherentTrials'; 
    ev.is_conflictTrial = conflictTrials'; 
    ev.is_rewardTrial = rewTrials';

    ev.block_trialOnOff = [trialStart' trialEnd']; 
    ev.block_stimOn  = stimOnsetRaw'; 

    ev.timeline_rewardOn = rewardAll'; 
    ev.timeline_audOnOff = audOnOff';  
    ev.timeline_visOnOff = visOnOff';  
    
    ev.timeline_audPeriodOnOff = audOnOff([1 numClicks*2],:)';
    ev.timeline_visPeriodOnOff = visOnOff([1 numClicks*2],:)';

    ev.stim_audAmplitude = block.events.audamplitudeValues';
    ev.stim_visContrast = block.events.viscontrastValues';
    ev.stim_audAzimuth  = audTrialsLoc';
    ev.stim_visAzimuth  = visTrialsLoc';

%%

function [OnsetAll,OffsetAll] = sortClicksByTrial(eventTimes,trialStart,trialEnd,numClicks,sortOnOff)
% sort events by trial 
% can be single events (sortOnOff=0) 
% or event pairs (sortOnOff=1)
% numClicks = expected events per trial  (% todo: make this more flexible?)
% 

TimesPerTrial = arrayfun(@(x,y) eventTimes((eventTimes>=x)& ...
        (eventTimes<=y)),trialStart,trialEnd,'UniformOutput',false);
nTrials = numel(TimesPerTrial); 
OnsetAll = nan(numClicks,nTrials);
OffsetAll = nan(numClicks,nTrials);

for myTrial=1:nTrials
    evPerTrial = TimesPerTrial{myTrial};
    if numel(evPerTrial)>0
        if (myTrial==1) && (numel(evPerTrial)>numClicks)
            evPerTrial(1:numel(evPerTrial)-numClicks)=[];  % empirical that sometimes  the screen does weird stuff on the 1st trial
        end
        % can sort on and offsets or just get all
        if sortOnOff==1
            OnsetAll(:,myTrial) = evPerTrial(1:2:end);
            OffsetAll(:,myTrial) = evPerTrial(2:2:end);
        else
            OnsetAll(:,myTrial) = evPerTrial;
        end
    end
end 
end
end
