function ev = AVprotocol(timeline, block, alignmentBlock)
    %%% This function will fetch all important information from the AV
    %%% protocols, during postactive or during training.

    % get stim info
    visTrial = block.events.viscontrastValues(1:numel(block.events.endTrialValues)) > 0;
    visTrialsLoc = block.events.visazimuthValues(1:numel(block.events.endTrialValues)); visTrialsLoc(~visTrial) = nan;
    audTrial = block.events.audamplitudeValues(1:numel(block.events.endTrialValues)) > 0;
    audTrialsLoc = block.events.audazimuthValues(1:numel(block.events.endTrialValues)); audTrialsLoc(~audTrial) = nan;
    rewTrials = block.outputs.rewardValues(1:numel(block.events.endTrialValues))>0;
    
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
    [visOnsetAll,visOffsetAll] = sortClicksByTrial(photoDiodeFlipTimes,trialStart,trialEnd,numClicks,1);

    
    
    %% auditory click times 
    audTrace = timeproc.extractChan(timeline,'audioOut');
    timelineTime = timeproc.extractChan(timeline,'time');
    audTrace = [0;diff(detrend(audTrace))];
    [~, thresh] = kmeans(audTrace(1:5:end),5);
    timelineClickOn = timelineTime(strfind((audTrace>max(thresh)*0.2)', [0 1]));
    timelineClickOff = timelineTime(strfind((audTrace<min(thresh)*0.2)', [0 1]));
    ClickTimes = sort([timelineClickOn timelineClickOff]);
    
    [audOnsetAll,audOffsetAll] = sortClicksByTrial(ClickTimes,trialStart,trialEnd,numClicks,1); 
    
    
    %% reward times 
    reward = timeproc.getChanEventTime(timeline,'rewardEcho'); 
    [rewardAll,~] = sortClicksByTrial(reward,trialStart,trialEnd,1,0);
    
    %% save it in ev
    ev.visStimOnset = visOnsetAll;
    ev.visStimOffset = visOffsetAll;
    ev.audStimOnset = audOnsetAll;
    ev.audStimOffset = audOffsetAll; 
    ev.rewardTimes = rewardAll; 
    ev.visOnBlock = stimOnsetRaw;
    ev.trialInfo.visTrial = visTrial;
    ev.trialInfo.visTrialsLoc = visTrialsLoc;
    ev.trialInfo.audTrial = audTrial;
    ev.trialInfo.audTrialsLoc = audTrialsLoc;
    ev.trialInfo.rewTrials = rewTrials;
    ev.trialInfo.visContrast=block.events.viscontrastValues;
    ev.trialInfo.Loudness = block.events.audamplitudeValues; 
end 

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
        if (myTrial==1) && (numel(evPerTrial)>numClicks*2)
            evPerTrial(1)=[];  % empirical that sometimes  the screen does weird stuff on the 1st trial                
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
    