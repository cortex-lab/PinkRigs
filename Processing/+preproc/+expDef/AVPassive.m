function ev = AVPassive(timeline, block, alignmentBlock)
    %% Fetches all important information from the AV passive protocols
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
    %   FOR TIMES: all relative to trial start
    %       ev.is_blankTrial      %logical: indicating "blank" trials
    %       ev.is_visualTrial     %logical: indicating "visual" trials
    %       ev.is_auditoryTrial   %logical: indicating "auditory" trials
    %       ev.is_coherentTrial   %logical: indicating "coherent" trials
    %       ev.is_conflictTrial   %logical: indicating "conflict" trials
    %       ev.is_rewardTrial   %logical: indicating "reward click" trials
    %
    %       ev.block_trialOnOff   %nx2 matrix: [starttime endtime]
    %       ev.block_stimOn       %nx1 matrix: start times for stimulus period
    %
    %       ev.timeline_rewardOn  %nx1 matrix: reward times
    %       ev.timeline_audOnOff  %nx8 matrix: [on off] times for aud stimuli (each click)
    %       ev.timeline_visOnOff  %nx8 matrix: [on off] times for vis stimuli (each flash)
    %
    %       ev.stim_audAmplitude        %nx1 matrix: aud amplitude
    %       ev.stim_visContrast         %nx1 matrix: vis contrast
    %       ev.stim_audAzimuth          %nx1 matrix: aud azimuth presented
    %       ev.stim_visAzimuth          %nx1 matrix: vis azimuth presented

    eIdx = 1:numel(block.events.endTrialValues);
    sR = 1/diff(timeline.rawDAQTimestamps(1:2));          %Timeline sample rate

    % get stim info
    visTrial = block.events.viscontrastValues(eIdx) > 0;
    visTrialsLoc = block.events.visazimuthValues(eIdx); visTrialsLoc(~visTrial) = nan;
    audTrial = block.events.audamplitudeValues(eIdx) > 0;
    audTrialsLoc = block.events.audazimuthValues(eIdx); audTrialsLoc(~audTrial) = nan;
    rewTrials = block.outputs.rewardValues(eIdx)>0;

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

    if length(trialStart)~=length(trialEnd)
        trialStart = trialStart(1:end-1);
        if length(trialStart)~=length(trialEnd)
            error('Discrepancy between the number of started vs. ended trials. Have a look.')
        end
    end
    trialStEnTimes = [trialStart(:) trialEnd(:)];
    timelineTime = timeproc.extractChan(timeline,'time');

    % Add delay to trial start and end because of issues with alignment?
    % It's a bit of a hacky thing to do.
    delay = 0.2;

    %% visual stimulus timings
    % get all screen flips
    [photoDiodeFlipTimes, photoName] = timeproc.extractBestPhotodiode(timeline, block);
    fprintf('****Using %s channel for photodiode...\n', photoName);

    % sort by trial
    p = block.paramsValues(1);
    numClicks = numel((p.clickDuration/2):1/p.clickRate:p.stimDuration);
    visOnOffByTrial = indexByTrial(trialStEnTimes+delay, photoDiodeFlipTimes');
    vis2Remove = cellfun(@(x) length(x)~=numClicks*2, visOnOffByTrial);
    visOnOffByTrial(vis2Remove)= deal({nan*ones(1, 2)});
    visOnOffByTrial = cellfun(@(x) [x(1:2:end) x(2:2:end)], visOnOffByTrial, 'uni', 0);
    visPeriodOnOff = cellfun(@(x) [x(1,1) x(end,2)], visOnOffByTrial, 'uni', 0);

    if sum(~vis2Remove) < 0.9*sum(visTrial)
        fprintf('****Removing more than 10 percent of visual trials..? \n')
    end
    if sum(~vis2Remove) < 0.5*sum(visTrial)
        warning('Removing more than 50% of visual trials..?!!')
    end

    %% auditory click times
    audTrace = timeproc.extractChan(timeline,'audioOut');
    audTrace = [0;diff(detrend(audTrace))];
    [~, thresh] = kmeans(audTrace(1:5:end),5);
    timelineClickOn = timelineTime(strfind((audTrace>max(thresh)*0.2)', [0 1]))';
    timelineClickOff = timelineTime(strfind((audTrace<min(thresh)*0.2)', [0 1]))';

    if length(timelineClickOn)~=length(timelineClickOff)
        error('There should always be an equal number on/off signals for clicks');
    end
    allClicks = sort([timelineClickOn timelineClickOff]);
    audOnOffByTrial = indexByTrial(trialStEnTimes+delay, allClicks(:,1), allClicks);
    aud2Remove = cellfun(@(x) size(x,1)~=numClicks, audOnOffByTrial);
    audOnOffByTrial(aud2Remove)= deal({nan*ones(1, 2)});
    audPeriodOnOff = cellfun(@(x) [x(1,1) x(end,2)], audOnOffByTrial, 'uni', 0);

    if sum(~aud2Remove) < 0.9*sum(audTrial)
        warning('Removing more than 10% of auditory trials..?')
    end
    if sum(~aud2Remove) < 0.5*sum(audTrial)
        error('Removing more than 50% of auditory trials..?!!')
    end

    %% reward times
    reward = timeproc.getChanEventTime(timeline,'rewardEcho');
    rewardAll = indexByTrial(trialStEnTimes+delay, reward(:));
    rew2Remove = cellfun(@(x) length(x)~=1, rewardAll);
    rewardAll(rew2Remove) = deal({nan});

    %% MOVEMENT
    wheelDeg = extractWheelDeg(timeline);
    wheelVel = diff([0; wheelDeg])*sR;

    sumWin = 51;
    wg = 3; %% THIS IS HARDCODED AS IT GIVES SAME THRESHOLDS AS FINAL MOVMENT STAGE
    whlDecThr = round(60./wg(1));
    velThresh  = sR*(whlDecThr*0.01)/sumWin;

    posVelScan = conv(wheelVel.*double(wheelVel>0) - double(wheelVel<0)*1e6, [ones(1,sumWin) zeros(1,sumWin-1)]./sumWin, 'same').*(wheelVel~=0);
    negVelScan = conv(wheelVel.*double(wheelVel<0) + double(wheelVel>0)*1e6, [ones(1,sumWin) zeros(1,sumWin-1)]./sumWin, 'same').*(wheelVel~=0);
    tstWin = [zeros(1, sumWin-1), 1];
    velThreshPoints = [(strfind((posVelScan'>=velThresh), tstWin)+sumWin-2) -1*(strfind((-1*negVelScan'>=velThresh), tstWin)+sumWin-2)]';

    [~, srtIdx] = sort(abs(velThreshPoints));
    moveOnsetIdx = abs(velThreshPoints(srtIdx));
    moveOnsetSign = sign(velThreshPoints(srtIdx))';
    moveOnsetDir = (((moveOnsetSign==-1)+1).*(abs(moveOnsetSign)))';

    if ~isempty(moveOnsetIdx)
        onsetTimDirByTrial = indexByTrial(trialStEnTimes, moveOnsetIdx/sR, [moveOnsetIdx/sR moveOnsetDir]);
        onsetTimDirByTrial(cellfun(@isempty, onsetTimDirByTrial)) = deal({[nan nan]});
    else
        onsetTimDirByTrial = repmat({[nan, nan]}, size(trialStEnTimes,1),1);
    end

    changePoints = strfind(diff([0,wheelDeg'])==0, [1 0]);
    trialStEnIdx = (trialStEnTimes*sR);
    points2Keep = sort([1 changePoints changePoints+1 length(wheelDeg) ceil(trialStEnIdx(:,1))'+1, floor(trialStEnIdx(:,2))'-1]);
    points2Keep(points2Keep > length(wheelDeg)) = [];
    wheelTraceTimeValue = [timelineTime(points2Keep)' wheelDeg(points2Keep)];
    wheelTraceTimeValueByTrial = indexByTrial(trialStEnTimes, wheelTraceTimeValue(:,1), wheelTraceTimeValue);
    wheelTraceTimeValueByTrial(cellfun(@isempty, wheelTraceTimeValueByTrial)) = deal({[nan nan]});

    %% save it in ev

    ev.is_blankTrial = blankTrials';
    ev.is_visualTrial = visOnlyTrials';
    ev.is_auditoryTrial = audOnlyTrials';
    ev.is_coherentTrial = coherentTrials';
    ev.is_conflictTrial = conflictTrials';
    ev.is_rewardTrial = rewTrials';

    ev.block_trialOn = single(trialStart');
    ev.block_trialOff = single(trialEnd');
    ev.block_stimOn  = single(stimOnsetRaw(eIdx)');

    ev.timeline_rewardOn = single(cell2mat(rewardAll));
    ev.timeline_audOn = cellfun(@(x) single(x(:,1)), audOnOffByTrial, 'uni', 0);
    ev.timeline_audOff = cellfun(@(x) single(x(:,2)), audOnOffByTrial, 'uni', 0);
    ev.timeline_visOn = cellfun(@(x) single(x(:,1)), visOnOffByTrial, 'uni', 0);
    ev.timeline_visOff = cellfun(@(x) single(x(:,2)), visOnOffByTrial, 'uni', 0);
    ev.timeline_allMoveOn = cellfun(@(x) x(:,1), onsetTimDirByTrial, 'uni', 0);
    ev.timeline_allMoveDir  = cellfun(@(x) x(:,2), onsetTimDirByTrial, 'uni', 0);
    ev.timeline_wheelTime  = cellfun(@(x) x(:,1), wheelTraceTimeValueByTrial, 'uni', 0);
    ev.timeline_wheelValue  = cellfun(@(x) x(:,2), wheelTraceTimeValueByTrial, 'uni', 0);

    audPeriodOnOff = cell2mat(audPeriodOnOff);
    ev.timeline_audPeriodOn = single(audPeriodOnOff(:,1));
    ev.timeline_audPeriodOff = single(audPeriodOnOff(:,2));

    visPeriodOnOff = cell2mat(visPeriodOnOff);
    ev.timeline_visPeriodOn = single(visPeriodOnOff(:,1));
    ev.timeline_visPeriodOff = single(visPeriodOnOff(:,2));

    ev.stim_audAmplitude = single(block.events.audamplitudeValues(eIdx)');
    ev.stim_visContrast = single(block.events.viscontrastValues(eIdx)');
    ev.stim_audAzimuth  = single(audTrialsLoc(eIdx)');
    ev.stim_visAzimuth  = single(visTrialsLoc(eIdx)');


    if length(unique(structfun(@length, ev)))~=1
        error('All fields of ev should have size [trials] on dim 1')
    end
end
