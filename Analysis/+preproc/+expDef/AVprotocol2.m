function ev = AVprotocol2(timeline, block, alignmentBlock)
%% A helper function for multisensoySpaceWorld experimental definition that produces standardised files with useful structures for further analysis.
% OUTPUTS
% "ev" is the new, compact, and restructured block file with the following fields:
    %.subject---------------Name of the mouse
    %.expDate---------------Date that the experiment was recorded
    %.expNum----------------Session number for experiment
    %.rigName---------------Name of the rig where the experiment took place
    %.expType---------------Type experiment ('training, inactivation, or ephys')
    %.expDef----------------Name of the expDef file
    %.performanceAVM--------Performance of mouse (%) on auditory, visual, and multisensory trials
    %.conditionParametersAV-[visDiff, audDiff], one row for each condition in the parameter set for the session
    
    %.trialType------------Structure with logical fields to identifying each class of trial
        %.blank----------------Trials with auditory in the center and visual contrast is zero
        %.auditory-------------Trials with auditory on left/right and visual contrast is zero
        %.visual---------------Trials with auditory in the center and visual contrast is not zero
        %.coherent-------------Trials where the auditory and visual stimuli agree (and are non-zero)
        %.conflict-------------Trials with the auditory and visual stimuli disagree (and are non-zero)
        
    %.timings---------------Structure containing timings for events on each trial
        %.trialStartEnd--------[start end] times for whole trial
        %.stimPeriodStart------Time of stimulus onset (aud and vis start at the same time)
        %.closedLoopStart------Time of closed loop initiation (typically 500ms after stimulus onset)
   
    %.stim------------------Structure containing information about the stimulus presented on each trial
        %.audAmplitude---------Aud amplitude (arbitrary number really, only ~=0 matters)
        %.audInitialAzimuth----Initial azimuthal location of auditory stimulus (+/- is right/left of mouse). Inf if not present.
        %.audDiff--------------Difference in initial azimuthal location (identical to audInitialAzimuth in most cases)
        %.visContrast----------Absolute visual contrast (+ for left and right). Ranges from 0 to 1
        %.visInitialAzimuth----Initial azimuthal location of visual stimulus (+/- is right/left of mouse). Inf if not present.
        %.visDiff--------------Difference in left/right contrast (+/- is right/left of mouse)
        %.conditionLabel-------The integer label fpr the condition being presented.
        
    %.inactivation----------Structure containing information about the inactivation paramters for each trial
        %.laserType------------Laser type, which can be off (0), unilateral (1), or bilateral (2)
        %.laserPower-----------Laser power (represents mW in more recent experiments)
        %.galvoType------------Galvo movement type, which can be stationary (1) or moving between two sites (2) while the laser is on
        %.laserOnsetDelay------Delay between laser onset and stimulus onset
        %.galvoPosition--------Galvo coordinates used on each trial ([LM axis, AP axis], +/- is right/left hemisphere)
        %.laserOnOff-----------[on off] times for the laser, relative to trial start.

    %.outcome---------------Structure containing information about the outcome of each trial
        %.timeToWheelMove------Time between the stimulus onset and significant wheel movement
        %.responseRecorded---------[on off] times for the laser, relative to trial start.       
        
    %.params----------------The parameters from signals. Basically never used, but saved for completion.
        

% "x.newRaw" is a structure comprising potentially useful raw data (such as wheel movement and timeline data) which is not used for a lot of analyses and
% so should only be loaded if necessary (as it is large). Fields are:
    %.visStimOnOffTTimeValue---------cell array, each contatining an [nx2] vector of [time(s), value(on/off is 1/0)] for the visual stimulus
    %.audStimOnOffTTimeValue---------cell array, each contatining an [nx2] vector of [time(s), value(on/off is 1/0)] for the auditory stimulus
    %.wheelTimeValue-----------------wheel [times(s), position(deg)] over the course of the entire session
    %.visAzimuthTimeValue------------cell array, each contatining an [nx2] vector of [time(s), value(deg)] for the visual stimulus azimuth
    %.audAzimuthTimeValue------------cell array, each contatining an [nx2] vector of [time(s), value(deg)] for the auditory stimulus azimuth
    
    
    
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