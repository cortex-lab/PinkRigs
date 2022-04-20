function ev = AVprotocol2(timeline, block, alignmentBlock)
%% A helper function for multisensoySpaceWorld experimental definition that produces standardised files with useful structures for further analysis.
% OUTPUTS
% "ev" is the new, compact, and restructured block file with the following fields:
    %.performanceAVM--------Performance of mouse (%) on auditory, visual, and multisensory trials
    
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
        
    %.outcome---------------Structure containing information about the outcome of each trial
        %.timeToWheelMove------Time between the stimulus onset and significant wheel movement
        %.responseRecorded---------[on off] times for the laser, relative to trial start.       
        
%% Convert to shorter names for ease of use later
e = block.events;                     %Event structure
v = block.paramsValues;  %Parameter values at start of trial
vIdx = e.repeatNumValues(1:length(e.endTrialTimes))==1;            %Indices of valid trials (0 for repeats)

%% The number of repeats and timeouts for each trial type presented
%Invalidate trials that are repeats following an incorrect choice (because the mouse knows which way to go based on the incorrect choice) and trials
%where there were multiple repeats because of a timeout (i.e. only the initial timeout response is counted as valid)
for i = find(vIdx)
    if i < length(vIdx) && e.responseTypeValues(i) == 0
        nextResponse = min([i+find(e.responseTypeValues(i+1:length(vIdx))~=0 | e.repeatNumValues(i+1:length(vIdx))==1,1), length(vIdx)+1]);
        if e.repeatNumValues(nextResponse)==1 || nextResponse >= length(vIdx); continue; end
        vIdx(nextResponse) = 1;
    end
end

%% Extract meaningful data from the block file
%eIdx is just an logical for all trials that ended (if the experiment ends mid-trial, there may be an extra index for some events)
eIdx = 1:length(e.endTrialTimes);
vIdx = vIdx(eIdx);

if isfield(v, 'audAmplitude')
    audAmplitude = [v(eIdx).audAmplitude]';               %Convert amplitudes to matrix. Assumes one value for each trial.
    visContrast = [v(eIdx).visContrast]';                 %Convert amplitudes to matrix. Assumes one value for each trial.
    correctResponse = [v(eIdx).correctResponse]';         %Convert correctResponse on each trial to matrix. Assumes one value for each trial.
    audInitialAzimuth = [v(eIdx).audInitialAzimuth]';     %Convert audInitialAzimuth on each trial to matrix. Assumes one value for each trial.
    visInitialAzimuth = [v(eIdx).visInitialAzimuth]';     %Convert visInitialAzimuth on each trial to matrix. Assumes one value for each trial.
    clickRate = block.paramsValues.clickRate;
    clickDuration = block.paramsValues.clickDuration;
else
    audAmplitude = e.audAmplitudeValues(eIdx)';               %Convert amplitudes to matrix. Assumes one value for each trial.
    visContrast = e.visContrastValues(eIdx)';                 %Convert contrast to matrix. Assumes one value for each trial.
    correctResponse = e.correctResponseValues(eIdx)';         %Convert correctResponse on each trial to matrix. Assumes one value for each trial.
    audInitialAzimuth = e.audInitialAzimuthValues(eIdx)';     %Convert audInitialAzimuth on each trial to matrix. Assumes one value for each trial.
    visInitialAzimuth = e.visInitialAzimuthValues(eIdx)';     %Convert visInitialAzimuth on each trial to matrix. Assumes one value for each trial.
    clickRate = block.events.selected_paramsetValues.clickRate;
    clickDuration = block.events.selected_paramsetValues.clickDuration;
end
audInitialAzimuth(audAmplitude==0) = inf;             %Change case when audAmplitude was 0 to have infinite azimuth (an indication of no azimuth value)
visInitialAzimuth(visContrast==0) = inf;              %Change case when visContrast was 0 to have infinite azimuth (an indication of no azimuth value)


%Get trial start/end times, stim start times, closed loop start times, feedback times, etc.
trialTimes = [e.newTrialTimes(eIdx)' e.endTrialTimes(eIdx)'];
stimPeriodStart = e.stimPeriodOnOffTimes(e.stimPeriodOnOffValues == 1)'; 
stimPeriodStart = stimPeriodStart(eIdx);
feedbackTimes = e.feedbackTimes(eIdx)';
feedbackValues = e.feedbackValues(eIdx)';
timeOuts = feedbackValues==0;
timeToFeedback = feedbackTimes-stimPeriodStart;

%%
%Calculate an approximate time to the first wheel movement. This is different from the "timeToFeedback" in that it is based on wheel movement, rather
%than the time when the threshold was reached. WheelMove is an interpolation of the wheel movement (to get it's nearest position at every ms).

%Define a sample rate (sR--used the same as timeline) and resample wheelValues at that rate using 'pchip' and 'extrap' to get rawWheel. Get the 
%indices for stimOnset and feedback based on event times and sR.  
% PIP SHOULD WRITE A FUNCTION TO DO THIS...

%Get the response the mouse made on each trial based on the correct response and then taking the opposite for incorrect trials. NOTE: this will not
%work for a task with more than two response options.
responseRecorded = double(correctResponse).*~timeOuts;
responseRecorded(feedbackValues<0) = -1*(responseRecorded(feedbackValues<0));
responseRecorded = ((responseRecorded>0)+1).*(responseRecorded~=0);
correctResponse = ((correctResponse>0)+1).*(correctResponse~=0);

% good sanity check to keep if mean(responseCalc(~isnan(responseCalc)) == responseRecorded(~isnan(responseCalc))) < 0.50 && sum(~isnan(responseCalc)) >= 50
%     warning('Why are most of the movements not in the same direction as the response?!?');
%     keyboard;
% end

%Create a "logical" for each trial type (blank, auditory, visual, coherent, and incoherent trials)
trialType.blank = (visContrast==0 | visInitialAzimuth==0) & (audAmplitude==0 | audInitialAzimuth==0);
trialType.auditory = (visContrast==0 | visInitialAzimuth==0) & (audAmplitude>0 & audInitialAzimuth~=0);
trialType.visual = (audAmplitude==0 | audInitialAzimuth==0) & (visContrast>0 & visInitialAzimuth~=0);
trialType.coherent = sign(visInitialAzimuth.*audInitialAzimuth)>0 & audAmplitude>0 & visContrast>0;
trialType.conflict = sign(visInitialAzimuth.*audInitialAzimuth)<0 & audAmplitude>0 & visContrast>0;
trialType.repeatNum = e.repeatNumValues(1:length(vIdx))';

audPerformance = round(mean(feedbackValues(trialType.auditory & responseRecorded~=0 & vIdx(:))>0)*100);
visPerformance = round(mean(feedbackValues(trialType.visual & responseRecorded~=0 & vIdx(:))>0)*100);
mulPerformance = round(mean(feedbackValues(trialType.coherent & responseRecorded~=0 & vIdx(:))>0)*100);

audDiff = audInitialAzimuth.*audAmplitude>0;
visDiff = sign(visInitialAzimuth).*visContrast;
visDiff(visContrast==0) = 0;

%% Info from timeline!! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
timelineTime = timeline.rawDAQTimestamps;             %Timestamps in the timeline file
sR = 1/diff(timeline.rawDAQTimestamps(1:2));          %Timeline sample rate

trialStTimes = preproc.align.event2Timeline(block.events.newTrialTimes, ...
    alignmentBlock.originTimes,alignmentBlock.timelineTimes);
trialEnTimes = preproc.align.event2Timeline(block.events.endTrialTimes, ...
    alignmentBlock.originTimes,alignmentBlock.timelineTimes);
trialStEnTimes = [trialStTimes(eIdx)' trialEnTimes(eIdx)'];

stimStartBlock = preproc.align.event2Timeline(block.events.stimPeriodOnOffTimes, ...
    alignmentBlock.originTimes,alignmentBlock.timelineTimes);
stimStartBlock = stimStartBlock(1:2:end);
stimStartBlock = stimStartBlock(eIdx)';

trialGapThresh = 1;
%% Reward times using standard code
tExt.rewardTimes = timeproc.getChanEventTime(timeline, 'rewardEcho')';

%% Extract audio clicks (these are pretty reliable, so can extract every click)
%Detrend timeline trace, threshold using kmeans, detect onsets and offsets of sound, estimate duration from this.
audTrace = timeproc.extractChan(timeline,'audioOut');
audTrace = [0;diff(detrend(audTrace))];
[~, thresh] = kmeans(audTrace(1:5:end),5);
timelineClickOn = timelineTime(strfind((audTrace>max(thresh)*0.2)', [0 1]));
timelineClickOff = timelineTime(strfind((audTrace<min(thresh)*0.2)', [0 1]));
detectedDuration = round(mean(timelineClickOff-timelineClickOn)*1000);

%Sanity check: same number of onsets and offsets, check that detected duration matches the duration parameter (assumed constant here)
if length(timelineClickOn)~=length(timelineClickOff); error('There should always be an equal number on/off signals for clicks'); end
if abs(detectedDuration-(unique(clickDuration)*1000))>3; error('Diff in detected and requested click durations'); end

%Create vector that is sorted by time: [onset time, offset time, 1, 0] and find large gaps between successive onsets (stimulus period onsets)
aStimOnOffTV = sortrows([[timelineClickOn';timelineClickOff'] [timelineClickOn'*0+1; timelineClickOff'*0]],1);
largeAudGaps = sort([find(diff([0; aStimOnOffTV(:,1)])>trialGapThresh); find(diff([aStimOnOffTV(:,1); 10e10])>trialGapThresh)]);

%%%%%%%%%%%%%%%%%%%%STILL NEEDS TO BE COMMENTED BELOW%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Sanity check (should be match between stim starts from block and from timeline)
audstimStartTimeline = aStimOnOffTV(largeAudGaps,1);
audstimStartTimeline = audstimStartTimeline(aStimOnOffTV(largeAudGaps,2)==1);
nonAudTrials = audAmplitude(eIdx) == 0; 
[compareIndex] = getNearestPoint(stimStartBlock(~nonAudTrials), audstimStartTimeline);
audError = 0;
if any(compareIndex-(1:numel(compareIndex))')
    audError = 1;
    [compareIndex] = getNearestPoint(stimStartBlock(~nonAudTrials), audstimStartTimeline(~nonAudTrials));
    if ~any(compareIndex-(1:numel(compareIndex))') && length(largeAudGaps)/2 == length(nonAudTrials)
        fprintf('WARNING: Detected that AmpAud = 0 trials still generated signals in Timeline. Will remove these \n')
        
        largeAudGaps([find(nonAudTrials)*2-1 find(nonAudTrials)*2]) = [];
        audstimStartTimeline = aStimOnOffTV(largeAudGaps,1);
        audstimStartTimeline = audstimStartTimeline(aStimOnOffTV(largeAudGaps,2)==1);
        [compareIndex] = getNearestPoint(stimStartBlock(~nonAudTrials), audstimStartTimeline);
        if ~any(compareIndex-(1:numel(compareIndex)))
            audError = 0;
            keepIdx = cell2mat(arrayfun(@(x) largeAudGaps(x):largeAudGaps(x+1), 1:2:length(largeAudGaps), 'uni', 0));
            aStimOnOffTV = aStimOnOffTV(keepIdx,:);
            largeAudGaps = sort([find(diff([0; aStimOnOffTV(:,1)])>trialGapThresh); find(diff([aStimOnOffTV(:,1); 10e10])>trialGapThresh)]);
        end
    end
end
if audError; error('Error in matching auditory stimulus start and end times \n'); end

tExt.audStimOnOff = [aStimOnOffTV(aStimOnOffTV(:,2)==1,1) aStimOnOffTV(aStimOnOffTV(:,2)==0,1)];
aStimOnOffTV = aStimOnOffTV(largeAudGaps,:);
tExt.audStimPeriodOnOff = [aStimOnOffTV(aStimOnOffTV(:,2)==1,1) aStimOnOffTV(aStimOnOffTV(:,2)==0,1)];

%% Extract visual onsets (unreliable after initial flip)
% Change visual stimulus times to timeline version
photoDiodeFlipTimes = timeproc.getChanEventTime(timeline, 'photoDiode');

trialGapThresh = 1./max(clickRate)*5;
largeVisGaps = photoDiodeFlipTimes(sort([find(diff([0; photoDiodeFlipTimes])>trialGapThresh); find(diff([photoDiodeFlipTimes; 10e10])>trialGapThresh)]));
zeroContrastTrials = visContrast(eIdx)==0;
largeGapsByTrial = arrayfun(@(x,y) largeVisGaps(largeVisGaps>x & largeVisGaps<y), trialStEnTimes(:,1), trialStEnTimes(:,2), 'uni', 0);

zeroContrastTrials(cellfun(@(x) rem(length(x),2)~=0, largeGapsByTrial)) = 1;
largeVisGaps = cell2mat(largeGapsByTrial(~zeroContrastTrials));
stimStartRef = stimStartBlock(~zeroContrastTrials);
largeVisGaps = [largeVisGaps(1:2:end) largeVisGaps(2:2:end)];
largeVisGaps(diff(largeVisGaps,[],2)<(1/2000),:) = [];

% Sanity check (should be match between stim starts from block and from timeline)
[compareIndex] = getNearestPoint(stimStartRef, largeVisGaps(:,1)');
if any(compareIndex-(1:numel(compareIndex))')
    fprintf('WARNING: problem matching visual stimulus start and end times \n');
    fprintf('Will try removing points that do not match stimulus starts \n');
    
    [~, nearestPoint] = getNearestPoint(largeVisGaps(:,1), stimStartRef);
    largeVisGaps(nearestPoint>0.75,:) = [];
    
    [compareIndex] = getNearestPoint(stimStartRef, largeVisGaps(:,1)')';
    if any(compareIndex-(1:numel(compareIndex))); error('Error in matching visual stimulus start and end times \n'); end
end

visStimPeriodOnOffValues = 0*sort(largeVisGaps(:))'+1;
visStimPeriodOnOffValues(2:2:end) = 0;
visStimPeriodOnOffTimes = sort(largeVisGaps(:))';

vStimOnOffTV = [visStimPeriodOnOffTimes' visStimPeriodOnOffValues'];
vStimOnOffTV = vStimOnOffTV(1:find(vStimOnOffTV(:,2)==0,1,'last'),:);
tExt.visStimPeriodOnOff = [vStimOnOffTV(vStimOnOffTV(:,2)==1,1) vStimOnOffTV(vStimOnOffTV(:,2)==0,1)];


%%
% Could add this in for pasive
% if any(contains(fineTune, 'flashesfine'))
%     photoFlipsByTrial = arrayfun(@(x,y) find(photoDiodeFlipTimes>=x & photoDiodeFlipTimes<=y), tExt.visStimPeriodOnOff(:,1), tExt.visStimPeriodOnOff(:,2), 'uni', 0);
%     expectedFlashTrainLength = [clickRate]'.*[block.paramsValues.responseWindow]'*2;
%     misMatchFlashtrain = expectedFlashTrainLength(~zeroContrastTrials)-cellfun(@length,photoFlipsByTrial);
%     if any(misMatchFlashtrain)
%         photoFlipsByTrial(misMatchFlashtrain~=0) = [];
%         fprintf('Warning: Removing flash times for trials that do not match predicted flash length \n');
%     end
%     
%     block.events.visStimOnOffTimes = sort(photoDiodeFlipTimes(cell2mat(photoFlipsByTrial)))';
%     block.events.visStimOnOffValues = photoDiodeFlipTimes(cell2mat(photoFlipsByTrial))'*0+1;
%     block.events.visStimOnOffValues(2:2:end) = 0;
%     vStimOnOffTV = [block.events.visStimOnOffTimes' block.events.visStimOnOffValues'];
%     tExt.visStimOnOff = [vStimOnOffTV(vStimOnOffTV(:,2)==1,1) vStimOnOffTV(vStimOnOffTV(:,2)==0,1)];
% end

%% MOVEMENT
responseMadeIdx = responseRecorded ~= 0;
timelineVisOnset = indexByTrial(trialStEnTimes, tExt.visStimPeriodOnOff(:,1), tExt.visStimPeriodOnOff(:,1));
timelineVisOnset(cellfun(@isempty, timelineVisOnset)) = deal({nan});
timelineAudOnset = indexByTrial(trialStEnTimes, tExt.audStimPeriodOnOff(:,1), tExt.audStimPeriodOnOff(:,1));
timelineAudOnset(cellfun(@isempty, timelineAudOnset)) = deal({nan});
timelineStimOnset = min(cell2mat([timelineVisOnset timelineAudOnset]), [],2, 'omitnan');

stimOnsetIdx = round(timelineStimOnset(responseMadeIdx)*sR);

wheelDeg = extractWheelDeg(timeline);
wheelVel = diff([0; wheelDeg])*sR;

sumWin = 51;
whlDecThr = round(60./block.events.selected_paramsetValues.wheelGain);
velThresh  = sR*(whlDecThr*0.01)/sumWin;

posVelScan = conv(wheelVel.*double(wheelVel>0) - double(wheelVel<0)*1e6, [ones(1,sumWin) zeros(1,sumWin-1)]./sumWin, 'same').*(wheelVel~=0);
negVelScan = conv(wheelVel.*double(wheelVel<0) + double(wheelVel>0)*1e6, [ones(1,sumWin) zeros(1,sumWin-1)]./sumWin, 'same').*(wheelVel~=0);
movingScan = smooth((posVelScan'>=velThresh) + (-1*negVelScan'>=velThresh),21);
falseIdx = (movingScan(stimOnsetIdx)~=0); %don't want trials when mouse is moving at stim onset

choiceCrsIdx = arrayfun(@(x,y) max([nan find(abs(wheelDeg(x:(x+(sR*1.5)))-wheelDeg(x))>whlDecThr,1)+x]), stimOnsetIdx);
choiceCrsIdx(falseIdx) = nan;
gdIdx = ~isnan(choiceCrsIdx);

choiceThreshTime = choiceCrsIdx/sR;
choiceThreshDirection = choiceThreshTime*nan;
choiceThreshDirection(gdIdx) = sign(wheelDeg(choiceCrsIdx(gdIdx)) - wheelDeg(choiceCrsIdx(gdIdx)-25));
choiceThreshDirection(gdIdx) = (((choiceThreshDirection(gdIdx)==-1)+1).*(abs(choiceThreshDirection(gdIdx))))';

tstWin = [zeros(1, sumWin-1), 1];
velThreshPoints = [(strfind((posVelScan'>=velThresh), tstWin)+sumWin-2) -1*(strfind((-1*negVelScan'>=velThresh), tstWin)+sumWin-2)]';

[~, srtIdx] = sort(abs(velThreshPoints));
moveOnsetIdx = abs(velThreshPoints(srtIdx));
moveOnsetSign = sign(velThreshPoints(srtIdx))';
moveOnsetDir = (((moveOnsetSign==-1)+1).*(abs(moveOnsetSign)))';

onsetTimDirByTrial = indexByTrial([stimOnsetIdx/sR trialStEnTimes(responseMadeIdx,2)], moveOnsetIdx/sR, [moveOnsetIdx/sR moveOnsetDir]);
onsetTimDirByTrial(cellfun(@isempty, onsetTimDirByTrial) | isnan(choiceCrsIdx)) = deal({[nan nan]});

%"firstMoveTimes" are the first onsets occuring after stimOnsetIdx. "largeMoveTimes" are the first onsets occuring after stimOnsetIdx that match the
%sign of the threshold crossing defined earlier. Eliminate any that are longer than 1.5s, as these would be timeouts. Also, remove onsets when the
%mouse was aready moving at the time of the stimulus onset (impossible to get an accurate movement onset time in this case)
firstMoveTimeDir = cell2mat(cellfun(@(x) x(1,:), onsetTimDirByTrial, 'uni', 0));
choiceInitTimeDir = cellfun(@(x,y) x(find(x(:,1)<y,1,'last'),:), onsetTimDirByTrial, num2cell(choiceThreshTime(:,1)), 'uni', 0);
choiceInitTimeDir(cellfun(@isempty, choiceInitTimeDir)) = {[nan nan]};
choiceInitTimeDir = cell2mat(choiceInitTimeDir);


%SANITY CHECK
blockTstValues = responseRecorded(responseMadeIdx);
tstIdx = ~isnan(choiceInitTimeDir(:,2));
if mean(choiceInitTimeDir(tstIdx,2) == blockTstValues(tstIdx)) < 0.75
    error('Why are most of the movements not in the same direction as the response?!?')
end

tExt.firstMoveTimeDir = firstMoveTimeDir;
tExt.choiceInitTimeDir = choiceInitTimeDir;
tExt.choiceThreshTimeDir = [choiceThreshTime, choiceThreshDirection];
tExt.allMovOnsetsTimDirByTrial = onsetTimDirByTrial;

changePoints = strfind(diff([0,wheelDeg'])==0, [1 0]);
trialStEnIdx = (trialStEnTimes*sR);
points2Keep = sort([1 changePoints changePoints+1 length(wheelDeg) ceil(trialStEnIdx(:,1))'+1, floor(trialStEnIdx(:,2))'-1]);
tExt.wheelTraceTimeValue = [timelineTime(points2Keep)' wheelDeg(points2Keep)];

%%
rawFields = fields(tExt);
for i = 1:length(rawFields)
    currField = rawFields{i};
    currData = tExt.(currField);
    tExt.(currField) = indexByTrial(trialStEnTimes, currData(:,1), currData);
    emptyIdx = cellfun(@isempty, tExt.(currField));

    if any(strcmp(currField, {'audStimOnOff'; 'visStimOnOff'; 'rewardTimes';'wheelTraceTimeValue'}))
        nColumns = max(cellfun(@(x) size(x,2), tExt.(currField)));
        tExt.(currField)(emptyIdx) = {nan*ones(1,nColumns)};
        tExt.(currField) = cellfun(@single,tExt.(currField), 'uni', 0);
    end
    if any(strcmp(currField, {'audStimPeriodOnOff'; 'visStimPeriodOnOff'; 'laserTTLPeriodOnOff'; 'firstMoveTimeDir'; 'choiceInitTimeDir'; 'choiceThreshTimeDir'}))
        nColumns = max(cellfun(@(x) size(x,2), tExt.(currField)));
        tExt.(currField)(emptyIdx) = {nan*ones(1, nColumns)};
        tExt.(currField) = single(cell2mat(tExt.(currField)));
    end
end

%% Populate n with all fields;
ev.performanceAVM = [audPerformance visPerformance mulPerformance];
ev.trialType = trialType; 
ev.trialType.validTrial = vIdx(:);
ev.timings.trialStartEnd = trialTimes;
ev.timings.stimPeriodStart = stimPeriodStart;
ev.timeline = tExt;
ev.stim.correctResponse = correctResponse;
ev.stim.audAmplitude = audAmplitude;
ev.stim.audInitialAzimuth = audInitialAzimuth;
ev.stim.audDiff = audDiff;
ev.stim.visContrast = visContrast;
ev.stim.visInitialAzimuth = visInitialAzimuth;
ev.stim.visDiff = visDiff;
ev.outcome.responseRecorded = responseRecorded;
ev.outcome.feedbackGiven = feedbackValues;
ev.outcome.timeToFeedback = timeToFeedback;

