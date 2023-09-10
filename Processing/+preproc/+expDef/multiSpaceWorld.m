function ev = multiSpaceWorld(~, block, ~)
%% Fetches all important information from the imageWorld protocols
%
% Parameters:
% -------------------
% timeline: struct
%   Timeline structure. (not used in this past data)
% block: struct
%   Block structure
% alignmentBlock: struct (not used in this past data)
%   Alignement structure, containing fields "originTimes" and
%   "timelineTimes"
%
% Returns:
% -------------------
% ev: struct
%   Structure containing all relevant events information.
%   All fields should have the form [nxm] where n is the number of trials.
%   FOR TIMES: all relative to trial start
%   FOR DIRECTIONS: 2 =  rightward choice, 1 = leftward choice
%       ev.is_blankTrial      %logical: indicating "blank" trials
%       ev.is_visualTrial     %logical: indicating "visual" trials
%       ev.is_auditoryTrial   %logical: indicating "auditory" trials
%       ev.is_coherentTrial   %logical: indicating "coherent" trials
%       ev.is_conflictTrial   %logical: indicating "conflict" trials
%       ev.is_validTrial      %logical: indicating "valid" trials (used for analysis)
%
%       ev.block_trialOnOff   %nx2 matrix: [starttime endtime]
%       ev.block_stimOn       %nx1 matrix: start times for stimulus period
%
%       ev.timeline_rewardOn  %nx1 cell: reward times (manual rewards included)
%       ev.timeline_audOnOff  %nx1 cell: [on off] times for aud stimuli (each click)
%       ev.timeline_visOnOff  %nx1 cell: [on off] times for vis stimuli (each flash)
%
%       ev.timeline_audPeriodOnOff %nx2 matrix: [on off] times for the "whole" aud stimulus
%       ev.timeline_visPeriodOnOff %nx2 matrix: [on off] times for the "whole" vis stimulus
%       ev.timeline_firstMoveOn    %nx1 matrix: time for the first movement initiation
%       ev.timeline_firstMoveDir   %nx1 matrix: direction of first movement initiation
%       ev.timeline_choiceMoveOn   %nx1 matrix: time of "choice" movement initiation
%       ev.timeline_choiceMoveDir  %nx1 matrix: direction of "choice" movement
%       ev.timeline_choiceThreshOn %nx2 matrix: time that wheel crosses decision threshold
%       ev.timeline_allMoveOn      %nx1 cell:   times for all movement onsets
%       ev.timeline_allMoveDir     %nx1 cell:   direction for all movement onsets
%       ev.timeline_wheelTimeValue %nx2 cell:   [times wheelPosition(deg)]
%
%       ev.stim_correctResponse     %nx1 matrix: correct answer NOT mouse choice ev.stim_audAmplitude        %nx1 matrix: aud amplitude
%       ev.stim_audAzimuth          %nx1 matrix: aud azimuth presented
%       ev.stim_visContrast         %nx1 matrix: vis contrast
%       ev.stim_visAzimuth          %nx1 matrix: vis azimuth presented
%
%       ev.response_direction      %nx1 matrix. recorded response (1/2 for left/right)
%       ev.response_feedback       %nx1 matrix. -1/0/1 for incorrect/timeout/reward

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

audAmplitude = [v(eIdx).audAmplitude]';               %Convert amplitudes to matrix. Assumes one value for each trial.
visContrast = [v(eIdx).visContrast]';                 %Convert amplitudes to matrix. Assumes one value for each trial.
correctResponse = [v(eIdx).correctResponse]';         %Convert correctResponse on each trial to matrix. Assumes one value for each trial.
audInitialAzimuth = [v(eIdx).audInitialAzimuth]';     %Convert audInitialAzimuth on each trial to matrix. Assumes one value for each trial.
audInitialAzimuth(audAmplitude==0) = nan;             %Change case when audAmplitude was 0 to have infinite azimuth (an indication of no azimuth value)
visInitialAzimuth = [v(eIdx).visInitialAzimuth]';     %Convert visInitialAzimuth on each trial to matrix. Assumes one value for each trial.
visInitialAzimuth(visContrast==0) = nan;              %Change case when visContrast was 0 to have infinite azimuth (an indication of no azimuth value)

%Get trial start/end times, stim start times, closed loop start times, feedback times, etc.
trialTimes = [e.newTrialTimes(eIdx)' e.endTrialTimes(eIdx)'];
stimPeriodStart = e.stimPeriodOnOffTimes(e.stimPeriodOnOffValues == 1)';
stimPeriodStart = stimPeriodStart(eIdx);
feedbackValues = e.feedbackValues(eIdx)';
feedbackTimes = e.feedbackTimes(eIdx)';
timeOuts = feedbackValues==0;

%%
%Calculate an approximate time to the first wheel movement. This is different from the "timeToFeedback" in that it is based on wheel movement, rather
%than the time when the threshold was reached. WheelMove is an interpolation of the wheel movement (to get it's nearest position at every ms).

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
is_noStimTrial = isnan(visInitialAzimuth) & isnan(audInitialAzimuth);
is_blankTrial = (visContrast==0 | visInitialAzimuth==0) & (audAmplitude>0) & (audInitialAzimuth==0);
is_auditoryTrial = (visContrast==0 | visInitialAzimuth==0) & (audAmplitude>0 & audInitialAzimuth~=0);
is_visualTrial = (audAmplitude==0 | audInitialAzimuth==0) & (visContrast>0 & visInitialAzimuth~=0);
is_coherentTrial = sign(visInitialAzimuth.*audInitialAzimuth)>0 & audAmplitude>0 & visContrast>0;
is_conflictTrial = sign(visInitialAzimuth.*audInitialAzimuth)<0 & audAmplitude>0 & visContrast>0;

%% Get movement info (not from timeline)
%%
%Calculate an approximate time to the first wheel movement. This is different from the "timeToFeedback" in that it is based on wheel movement, rather
%than the time when the threshold was reached. WheelMove is an interpolation of the wheel movement (to get it's nearest position at every ms).
times2Subtract = [stimPeriodStart stimPeriodStart*0];
repeatPoints = strfind(diff([-1000,block.inputs.wheelValues])==0, [1 1]);
wheelValue = block.inputs.wheelValues(setdiff(1:end, repeatPoints))';
wheelTime = block.inputs.wheelTimes(setdiff(1:end, repeatPoints))';
wheelTV = [wheelTime wheelValue-wheelValue(1)];
wheelTV = indexByTrial(trialTimes, wheelTV(:,1), [wheelTV(:,1) wheelTV(:,2)], times2Subtract);
wheelTV(cellfun(@isempty, wheelTV)) = deal({[0 0]});
wheelTV = cellfun(@(x) single([x(:,1) x(:,2)-x(find([x(1:end-1,1);1]>0,1),2)]), wheelTV, 'uni', 0);

visAzi = indexByTrial(trialTimes, e.visAzimuthTimes', [e.visAzimuthTimes' e.visAzimuthValues'], times2Subtract);
%%
%Calculate the decision threshold (necessary because different rotary
%encoders were used)
sR = 1000;
absVisAzi =  cellfun(@(x) abs(x(:,2)-x(1,2)), visAzi, 'uni', 0);
aziSampTV = cellfun(@(x,y) x(y>5 & y<55,:), visAzi, absVisAzi, 'uni', 0);
idxV = cellfun(@length, aziSampTV)>2;
corrWheel = cellfun(@(x,y) interp1(x(:,1), x(:,2), y(:,1), 'nearest', 'extrap'), wheelTV(idxV), aziSampTV(idxV), 'uni', 0);

visAziDiff = cell2mat(cellfun(@(x) diff(x(:,2)), aziSampTV(idxV), 'uni', 0));
corrWheelDiff = cell2mat(cellfun(@(x) diff(x), corrWheel, 'uni', 0));
idxZ = visAziDiff==0 | corrWheelDiff==0;
decisionThreshold = abs(round(median(corrWheelDiff(~idxZ)./visAziDiff(~idxZ))*60));

wheelTime = 1/sR:1/sR:block.inputs.wheelTimes(end);
rawWheel = interp1(block.inputs.wheelTimes, block.inputs.wheelValues', wheelTime', 'pchip', 'extrap')';
rawWheel = rawWheel-rawWheel(1);

if block.paramsValues(1).wheelGain<0
    wheelTV = cellfun(@(x) [x(:,1) x(:,2)*-1], wheelTV, 'uni', 0);
    rawWheel = rawWheel*-1;
    warning('Wheel connected in reverse... adjusting.');
end
stimOnsetIdx = round(stimPeriodStart(~timeOuts)*sR);
reactBound = num2cell([stimOnsetIdx round((feedbackTimes(~timeOuts)+0.25)*sR)],2);

%%
%Define a summation window (smthW--51ms) and velocity threshhold. sR*3/smthW means the wheel will need to move at least 3 "units" in 50ms for this
%to count as a movement initiation. Obviously, the physical distance of a "unit" depends on the rotary encoder. 3 seems to work well for 360 and 1024 
%encoders (the only ones I have used). I don't know about 100 encoders. 
smthW = 51;
velThresh = decisionThreshold*0.2; 

%Get wheel velocity ("wheelVel") from the interpolated wheel, and then use "posVelScan" and "negVelScan" to detect continuous velocity for smthW 
%movements in either direction. Note, here we use forward smoothing, so we are looking for times when the velocity initiates, and then continues for 
%the duration of "smthW". Also note, we introduce huge values whenever the wheel is moving in the opposite direction so that any "smthW" including a
%move in the opposite direction is eliminated from the scan. Times when the velocity is zero cannot be movement inititations either, so we multiply by
%(wheelVel~=0)

wheelVel = diff([rawWheel(1); rawWheel'])*sR; %In wheel ticks per second
posVelScan = conv(wheelVel.*double(wheelVel>0) - double(wheelVel<0)*1e6, [ones(1,smthW), zeros(1,smthW-1),]./smthW, 'same').*(wheelVel~=0);
negVelScan = conv(wheelVel.*double(wheelVel<0) + double(wheelVel>0)*1e6, [ones(1,smthW), zeros(1,smthW-1),]./smthW, 'same').*(wheelVel~=0);
movingScan = smooth((posVelScan'>=velThresh) + (-1*negVelScan'>=velThresh),21);
falseIdx = (movingScan(stimOnsetIdx)~=0); %don't want trials when mouse is moving at stim onset

%Identify onsets in both directions that exceed "velThresh", sort them, and record their sign. Also, extract all times the mouse is "moving"
tstWin = [zeros(1, smthW-1), 1];
velThreshPoints = [(strfind((posVelScan'>=velThresh), tstWin)+smthW-2) -1*(strfind((-1*negVelScan'>=velThresh), tstWin)+smthW-2)]';
[~, srtIdx] = sort(abs(velThreshPoints));
moveOnsetIdx = abs(velThreshPoints(srtIdx));
moveOnsetSign = sign(velThreshPoints(srtIdx))';
moveOnsetDir = (((moveOnsetSign==-1)+1).*(abs(moveOnsetSign)))';
onsetsByTrial = indexByTrial(cell2mat(reactBound)/sR, moveOnsetIdx/sR, [moveOnsetIdx/sR moveOnsetDir], [stimOnsetIdx/sR stimOnsetIdx*0]);
timeToThresh = (cellfun(@(x) max([nan find(abs(rawWheel(x(1):x(2))-rawWheel(x(1)))>decisionThreshold,1)+x(1)]), reactBound)-stimOnsetIdx)/sR;

badIdx = cellfun(@isempty, onsetsByTrial) | falseIdx | isnan(timeToThresh);
onsetsByTrial(badIdx) = deal({[nan nan]});
timeToThresh(badIdx) = nan;

%"firstMoveTimes" are the first onsets occuring after stimOnsetIdx. Eliminate any that are longer than 1.5s, as these would be timeouts. Also, remove 
%onsets when the mouse was aready moving at the time of the stimulus onset (impossible to get an accurate movement onset time in this case)
moveOnsetsTimeDir = repmat({[nan, nan]}, length(feedbackValues),1);
moveOnsetsTimeDir(~timeOuts) = onsetsByTrial;
timeToFirstMove = cell2mat(cellfun(@(x) x(1,1), moveOnsetsTimeDir, 'uni', 0));
dirOfFirstMove = cell2mat(cellfun(@(x) x(1,2), moveOnsetsTimeDir, 'uni', 0));
timeToResponseThresh = nan*timeToFirstMove;
timeToResponseThresh(~timeOuts) = timeToThresh;
reactionTime = arrayfun(@(x,y) max([nan x{1}(find(x{1}(:,1)<y, 1, 'last'),1)]), moveOnsetsTimeDir, timeToResponseThresh);
responseCalc = arrayfun(@(x,y) max([nan x{1}(find(x{1}(:,1)<y, 1, 'last'),2)]), moveOnsetsTimeDir, timeToResponseThresh);

useIdx = ~isnan(responseCalc) & reactionTime < 0.5;
if mean(responseCalc(useIdx) == responseRecorded(useIdx)) < 0.50 ...
        && sum(~isnan(responseCalc)&useIdx) >= 50 %&& mean(feedbackValues(feedbackValues~=0)>0)> 0.75
    warning('Why are most of the movements not in the same direction as the response?!?');
end

if isstruct(block.galvoLog)
    %Galvo position is the position of the galvos on each trial. It is changed so that for bilateral trials, the ML axis is always positive (bilateral
    %trials are when the laserTypeValue for that trial was 2). Note that the galvoPosValues output from the expDef are indices for the galvoCoords (with a
    %-ve index indicating the left himisphere). That's why we need to get the galvo posiiton on each trial by using the abs of this index and then
    %multiplying the ML coordinate by the sign of the original index.
    is_laserTrial = (e.laserTypeValues(eIdx)~=0 & ~isnan(e.laserTypeValues(eIdx)))';
    stim_laser1_power = e.laserPowerValues(eIdx)';

    inactivationSites = e.galvoCoordsValues(:,1:2);
    galvoPosValues = e.galvoPosValues(eIdx)';
    galvoPosition = inactivationSites(abs(galvoPosValues),:);
    galvoPosition(e.laserTypeValues(eIdx)'~=2,1) = galvoPosition(e.laserTypeValues(eIdx)'~=2,1).*sign(galvoPosValues(e.laserTypeValues(eIdx)'~=2));
    stim_laserPosition = galvoPosition;
else
    is_laserTrial = zeros(length(feedbackValues),1)>0;
    stim_laser1_power = zeros(length(feedbackValues),1)>0;
    stim_laserPosition = num2cell(zeros(length(feedbackValues),1)>0);
end


%% tstPlot
% cla;
% hold on;
% diffMove = find(timeToFirstMove ~= reactionTime & ~isnan(reactionTime));
% idx = diffMove(randperm(length(diffMove),1));
% xDat = timeToFirstMove(idx)-0.3:0.001:timeToResponseThresh(idx)+0.3;
% wheelPos = interp1(wheelTV{idx}(:,1), wheelTV{idx}(:,2), xDat, 'nearest', 'extrap');
% tIdx = knnsearch(xDat', [timeToFirstMove(idx); reactionTime(idx)]);
% plot(xDat, wheelPos, 'k'); 
% plot(xDat(tIdx(1)), wheelPos(tIdx(1)), '*r'); 
% plot(xDat(tIdx(2)), wheelPos(tIdx(2)), '*b'); 
% drawnow;

%% Populate n with all fields;
ev.is_blankTrial = is_blankTrial;
ev.is_visualTrial = is_visualTrial;
ev.is_auditoryTrial = is_auditoryTrial;
ev.is_coherentTrial = is_coherentTrial;
ev.is_conflictTrial = is_conflictTrial;
ev.is_validTrial = vIdx(:) & ~is_noStimTrial;
ev.is_noStimTrial = is_noStimTrial;

ev.block_trialOn = single(trialTimes(:,1));
ev.block_trialOff = single(trialTimes(:,2));
ev.block_stimOn = single(stimPeriodStart);

ev.timeline_rewardOn = feedbackValues*nan;
ev.timeline_audOn = feedbackValues*nan;
ev.timeline_audOff = feedbackValues*nan;
ev.timeline_visOn = feedbackValues*nan;
ev.timeline_visOff = feedbackValues*nan;

ev.timeline_audPeriodOn = feedbackValues*nan;
ev.timeline_audPeriodOff = feedbackValues*nan;
ev.timeline_visPeriodOn = feedbackValues*nan;
ev.timeline_visPeriodOff = feedbackValues*nan;
ev.timeline_firstMoveOn = timeToFirstMove(:,1)+ev.block_stimOn;
ev.timeline_firstMoveDir = dirOfFirstMove;
ev.timeline_choiceMoveOn = reactionTime+ev.block_stimOn;
ev.timeline_choiceMoveDir = responseCalc;
ev.timeline_choiceThreshOn = timeToResponseThresh+ev.block_stimOn;
ev.timeline_allMoveOn = cellfun(@(x) x(:,1), moveOnsetsTimeDir, 'uni', 0);
ev.timeline_allMoveDir  = cellfun(@(x) x(:,2), moveOnsetsTimeDir, 'uni', 0);
ev.timeline_wheelTime  = cellfun(@(x,y) x(:,1)+y, wheelTV, num2cell(ev.block_stimOn), 'uni', 0);

% NOTE: these are "raw" rather than in degrees of wheel movement as without
% recording the rotary encoder (which we didn't in some old recordings) it
% is not possible to know this...
ev.timeline_wheelValue  = cellfun(@(x) x(:,2), wheelTV, 'uni', 0);

ev.is_laserTrial = is_laserTrial(:);
ev.timeline_laserOn_rampStart = is_laserTrial(:)*nan;
ev.timeline_laserOn_rampEnd = is_laserTrial(:)*nan;
ev.timeline_laserOff_rampStart = is_laserTrial(:)*nan;
ev.timeline_laserOff_rampEnd = is_laserTrial(:)*nan;
ev.stim_laserPosition = num2cell(stim_laserPosition,2);
ev.stim_laser1_power = stim_laser1_power(:);
ev.stim_laser2_power = is_laserTrial(:)*nan;

ev.stim_correctResponse = single(correctResponse);
ev.stim_repeatNum = single(e.repeatNumValues(eIdx)');
ev.stim_audAmplitude = single(audAmplitude);
ev.stim_audAzimuth = single(audInitialAzimuth);
ev.stim_visContrast = single(visContrast);
ev.stim_visAzimuth = single(visInitialAzimuth);
ev.stim_closedLoop = single(ones(length(correctResponse),1));

ev.response_direction = single(responseCalc);
ev.response_feedback = single(feedbackValues);

 end
