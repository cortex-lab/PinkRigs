function [previous_session]=calculate_performance(blockfile)

load(blockfile);

%% calculate performance %%%
eIdx = 1:length(block.events.endTrialTimes);
repeatnum=block.events.repeatNumValues(eIdx);
responseType=block.events.responseTypeValues(eIdx);
timeoutSinceTrialChange = block.events.timeOutsSinceTrialChangeValues(eIdx); 
validtrialIdx=find((repeatnum==1)&(responseType~=0));
% 
%validtrialIdx = find((responseType~=0) & (repeatnum==1 | timeoutSinceTrialChange == repeatnum-1)); 

% calculate parameters on valid trials

% so this should be saved in the other one and also rechecked as in
% previous stuff it will have this in events

% old training vs new training some values saved differently
if isfield(block.paramsValues,'visContrast')
    visContrast=[block.paramsValues(validtrialIdx).visContrast];
    audAmplitude=[block.paramsValues(validtrialIdx).audAmplitude];

elseif isfield(block.events,'visContrastValues')
    visContrast=block.events.visContrastValues(validtrialIdx);
    audAmplitude=block.events.audAmplitudeValues(validtrialIdx);
end

visInitialAzimuth=block.events.visInitialAzimuthValues(validtrialIdx);
audInitialAzimuth=block.events.audInitialAzimuthValues(validtrialIdx);

%Create a "logical" for each trial type (blank, auditory, visual, coherent, and incoherent trials)
trialType.blank = (visContrast==0 | visInitialAzimuth==0) & (audAmplitude==0 | audInitialAzimuth==0);
trialType.auditory = (visContrast==0 | visInitialAzimuth==0) & (audAmplitude>0 & audInitialAzimuth~=0);
% only take coherent trials with high contrast to use for further
% updating
trialType.visual = (audAmplitude==0 | audInitialAzimuth==0) & (visContrast==max(visContrast) & visInitialAzimuth~=0);

trialType.coherent = sign(visInitialAzimuth.*audInitialAzimuth)>0 & audAmplitude>0 & visContrast==max(visContrast);
trialType.conflict = sign(visInitialAzimuth.*audInitialAzimuth)<0 & audAmplitude>0 & visContrast>0;

% responses
responses=block.events.responseTypeValues(validtrialIdx);
feedbacktype=block.events.feedbackValues(validtrialIdx); % timeout is when this is 0.

%trialType.repeatNum = e.repeatNumValues(1:length(vIdx))';

previous_session.trialnum=size(validtrialIdx,2);
previous_session.performance.coh=mean(feedbacktype(trialType.coherent==1)>0);
previous_session.performance.aud=mean(feedbacktype(trialType.auditory==1)>0);
previous_session.performance.vis=mean(feedbacktype(trialType.visual==1)>0);
previous_session.bias=mean(responses(feedbacktype~=0)>0);


% calculate some more stringent parameters on auditory
previous_session.performance.audR=mean(responses((trialType.auditory==1)&(audInitialAzimuth==60))>0);
previous_session.performance.audL=mean(responses((trialType.auditory==1)&(audInitialAzimuth==-60))>0);
previous_session.performance.visR=mean(responses((trialType.visual==1)&(visInitialAzimuth==60))>0);
previous_session.performance.visL=mean(responses((trialType.visual==1)&(visInitialAzimuth==-60))>0);

previous_session.rewardSize=str2num(block.events.totalRewardValues(1:3));


if isfield(block.paramsValues,'wheelGain')
    previous_session.wheelGain=block.paramsValues.wheelGain;
else
    previous_session.wheelGain=block.events.selected_paramsetValues.wheelGain;
end

if isfield(block.paramsValues,'wheelMovementProbability')
    previous_session.wheelMovementProbability=block.paramsValues.wheelMovementProbability;
else
    previous_session.wheelMovementProbability=block.events.selected_paramsetValues.wheelMovementProbability;
end

% determine which stage the training we were at before
%         stage=1;
%         if size(unique(visContrast),2)>3
%             stage=2;
%             if sum(trialType.auditory)>0
%                 stage=3;
%                 if size(unique(audInitialAzimuth),2)>2
%                     stage=4;
%                     if sum(trialType.blank)>0
%                         stage=5;
%                     end
%                 end
%             end
%         end
stage = block.events.selected_paramsetValues.trainingStage;


previous_session.stage=stage;
previous_session.responseWindow = block.events.selected_paramsetValues.responseWindow;
end