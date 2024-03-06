% plot training varaibles 
clc
clear all
close all
addpath(genpath('C:\Users\Flora\Documents\Github\PinkRigs'));


%% Get exp ref

params.subject = {['all']}; 
params.expDate = {['2022-01-01:2024-01-01']};
%params.expNum ={'1'}; 

params.expDef = 't'; 

exp2checkList = csv.queryExp(params);

%% throw away experiments that are too short
exp2checkList = exp2checkList(str2double(exp2checkList.expDuration)>1000,:);
% and where ephys fdolder existed
exp2checkList = exp2checkList(str2double(exp2checkList.existEphys)==0,:);

% for every experiment load block

expcount = size(exp2checkList); 
expcount = expcount(1);


%%
subject = strings(expcount,1); stage = zeros(expcount,1); 
trialNum = zeros(expcount,1); bias = zeros(expcount,1); coh = zeros(expcount,1); 
aud = zeros(expcount,1); vis = zeros(expcount,1); 

for exp=1:expcount
    tDat = exp2checkList(exp,:); 
    [mysubject, expDate, expNum, server] = parseExpPath(tDat.expFolder{1});
    blkName = [datestr(expDate, 'yyyy-mm-dd') '_' expNum '_' mysubject '_Block.mat'];
    requestedBlock = load([tDat.expFolder{1} '\' blkName]);
    performance  = calculate_performance(requestedBlock.block);

    %% create table of data 
    subject(exp) = mysubject;
    stage(exp)=performance.stage; 
    trialNum(exp)=performance.trialnum; 
    bias(exp)=performance.bias;
    coh(exp)=performance.performance.coh;
    aud(exp)=performance.performance.aud;
    vis(exp)=performance.performance.vis;
end
%% ANALYSIS %% 

dat = table(subject,stage,trialNum,bias,coh,aud,vis); 
mice = unique(dat.subject); 


clear discard
to_exclude = {'default','AV001','AV002','AV003','AV004','AV005'}; 
for i=1:numel(to_exclude)
    discard(i,:) = strcmp(mice, to_exclude{i}); 
end 
discard = logical(sum(discard,1))'; 

mice = mice(~discard); 
% mice
%mice = ["AV006","AV007","AV008","AV005","AV013","AV014","AV015","FT032","FT035","FT034","FT036"];
%%
% stage progression per mice 
stages = cell(numel(mice),1); 
for i=1:numel(mice)
    tDat=dat(dat.subject == mice{i},:);
    stages{i}=tDat.stage(tDat.stage<6);
end 

% get some sort of average 
firstsess = cellfun(@(x) x(1),stages); 
stages = stages(firstsess==1,:);
trainingtime = cellfun(@numel,stages);

trainingprogress = NaN(numel(stages),max(trainingtime)); 

figure; 
for i=1:numel(stages)
    upgrades=stages(i); 
    trainingprogress(i,1:numel(upgrades{1}))=upgrades{1};  
    
    trainingprogress(i,numel(upgrades{1}):end)=upgrades{1}(end);
    % account for downstaging after implant 
    if sum(upgrades{1}==5)>0
        ix = find(upgrades{1}==5); 
        trainingprogress(i,ix:end)=5;
    end
    
    plot(upgrades{1}+rand(1)*.05,'Color',[1,1,1]*.5);
    hold on; 
end
plot(nanmean(trainingprogress),'k');
%%
figure; 
subplot(1,3,1)
clist=["black","magenta","blue"];
for i=2:4
    plot(sum(trainingprogress>i,1)/numel(stages)+rand(1)*.01,'Color',clist{i-1},'LineWidth',4);
    hold on;
end
set(gca,'box','off')
xlabel('# sessions')
ylabel('fraction of mice');
%
% trial numbers
tNum=NaN(numel(mice),max(trainingtime)); 
for i=1:numel(mice)
    tDat=dat(dat.subject == mice{i},:);
    tDat = tDat(tDat.stage<6,:);
    tNum(i,1:numel(tDat.trialNum))=tDat.trialNum;
end 

%figure;
subplot(1,3,2)
for i=1:numel(mice)
    plot(tNum(i,1:25),'Color',[1,1,1]*.5); 
    hold on;
end
plot(nanmean(tNum(:,1:25),1),'k','LineWidth',5); 
set(gca,'box','off')
xlim([0 26])
xlabel('# sessions')
ylabel('trial #');

%
subplot(1,3,3)
tBias=NaN(numel(mice),max(trainingtime)); 
for i=1:numel(mice)
    tDat=dat(dat.subject == mice{i},:);
    tDat = tDat(tDat.stage<5,:);
    tBias(i,1:numel(tDat.bias))=tDat.bias;
end 

for i=1:numel(mice)
    plot(tBias(i,1:25),'Color',[1,1,1]*.5); 
    hold on;
end
%plot(nanmean(tBias,1),'k','LineWidth',5); 
set(gca,'box','off')
xlim([0 26])
xlabel('# sessions')
ylabel('bias');

%% 
%figure;
% subplot(1,3,3)
% 
% for i=1:numel(mice)
%     tDat=dat(dat.subject == mice{i},:);
%     tDat = tDat(tDat.stage<6,:);
%     tDat = tDat((tDat.stage<4)&(tDat.stage>2),:);
%     
%     plot(tDat.coh,'Color',[1,1,1]*.5); 
%     plot(tDat.aud,'Color',[1,0,1]*.9); 
%     plot(tDat.vis,'Color',[0,0,1]*.9);
%     hold on
% end 



function [previous_session]=calculate_performance(block)


%% calculate performance %%%
eIdx = 1:length(block.events.endTrialTimes);
repeatnum=block.events.repeatNumValues(eIdx);
responseType=block.events.responseTypeValues(eIdx);
validtrialIdx=find((repeatnum==1)&(responseType~=0));


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
trialType.visual = (audAmplitude==0 | audInitialAzimuth==0) & (visContrast>.05 & visInitialAzimuth~=0);

trialType.coherent = sign(visInitialAzimuth.*audInitialAzimuth)>0 & audAmplitude>0 & visContrast>.05;
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

%previous_session.rewardSize=str2num(block.events.totalRewardValues(1:3));


if isfield(block.paramsValues,'wheelGain')
    previous_session.wheelGain=block.paramsValues.wheelGain;
else
    previous_session.wheelGain=block.events.selected_paramsetValues.wheelGain;
end


stage = block.events.selected_paramsetValues.trainingStage;


previous_session.stage=stage;
previous_session.responseWindow = block.events.selected_paramsetValues.responseWindow;
end

