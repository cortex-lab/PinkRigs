% some prelimiary alignment of microphone
clc; clear all; 
mname='AV025';
date='2022-11-09';
expnum=2; 
params.subject = {'FT010'};
params.expDate = '2021-03-16';
%params.expDef = 'p';
params.expNum = 7;
exp2checkList = csv.queryExp(params);
expInfo = exp2checkList(1,:);
expInfo = csv.loadData(expInfo, 'dataType',{'timeline','eventsFull'});
% load also the mic data (currently not implemented by csv)

expPathStub = strcat(expInfo.expDate, {'_'}, expInfo.expNum, {'_'}, expInfo.subject);
micPath = cell2mat([expInfo.expFolder '\' expPathStub '_mic.mat']);
load(micPath); 

%% get the beginning and the end period of the mic data
micdl=numel(micData);
timebin=60; % normally happens in the 1st 10s 
samplelenth=timebin*Fs;
tdat_start=double(micData(1:samplelenth))';
tdat_end=double(micData(micdl-(120*Fs):micdl));
%
clkDur=0.05; % should change to audOff-audOn later
numClicks = 4;
clickFreq=8; % Clicks are at 8 Hz. 
tic; 
[buzzerOn,~] = getBuzzerOn_from_mic(tdat_start,Fs);
[buzzerOff,~] = getBuzzerOn_from_mic(tdat_end,Fs); 
toc; 
%%
function [sndOnsetTspec,sndOffsetTspec]=getBuzzerOn_from_mic(tdat,Fs)
% tDat = unfiltered microphone data 
% Fs micrpophone sampling rate 
% clkDur click duration

% this code takes the power spectrum and measures the onset times when
% power is above threshold determined by powerThr 
% filters applied:  kaiser on raw data 
%                   butterworth on frequecy domain to only look at power of low
%                   frequencies

% this is not good enough as I think the mouse says something or idk but
% sometimes there is additional noise in this range 

% bandstop filter to isolate buzzer from generated clicks (8-16kHz);
tDat_bandstopped = bandstop(tdat,[8000*0.9 16000*1.1],Fs);


[s,fspec,tspec] = spectrogram(tDat_bandstopped,kaiser(512,5),440,1024,Fs);
s = abs(s);
x = mean(s(fspec>3000,:,:),1);
fc = 20; % butterworth filter size =
[b_hp,a_hp] = butter(3,fc/(1/2/mean(diff(tspec))),'low');
x = filter(b_hp,a_hp,x); x(1) = x(2);

%%
% hardcoded thresholds don't really work, the differences accross
% recordings are too different so I will do kmeans
[~, thresh] = kmeans(x',5); % cluster the photodiode trace to determine when is the cutoff for on and off
thresh = [min(thresh) + range(thresh)*0.3;  max(thresh) - range(thresh)*0.3]; 
powerThr=thresh(2);  

%
%%

sndOnsetSpec = find(diff((x > powerThr)) == 1);
sndOnsetTspec = tspec(sndOnsetSpec);  % onset times in s 
%sndOnset = ceil(sndOnsetTspec * Fs); % onet timepoints in micData


sndOffsetSpec = find(diff((x > powerThr)) == -1);
sndOffsetTspec = tspec(sndOffsetSpec);  % offset times in s 
%sndOffset = ceil(sndOffsetTspec * Fs); 

% plot the click detection 
figure; plot(x);
hold on; 
xline(sndOnsetSpec,'r');
hold on; 
xline(sndOffsetSpec,'r');
end