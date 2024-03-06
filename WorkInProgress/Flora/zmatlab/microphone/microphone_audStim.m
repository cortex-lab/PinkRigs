% some prelimiary alignment of microphone using the auditroy stimuli 

clc; clear all; 
params.subject = {['AV030']};
params.expDate = '2022-12-08';
params.expDef = 'p';
%params.expNum = 1;
exp2checkList = csv.queryExp(params);
expInfo = exp2checkList(1,:);
expInfo = csv.loadData(expInfo, 'dataType',{'timeline','events'});
% load also the mic data (currently not implemented by csv)

expPathStub = strcat(expInfo.expDate, {'_'}, expInfo.expNum, {'_'}, expInfo.subject);
micPath = cell2mat([expInfo.expFolder '\' expPathStub '_mic.mat']);
load(micPath); 

%% get the beginning and the end period of the mic data
micdl=numel(micData);

timebin=60; % normally happens in the 1st 10s 
samplelenth=timebin*Fs;
tdat_start=double(micData(1:samplelenth))';

%micdl = samplelenth*2;
tdat_end=double(micData(micdl-samplelenth:micdl));
%
clickFreq=8; % Clicks are at 8 Hz. 
tic; 
[audPeriodOn_start,~] = getAudPeriodOn_from_mic(tdat_start,Fs,clickFreq);
[audPeriodOn_end,~] = getAudPeriodOn_from_mic(tdat_end,Fs,clickFreq); 
toc; 
% the times are not relative to the whole trace so add that 
audPeriodOn_end=audPeriodOn_end+((micdl-samplelenth)/Fs); 
audPeriodOn_start = audPeriodOn_start(1:7);
% audPeriodOn_end(end) = [];

%%
% look at the onset times 
audOn_tl = expInfo.dataEvents{1,1}.timeline_audPeriodOn;
audOn_tl = audOn_tl(~isnan(audOn_tl));
ix_start=numel(audPeriodOn_start); 
tltimes_start=audOn_tl(1:ix_start);
ix_end=numel(audPeriodOn_end); 
tltimes_end=audOn_tl(numel(audOn_tl)-ix_end+1:numel(audOn_tl));
mictimes_align=[audPeriodOn_start audPeriodOn_end];
tltimes_align=[tltimes_start' tltimes_end'];
figure; plot(mictimes_align,tltimes_align,'o'); xlabel('micTimes'),ylabel('tlTimes'); 
%%
co=robustfit(mictimes_align,tltimes_align); 
% figure; plot(mictimes,tltimes,'.');
% hold on;
% a=floor(micdl/Fs);
% myt=0:0.1:a;
% fitmic_times = @(t)t*co3(2) + co3(1); 
% mytt=fitmic_times(myt); 
% plot(myt,mytt); 
%%
% fit the mictimes to timeline
% assuming that the micdata is equally sampled I suppose

% subsample micdata 
micdata_sub=downsample(micData,100);
maxtime=micdl/Fs; 
mictimes=linspace(0,maxtime,numel(micdata_sub));
%
fitmic_times = @(t)t*co(2) + co(1);

mictimes_tl=fitmic_times(mictimes); 

%% plot spectrogram per trial
c_idx = 3954400; 
example=double(micData(c_idx-1200000:c_idx+3000000));
[s,w,t]=spectrogram(example,kaiser(1000,10),[],[],Fs); 
figure; imagesc(t, w/1000, 20*log10(abs(s)));
%hold on; line([.5,.5], [0,100], 'Color', 'r','LineWidth',1);
%hold on; line([.7,.7], [0,100], 'Color', 'r','LineWidth',1);
caxis([30 120]);
colorbar;
axis xy 
xlabel('time (s)');
ylabel('freq(kHz)');


%% check average spectrogram -- do aud!!

visOn_tl = expInfo.dataEvents{1,1}.timeline_visPeriodOn;
visOn_tl = visOn_tl(~isnan(visOn_tl));

myM=mictimes_tl-visOn_tl(1:45); 
[~,ix]=min(abs(myM),[],2);
%
%figure; plot(micdata_sub(ix(2)-200:ix(2)+1000));
%
trgspeaker=zeros(25,200001);


% ss=zeros(25,513,3049); 
ix_upsampled=ix*100;
clear w s t
for tr=1:25
    trgspeaker(tr,:)=smooth(abs(double(micData(ix_upsampled(tr)-100000:ix_upsampled(tr)+100000))),5001); 
    noSmooth(tr,:)=double(micData(ix_upsampled(tr)-100000:ix_upsampled(tr)+100000)); 
    [s(:,:,tr),w(:,:,tr),t(:,:,tr)]=spectrogram(noSmooth(tr,:),kaiser(1000,10),[],[],Fs); 
end
figure; plot(mean(trgspeaker,1)); 

%
s = median(abs(s),3);
w = mean(w,3);


t = mean(t,3);

figure; imagesc(t, w/1000, 20*log10(abs(s)));
hold on; line([.5,.5], [0,100], 'Color', 'r','LineWidth',1);
%hold on; line([.7,.7], [0,100], 'Color', 'r','LineWidth',1);
caxis([30 120]);
colorbar;
axis xy 

xlabel('time (s)');
ylabel('freq(kHz)');

%%
function [sndOnsetTspec,sndOffsetTspec]=getAudPeriodOn_from_mic(tdat,Fs,clickFreq)
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
tDat_bandstopped = bandpass(tdat,[8000*0.9 16000*1.1],Fs);


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
thresh = [min(thresh) + range(thresh)*0.2;  max(thresh) - range(thresh)*0.3]; 
powerThr=thresh(1);  

%
%%

sndOnsetSpec = find(diff((x > powerThr)) == 1);
sndOnsetTspec = tspec(sndOnsetSpec);  % onset times in s 
%sndOnset = ceil(sndOnsetTspec * Fs); % onet timepoints in micData

sndOffsetSpec = find(diff((x > powerThr)) == -1);
sndOffsetTspec = tspec(sndOffsetSpec);  % offset times in s 

% get throw away single clicks, as those are either the buzzer or the
% reward. 
isPeriodStart = logical([0,diff(diff(sndOnsetTspec)<1/clickFreq*1.2)]>0);

% plot the click detection 
figure; plot(x);
hold on; 
onsets_kept = sndOnsetSpec(isPeriodStart);
for ii=1:numel(onsets_kept)
    xline(onsets_kept(ii),'r');
end

sndOnsetTspec=sndOnsetTspec(isPeriodStart);

end