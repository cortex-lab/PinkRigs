% microphone averages

clc; clear all; 
params.subject = {['AV030']};
params.expDate = '2022-12-09';
params.expDef = 'm';
%params.expNum = 1;
exp2checkList = csv.queryExp(params);
expInfo = exp2checkList(1,:);
expInfo = csv.loadData(expInfo, 'dataType',{'ev','mic'});
expPathStub = strcat(expInfo.expDate, {'_'}, expInfo.expNum, {'_'}, expInfo.subject);
alignPath = cell2mat([expInfo.expFolder '\' expPathStub{1} '_alignment.mat']);
load(alignPath,'mic');

%%
micdat = expInfo.dataMic{1,1};
micData  = micdat.micData; 
Fs =micdat.Fs;
% subsample micdata 
micdl =numel(micData); 
micdata_sub=downsample(micData,100);
maxtime=micdl/Fs; 
mictimes=linspace(0,maxtime,numel(micdata_sub));
fitmic_times = @(t)t*mic.slope + mic.intercept;
mictimes_tl=fitmic_times(mictimes); 

%% 
ev = expInfo.dataEvents{1,1}; 
ev.stim_visContrast = int16(ev.stim_visContrast*100);
contrasts = unique(ev.stim_visContrast); 
contrasts(contrasts<.00001)=[];
hold all;

for c=1:1
    on_time = ev.timeline_visPeriodOn(ev.is_visualTrial & ev.stim_audAzimuth==0);
    n_samples=40; 
    r_idx = randi([1,numel(on_time)],1,n_samples);
    on_time = on_time(r_idx); 
    plot_screen_sound(on_time,mictimes_tl,micData,Fs,0);
end
%%
function plot_screen_sound(ev_times,mic_times,micData,Fs,plotByFreq)
myM=mic_times-ev_times; 
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
    tDat_tone(tr,:) = bandpass(noSmooth(tr,:),[30000 70000],Fs);
    [s(:,:,tr),w(:,:,tr),t(:,:,tr)]=spectrogram(tDat_tone(tr,:),kaiser(1000,10),[],[],Fs); 
end
%figure; plot(mean(trgspeaker,1)); 

% average over trials 
s = median(abs(s),3);
w = mean(w,3);
t = mean(t,3);


if plotByFreq==1
    figure; imagesc(t, w/1000, 20*log10(abs(s)));
    hold on; line([.5,.5], [0,100], 'Color', 'r','LineWidth',1);
    %hold on; line([.7,.7], [0,100], 'Color', 'r','LineWidth',1);
    caxis([30 120]);
    colorbar;
    axis xy 
    
    xlabel('time (s)');
    ylabel('freq(kHz)');
else
    plot(t,mean(20*log10(abs(s)),1)); hold on;
    line([.5,.5], [16,20], 'Color', 'r','LineWidth',1);
end

end 

