
function [mictimes_tl,co] = mic(varargin)
% code to align the microphone using the buzzer 
% only works for sessions beyond Nov 2022
% 
params = csv.inputValidation(varargin{:});
pathStub = fullfile(params.expFolder{1}, ...
    [params.expDate{1} '_' params.expNum{1} '_' params.subject{1}]);

if ~isfield(params,'dataTimeline')
    fprintf(1, 'Loading timeline\n');
    loadedData = csv.loadData(params, 'dataType','timeline');
    timeline = loadedData.dataTimeline{1};
else
    timeline = params.dataTimeline{1};
end
% load the relevant raw data
loadedData = csv.loadData(params, 'dataType','mic');
%%
micDat = loadedData.dataMic{1};
% extract the timeline buzzer data
timelineBuzzerTimes = timeproc.getChanEventTime(timeline,'micSync');
% get the buzzer from the mic data
[micBuzzerTimeStart,~] = get_buzz_from_mic(micDat,1,pathStub); 
[micBuzzerTimeEnd,~] = get_buzz_from_mic(micDat,-1,pathStub);
micBuzzerTimes = [micBuzzerTimeStart;micBuzzerTimeEnd];

% calculate the slope and the intercept between the two points
m = (timelineBuzzerTimes(2) - timelineBuzzerTimes(1))/(micBuzzerTimes(2) - micBuzzerTimes(1));
c = timelineBuzzerTimes(1) - m*micBuzzerTimes(1);
co = [m;c];
% assuming even sampling 
mictimes=linspace(0,numel(micDat.micData)/micDat.Fs,numel(micDat.micData));
fitmic_times = @(t)m*t + c;
mictimes_tl=fitmic_times(mictimes);

end

function [sndOnsetTspec,sndOffsetTspec]=get_buzz_from_mic(micDat,startend,pathStub)
% micdat = struct, contains micData and Fs 
% micData = int16, unfiltered microphone data 
% Fs = double, micrpophone sampling rate 

% this code takes the power spectrum and measures the onset times when
% power is above threshold determined by powerThr 
% filters applied:  kaiser on raw data 
%                   butterworth on frequecy domain to only look at power of low
%                   frequencies

% subsample for the beginning of the rec or the end 
timebin=60; % sample the first/last 60s, normally happens in the 1st 10s
Fs = micDat.Fs; 
micData = micDat.micData; 

samplelenth=timebin*Fs;
tdat=double(micData(1:samplelenth))'; 
if startend==-1
    micdl = numel(micData);
    tdat = double(micData(micdl-samplelenth:micdl));
end

used_tone = 5000; % I empirically find that this tone is the best to use as this is low enough that it is below vocalisation and bg noise
tDat_tone = bandpass(tdat,[used_tone-500 used_tone+500],Fs);
[s,fspec,tspec] = spectrogram(tDat_tone,kaiser(512,5),440,1024,Fs);
s = abs(s);
x = mean(s(fspec>3000,:,:),1);
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

% plot the click detection
f = figure('visible','off'); hold all
plot(x)
xline(sndOnsetSpec,'r'); axis tight;
ylabel('mean spectrogram, filtered')
xlabel('mic sample')
if startend==-1
    saveas(f,[pathStub 'mic_buzz_end.png'],'png');
else
    saveas(f,[pathStub 'mic_buzz_start.png'],'png');
end
close; 
%xline(sndOffsetSpec,'r');

% add the time before the sample to get the the time relative to the
% beginning when the end of the recording is taken
if startend==-1
    sndOnsetTspec=sndOnsetTspec+((micdl-samplelenth)/Fs);
    sndOffsetTspec=sndOffsetTspec+((micdl-samplelenth)/Fs); 
end

end