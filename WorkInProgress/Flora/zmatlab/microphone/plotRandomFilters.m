%%

% p.clickDuration, sampleRate, fBnk, numAudChannels,randIdx, @help.genClkSet

% look at the randon filters
clkD = 0.08; 
spkF = 96000;
nCha = 8; 
% load an example SpeakerCalibration with fBnk
load('\\znas.cortexlab.net\Code\Rigging\config\ZELDA-STIM1\SpeakerCalibration.mat')
load('\\znas.cortexlab.net\Code\Rigging\ExpDefinitions\Flora\randomFilters_UBERset.mat')
%%
figure;
for i=1:10
speakerIdx = 2;
intS = reshape(pinknoise(clkD*spkF*(nCha-1)), [nCha-1,clkD*spkF,]); 
clkS= filtfilt(fBnk.rndF(i,:), 1, intS(speakerIdx,:))'; 
clkS = filtfilt(fBnk.cFlt.sosMatrix, fBnk.cFlt.ScaleValues, clkS);
clkS = filtfilt(fBnk.sFlt(speakerIdx,:), 1, clkS)';

[S,F,T] = spectrogram(clkS,kaiser(500,10),[],[],spkF,'yaxis');
% Calculate the average over time
average_spectrogram = mean(abs(S), 2);
average_spectrogram = 10*log10(average_spectrogram); 
average_spectrogram = average_spectrogram-average_spectrogram(1); 
% Plot the average spectrogram
subplot(10,1,i)
plot(F((F>2000) & (F<25000)), average_spectrogram((F>2000) & (F<25000)));
end 

xlabel('Frequency (Hz)');
ylabel('relative power/Frequency (dB/Hz)');