% plot driftmap of 4-shank recordings
%ksDir = 'Z:\AV008\2022-03-30\ephys\AV008_2022-03-30_ActivePassive_g0\AV008_2022-03-30_ActivePassive_g0_imec0\kilosort2';
ksDir = uigetdir; 
[spikeTimes, spikeAmps, spikeDepths, spikeXpos,spikeSites] =ksDriftmap(ksDir);

% get which shank each spike is on
shank_borders = [0,150,300,500,700];
spikeShankIdx = zeros(numel(spikeTimes),1); 
for i=1:4
    spikeShankIdx((spikeXpos>=shank_borders(i))&(spikeXpos<=shank_borders(i+1)))=i;
end 
% plot 
recorded_shanks = unique(spikeShankIdx);
recorded_shanks = recorded_shanks(recorded_shanks>0);  % a few spikes positions are NaN 

figure;
for i=1:numel(recorded_shanks)
     subplot(1,numel(recorded_shanks),i);
     plotDriftmap(spikeTimes(spikeShankIdx==recorded_shanks(i)),...
         spikeAmps(spikeShankIdx==recorded_shanks(i)), ...
         spikeDepths(spikeShankIdx==recorded_shanks(i)));
     ylim([min(spikeDepths)*0.9,max(spikeDepths)*1.1]);
     title(sprintf('Shank %.0d',i)); 
end 

