function plotDriftMap4shank(varargin)
    % function to plot Driftmap. Supports any configuration of a 2.0 probe.
    % Plots drift estimate if drift.times.npy/drift folder (in later version of pykilosort) exists.
    % user needs to navigate to kilosort folder upon running the function or input ks folder 
    % for example naviage to 
    % zaru\Subjects\AV025\2022-11-10\ephys\AV025_2022-11-10_ActivePassiveSparseNoise_g0\AV025_2022-11-10_ActivePassiveSparseNoise_g0_imec0\pyKS\output
    if ~isempty(varargin)
        ksDir = varargin{1};
    else 
        ksDir = uigetdir; 
    end
    [spikeTimes, spikeAmps, spikeDepths, spikeXpos,~] =ksDriftmap(ksDir);
    if exist([ksDir '\drift'])
        plotDrift = true; 
%         spikeDepths = readNPY([ksDir '\drift\spike_depths.npy']);
%         spikeAmps = readNPY([ksDir '\drift\spike_amps.npy']);
    else 
        plotDrift = false;
    end
    % get which shank each spike is on
    shank_borders = [0,150,300,500,700];
    spikeShankIdx = zeros(numel(spikeTimes),1); 
    for i=1:4
        spikeShankIdx((spikeXpos>=shank_borders(i))&(spikeXpos<=shank_borders(i+1)))=i;
    end 
    % plot 
    recorded_shanks = unique(spikeShankIdx);
    recorded_shanks = recorded_shanks(recorded_shanks>0);  % a few spikes positions are NaN 
    %
    figure;
    for i=1:numel(recorded_shanks)
         if plotDrift
            subplot(1,numel(recorded_shanks)+1,i);
         else
            subplot(1,numel(recorded_shanks),i);
         end
         plotDriftmap(spikeTimes(spikeShankIdx==recorded_shanks(i)),...
             spikeAmps(spikeShankIdx==recorded_shanks(i)), ...
             spikeDepths(spikeShankIdx==recorded_shanks(i)));
         ylim([min(spikeDepths)*0.98,max(spikeDepths)*1.02]);
         title(sprintf('Shank %.0d',i)); 
    end 

    % add the calculated drift from pyKS on the plots. 
    %the drift is not calculated per shank but as total drift
    if plotDrift 
%         drift_times = readNPY([ksDir '\' 'drift.times.npy']); 
%         drift_um = readNPY([ksDir '\' 'drift.um.npy']); 
%         drift_depths_um = readNPY([ksDir '\' 'drift_depths.um.npy']); 

        drift_um = readNPY([ksDir '\drift\' 'dshift.npy']); 
        drift_depths_um = (readNPY([ksDir '\drift\' 'yblk.npy']))'; 
        
        dl = size(drift_um); 
        drift_times = 1:dl(1); 

% add the calculated drift from pyKS on the plots. 
% the drift is not calculated per shank but as total drift
% drift_times = readNPY([ksDir '\' 'drift.times.npy']); 
% drift_um = readNPY([ksDir '\' 'drift.um.npy']); 
% drift_depths_um = readNPY([ksDir '\' 'drift_depths.um.npy']); 
% 
% subplot(1,numel(recorded_shanks)+1,numel(recorded_shanks)+1);
% % add the starting position to each driftunm
% drift_um_corrected_start = drift_um + repmat(drift_depths_um,numel(drift_times),1); 
% plot(drift_times,drift_um_corrected_start);
% ylim([min(spikeDepths)*0.98,max(spikeDepths)*1.02]);
% title('drift estimate by pykilosort');
        subplot(1,numel(recorded_shanks)+1,numel(recorded_shanks)+1);
        % add the starting position to each driftunm
        drift_um_corrected_start = drift_um + repmat(drift_depths_um,numel(drift_times),1); 
        plot(drift_times,drift_um_corrected_start);
        ylim([min(spikeDepths)*0.98,max(spikeDepths)*1.02]);
        title('drift estimate by pykilosort');
    end
end
