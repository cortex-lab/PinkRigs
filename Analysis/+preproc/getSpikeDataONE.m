function spk = getSpikeDataONE(ephysPath,KSFolder)
    %%% This function will load the spike data (spike time, templates, etc.). 
    %%% "spk" contains a condensed summary with spikes times & template info.

    %% Parameters

    % so that one can give this a custom folder
    if ~exist('KSFolder','var')
        KSFolder = fullfile(ephysPath,'kilosort2');
    end
    
    %% Get spike info
    
    % Load params
    spikeStruct = loadParamsPy(fullfile(KSFolder, 'params.py'));
    
    % Load spike times
    ss = readNPY(fullfile(KSFolder, 'spike_times.npy'));
    st = double(ss)/spikeStruct.sample_rate;
    
    % Load spike templates
    spikeTemplates = readNPY(fullfile(KSFolder, 'spike_templates.npy')); % note: zero-indexed
    
    % Load spike amplitudes
    tempScalingAmps = readNPY(fullfile(KSFolder, 'amplitudes.npy'));
    tempScalingAmps = single(tempScalingAmps);
    
    % Load pc features
    pcFeat = readNPY(fullfile(KSFolder,'pc_features.npy')); % nSpikes x nFeatures x nLocalChannels
	pcFeatInd = readNPY(fullfile(KSFolder,'pc_feature_ind.npy')); % nTemplates x nLocalChannels

    % Load KS labels
	cgsFile = fullfile(KSFolder, 'cluster_KSLabel.tsv');
    [cids_KS, cgs_KS] = readClusterGroupsCSV(cgsFile); % cids should be the same as unique(spikeTemplates)??
    
    % Load coords
    coords = readNPY(fullfile(KSFolder, 'channel_positions.npy'));
    ycoords = coords(:,2); xcoords = coords(:,1);
    
    % Compute depths
    pcFeat = squeeze(pcFeat(:,1,:)); % take first PC only
    pcFeat(pcFeat<0) = 0; % some entries are negative, but we don't really want to push the CoM away from there.
    
    % which channels for each spike?
    spikeFeatInd = pcFeatInd(spikeTemplates+1,:);
    % ycoords of those channels?
    spikeFeatYcoords = ycoords(spikeFeatInd+1); % 2D matrix of size #spikes x 12
    % center of mass is sum(coords.*features)/sum(features)
    spikeDepths = sum(spikeFeatYcoords.*pcFeat.^2,2)./sum(pcFeat.^2,2);
    
    spikeFeatXcoords = xcoords(spikeFeatInd+1); % 2D matrix of size #spikes x 12
    spikeXPos = sum(spikeFeatXcoords.*pcFeat.^2,2)./sum(pcFeat.^2,2);

    [~,spikeShankIDs] = min(abs(spikeXPos - repmat([0 200 400 600], [numel(spikeXPos),1])),[],2);
    spikeShankIDs = uint8(spikeShankIDs-1);

    
    %% Get template info
    
    % don't have template amplitude here?
    temps = readNPY(fullfile(KSFolder, 'templates.npy'));
    temps_ind = readNPY(fullfile(KSFolder, 'templates_ind.npy'));
    temps_ind = single(temps_ind);
    
    temp_KSLabels = zeros(1,numel(cids_KS),'uint8');
    temp_xpos = zeros(1,numel(cids_KS),'single');
    temp_depths = zeros(1,numel(cids_KS),'single');
    temp_shankIDs = zeros(1,numel(cids_KS),'uint8');
    for ii = 1:numel(cids_KS)
        temp = cids_KS(ii);
        spkIdx = spikeTemplates == temp;
        temp_KSLabels(ii) = cgs_KS(cids_KS == temp);
        temp_xpos(ii) = nanmedian(spikeXPos(spkIdx)); % not sure why there can be nans here
        temp_depths(ii) = nanmedian(spikeDepths(spkIdx)); 
        temp_shankIDs(ii) = nanmedian(spikeShankIDs(spkIdx));
    end
    
    %% Get cluster info after manual curation too (phy)
    %%% Would need a check that it has actually been curated
    
    if exist(fullfile(KSFolder, 'cluster_group.tsv'),'file') 
       cgsFile = fullfile(KSFolder, 'cluster_group.tsv');
       [cids, cgs] = readClusterGroupsCSV(cgsFile);
    end 
    
    if (numel(cgs) == numel(cgs_KS)) && all(cgs == cgs_KS)
        manuallyCurated = 0;
    else
        manuallyCurated = 1;
    end
    
    if manuallyCurated
        clu = readNPY(fullfile(KSFolder, 'spike_clusters.npy')); 
        
        clus_KSLabels = zeros(1,numel(cids),'uint8');
        clus_xpos = zeros(1,numel(cids),'single');
        clus_depths = zeros(1,numel(cids),'single');
        clus_shankIDs = zeros(1,numel(cids),'uint8');
        for ii = 1:numel(cids)
            temp = cids(ii);
            spkIdx = clu == temp;
            clus_KSLabels(ii) = cgs(cids == temp);
            clus_xpos(ii) = nanmedian(spikeXPos(spkIdx)); % not sure why there can be nans here
            clus_depths(ii) = nanmedian(spikeDepths(spkIdx)); 
            clus_shankIDs(ii) = nanmedian(spikeShankIDs(spkIdx)); 
        end
    end


    %% Save it in spk
    
    % spikes
    spk.spikes.times = st;
    spk.spikes.templates = spikeTemplates;
    spk.spikes.amps = tempScalingAmps;
    spk.spikes.depths = spikeDepths;
    spk.spikes.av_xpos = spikeXPos;
    spk.spikes.av_shankIDs = spikeShankIDs; 
    
    % templates
    spk.templates.av_IDs = cids_KS;
    spk.templates.av_KSLabels = temp_KSLabels;
    spk.templates.waveforms = temps; % maybe to remove/redundant with qMetrics?
    spk.templates.waveformsChannels = temps_ind; % maybe to remove/redundant with qMetrics?
    spk.templates.depths = temp_depths;
    spk.templates.av_xpos = temp_xpos;
    spk.templates.av_shankIDs = temp_shankIDs;
    
    if manuallyCurated
        % spikes
        spk.spikes.clusters = clu;
        
        % clusters (that can be manually curated)
        spk.clusters.av_IDs = cids;
        spk.clusters.depths = clus_depths;
        spk.clusters.av_xpos = clus_xpos;
        spk.clusters.av_shankID = clus_shankIDs;
        spk.clusters.av_Labels = clus_KSLabels;
    end
    
end
    