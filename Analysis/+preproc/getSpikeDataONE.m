function spk = getSpikeDataONE(ephysPath,KSFolder)
    %%% This function will load the spike data (spike time, templates,
    %%% but also template info, etc.). "spk" contains a condensed summary
    %%% with spikes times & template info.
    %%%
    %%% This code is inspired by the code from kilotrode
    %%% (https://github.com/cortex-lab/kilotrodeRig).

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
    
    % Load pc features
    pcFeat = readNPY(fullfile(KSFolder,'pc_features.npy')); % nSpikes x nFeatures x nLocalChannels
	pcFeatInd = readNPY(fullfile(KSFolder,'pc_feature_ind.npy')); % nTemplates x nLocalChannels

    % Load KS labels
	cgsFile = fullfile(KSFolder, 'cluster_KSLabel.tsv');
    [cids_KS, cgs_KS] = readClusterGroupsCSV(cgsFile); % cids should be the same as unique(spikeTemplates)??
    
  
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
    spikeShankIDs = spikeShankIDs-1;

    
    %% Get cluster info
    
    % don't have template amplitude here?
    temps = readNPY(fullfile(KSFolder, 'templates.npy'));
    temps_ind = readNPY(fullfile(KSFolder, 'templates_ind.npy'));
    
    %%% clusterList not exactly the same as sp.cids..???
    templatesList = unique(spikeTemplates);
    temp_KSLabels = nan(1,numel(templatesList));
    temp_xpos = nan(1,numel(templatesList));
    temp_depths = nan(1,numel(templatesList));
    temp_shankIDs = nan(1,numel(templatesList));

    
    for ii = 1:numel(templatesList)
        temp = templatesList(ii);
        spkIdx = spikeTemplates == temp;

        temp_KSLabels(ii) = cgs_KS(cids_KS == temp);
        temp_xpos(ii) = nanmedian(spikeXPos(spkIdx)); % not sure why there can be nans here
        temp_depths(ii) = nanmedian(spikeDepths(spkIdx)); 
        temp_shankIDs(ii) = nanmedian(spikeShankIDs(spkIdx)); 
    end
    

    % get cluster info after manual curation too (phy)
    if exist(fullfile(KSFolder, 'cluster_group.tsv')) 
       cgsFile = fullfile(KSFolder, 'cluster_group.tsv');
       [cids, cgs] = readClusterGroupsCSV(cgsFile);
    end 
    clu = readNPY(fullfile(KSFolder, 'spike_clusters.npy')); 

    % Remove noise clusters 
    noiseTemplates = cids(cgs==0);
    st = st(~ismember(spikeTemplates, noiseTemplates));
    tempScalingAmps = tempScalingAmps(~ismember(spikeTemplates, noiseTemplates));
    pcFeat = pcFeat(~ismember(spikeTemplates, noiseTemplates), :,:);
    spikeTemplates = spikeTemplates(~ismember(spikeTemplates, noiseTemplates));
    pcFeatInd = pcFeatInd(~ismember(cids, noiseTemplates),:);
    cgs = cgs(~ismember(cids, noiseTemplates));
    cids = cids(~ismember(cids, noiseTemplates));

    for ii = 1:numel(cids)
        temp = cids(ii);
        spkIdx = clu == temp;

        clus_KSLabels(ii) = cgs(cids == temp);
        clus_xpos(ii) = nanmedian(spikeXPos(spkIdx)); % not sure why there can be nans here
        clus_depths(ii) = nanmedian(spikeDepths(spkIdx)); 
        clus_shankIDs(ii) = nanmedian(spikeShankIDs(spkIdx)); 
    end


    %% Save it in spk
    
    % spikes
    spk.spikes.times = st;
    spk.spikes.templates = spikeTemplates;
    spk.spikes.amps = tempScalingAmps;
    spk.spikes.depths = spikeDepths;
    spk.spikes.av_xpos = spikeXPos;
    spk.spikes.av_shankIDs = spikeShankIDs;
    spk.spikes.clusters = clu; 
    
    % templates
    spk.templates.av_IDs = templatesList;
    spk.templates.av_KSLabels = temp_KSLabels;
    spk.templates.waveforms = temps; % maybe to remove/redundant with qMetrics?
    spk.templates.waveformsChannels = temps_ind; % maybe to remove/redundant with qMetrics?
    spk.templates.depths = temp_depths;
    spk.templates.av_xpos = temp_xpos;
    spk.templates.av_shankIDs = temp_shankIDs;

    %clusters (that can be manually curated)
    spk.clusters.av_IDs = cids; 
    spk.clusters.depths = clus_depths; 
    spk.clusters.av_xpos = clus_xpos; 
    spk.clusters.av_shankID = clus_shankIDs;
    spk.clusters.av_Labels = clus_KSLabels;

    
    