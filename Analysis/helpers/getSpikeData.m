function spk = getSpikeData(ephysPath,varargin)
    %%% This function will align load the spike data (spike time, clusters,
    %%% but also cluster info, etc.). 
    %%%
    %%% This code is inspired by the code from kilotrode
    %%% (https://github.com/cortex-lab/kilotrodeRig).

    params.loadPCs = 0;
    if ~isempty(varargin)
        paramsIn = varargin{1};
        params = parseInputParams(params,paramsIn);
    end
    
    % Load spike data
    ksDir = fullfile(ephysPath,'kilosort2');

    % Get param.py
    spk = loadParamsPy(fullfile(ksDir, 'params.py'));

    % Get spike times
    ss = readNPY(fullfile(ksDir, 'spike_times.npy'));
    st = double(ss)/spk.sample_rate;
    spikeTemplates = readNPY(fullfile(ksDir, 'spike_templates.npy')); % note: zero-indexed
    
    if exist(fullfile(ksDir, 'spike_clusters.npy'),'file')
        clu = readNPY(fullfile(ksDir, 'spike_clusters.npy'));
    else
        clu = spikeTemplates;
    end
    
    tempScalingAmps = readNPY(fullfile(ksDir, 'amplitudes.npy'));
    
    if params.loadPCs
        pcFeat = readNPY(fullfile(ksDir,'pc_features.npy')); % nSpikes x nFeatures x nLocalChannels
        pcFeatInd = readNPY(fullfile(ksDir,'pc_feature_ind.npy')); % nTemplates x nLocalChannels
    else
        pcFeat = [];
        pcFeatInd = [];
    end
    
    cgsFile = '';
    if exist(fullfile(ksDir, 'cluster_groups.csv'))
        cgsFile = fullfile(ksDir, 'cluster_groups.csv');
    end
    if exist(fullfile(ksDir, 'cluster_group.tsv'))
        cgsFile = fullfile(ksDir, 'cluster_group.tsv');
    end
    if ~isempty(cgsFile)
        [cids, cgs] = readClusterGroupsCSV(cgsFile);
        
        if params.excludeNoise
            noiseClusters = cids(cgs==0);
            
            st = st(~ismember(clu, noiseClusters));
            spikeTemplates = spikeTemplates(~ismember(clu, noiseClusters));
            tempScalingAmps = tempScalingAmps(~ismember(clu, noiseClusters));
            
            if params.loadPCs
                pcFeat = pcFeat(~ismember(clu, noiseClusters), :,:);
                %pcFeatInd = pcFeatInd(~ismember(cids, noiseClusters),:);
            end
            
            clu = clu(~ismember(clu, noiseClusters));
            cgs = cgs(~ismember(cids, noiseClusters));
            cids = cids(~ismember(cids, noiseClusters));
            
            
        end
        
    else
        clu = spikeTemplates;
        
        cids = unique(spikeTemplates);
        cgs = 3*ones(size(cids));
    end
    
    
    coords = readNPY(fullfile(ksDir, 'channel_positions.npy'));
    ycoords = coords(:,2); xcoords = coords(:,1);
    temps = readNPY(fullfile(ksDir, 'templates.npy'));
    
    winv = readNPY(fullfile(ksDir, 'whitening_mat_inv.npy'));
    
    spk.st = st;
    spk.spikeTemplates = spikeTemplates;
    spk.clu = clu;
    spk.tempScalingAmps = tempScalingAmps;
    spk.cgs = cgs;
    spk.cids = cids;
    spk.xcoords = xcoords;
    spk.ycoords = ycoords;
    spk.temps = temps;
    spk.winv = winv;
    spk.pcFeat = pcFeat;
    spk.pcFeatInd = pcFeatInd;