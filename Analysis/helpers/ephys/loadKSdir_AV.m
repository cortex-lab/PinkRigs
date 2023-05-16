function spikeStruct = loadKSdir_AV(ksDir, params)
    %% Fetches all spikes information from KS folder
    % Function taken from the spikes repository 
    % adapted to take the output of pykilosort even on stitched data
    %
    % Parameters:
    % -------------------
    % ksDir: str
    %   Path to KS directory.
    % params: struct
    %   Parameter structure containing optional fields:
    %       excludeNoise: excludes the noisy units
    %       loadPCs: will load the spikesorting's PC features
    %       uncorrected_time: wether to load corrected time for stitched
    %       recordings
    %
    % Returns: 
    % -------------------
    % spikeStruct: struct
    %   Structure containing all relevant spikes information.
    %     st: spike times
    %     spikeTemplates: associated template
    %     clu: associated cluster
    %     tempScalingAmps: scaling factor of the template for each spike
    %     cgs: labels (0=noise, 1=mua, 2=good, 3=unsorted or can't find it)
    %     cids: associated cluster ID
    %     xcoords: channels x coordinate
    %     ycoords: channels y coordinate
    %     temps: templates
    %     winv: inverse of the whitening matrix
    %     pcFeat: PC features
    %     pcFeatInd: PC features indices
    %     spk_dataset_idx: introduced variable in the stitched dataset
       
    if ~exist('params','var')
        params = [];
    end
    
    if ~isfield(params, 'excludeNoise')
        params.excludeNoise = true;
    end
    if ~isfield(params, 'loadPCs')
        params.loadPCs = false;
    end
    
    if isfield(params, 'uncorrected_time')
        uncorrected_time = true;
    else
        uncorrected_time = false;
    end
    % load spike data
    
    spikeStruct = loadParamsPy(fullfile(ksDir, 'params.py'));
    
    % introduced to be able to load stitched data 
    if exist(fullfile(ksDir, 'spike_times_corrected.npy')) && (uncorrected_time==false)
        ss = readNPY(fullfile(ksDir, 'spike_times_corrected.npy'));
    else
        ss = readNPY(fullfile(ksDir, 'spike_times.npy'));
    end 
    st = double(ss)/spikeStruct.sample_rate;
    spikeTemplates = readNPY(fullfile(ksDir, 'spike_templates.npy')); % note: zero-indexed
    
    if exist(fullfile(ksDir, 'spike_clusters.npy'))
        clu = readNPY(fullfile(ksDir, 'spike_clusters.npy'));
    else
        clu = spikeTemplates;
    end
    
    %introduced variable in the stitched dataset
    if exist(fullfile(ksDir, 'spike_datasets.npy'))
        spk_dataset_idx = readNPY(fullfile(ksDir, 'spike_datasets.npy')); 
    else
        spk_dataset_idx = ones(numel(st),1); 
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
    if isempty(cgsFile)
       cgsFile = fullfile(ksDir, 'cluster_KSLabel.tsv');
    end
    if ~isempty(cgsFile)
        [cids, cgs] = readClusterGroupsCSV_AV(cgsFile);
    
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
    
    spikeStruct.st = st;
    spikeStruct.spikeTemplates = spikeTemplates;
    spikeStruct.clu = clu;
    spikeStruct.tempScalingAmps = tempScalingAmps;
    spikeStruct.cgs = cgs;
    spikeStruct.cids = cids;
    spikeStruct.xcoords = xcoords;
    spikeStruct.ycoords = ycoords;
    spikeStruct.temps = temps;
    spikeStruct.winv = winv;
    spikeStruct.pcFeat = pcFeat;
    spikeStruct.pcFeatInd = pcFeatInd;
    spikeStruct.spk_dataset_idx=spk_dataset_idx; 
end