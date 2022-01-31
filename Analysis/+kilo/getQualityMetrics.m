function getQualityMetrics(ksFolder,ephysFolder)
    
    %% First get params
    getBombcellParams
    
    %% File name
    qMetricsFile = fullfile(ksFolder,'qualityMetrics.mat');
    
    param.rawFolder =  ephysFolder; % for bombcell
    
    %% Load everything
    templates = readNPY([ksFolder, filesep, 'templates.npy']);
    channel_positions = readNPY([ksFolder, filesep, 'channel_positions.npy']);
    spike_times_seconds = double(readNPY([ksFolder, filesep, 'spike_times.npy']))./30000;
    spike_times = double(readNPY([ksFolder, filesep, 'spike_times.npy'])); % sample rate hard-coded as 30000 - should load this in from params
    spike_templates = readNPY([ksFolder, filesep, 'spike_templates.npy']) + 1; % 0-idx -> 1-idx
    template_amplitude = readNPY([ksFolder, filesep, 'amplitudes.npy']);
    spike_clusters = readNPY([ksFolder, filesep, 'spike_clusters.npy']) + 1;
    pc_features = readNPY([ksFolder, filesep, 'pc_features.npy']) ;
    pc_feature_ind = readNPY([ksFolder, filesep, 'pc_feature_ind.npy']) + 1;
    
    %% Run qualityMetrics
    [qMetric, goodUnits] = bc_runAllQualityMetrics(param, spike_times, spike_templates, ...
        templates, template_amplitude,pc_features,pc_feature_ind);
    
    goodUnits = qMetric.percSpikesMissing <= param.maxPercSpikesMissing & qMetric.nSpikes > param.minNumSpikes & ...
        qMetric.nPeaks <= param.maxNPeaks & qMetric.nTroughs <= param.maxNTroughs & qMetric.Fp <= param.maxRPVviolations & ...
        qMetric.rawAmplitude > param.minAmplitude;
    
    %% save
    save(qMetricsFile,'qMetric','goodUnits')