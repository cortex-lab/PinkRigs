function getQualityMetrics(ksFolder,ephysFolder)

    %% Get KS output
    [spikeTimes, spikeTemplates, ...
        templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx, channelPositions] = bc_loadEphysData(ksFolder);
    
    %% Set paths
    savePath = ksFolder;
    ephysRawFile = dir(fullfile(ephysFolder,'*ap.cbin'));
    ephysap_path = fullfile(ephysRawFile.folder,ephysRawFile.name);

    %% quality metric parameters and thresholds
    param = bc_qualityParamValues(ksFolder,ephysap_path);

    %% compute quality metrics
    [qMetric, unitType] = bc_runAllQualityMetrics(param, spikeTimes, spikeTemplates, ...
        templateWaveforms, templateAmplitudes,pcFeatures,pcFeatureIdx,channelPositions, savePath);
    
    %% save
    bc_saveQMetrics(qMetric,param,savePath)