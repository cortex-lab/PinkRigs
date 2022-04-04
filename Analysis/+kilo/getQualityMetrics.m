function getQualityMetrics(ksFolder,ephysFolder)

    %% Get KS output
    [spikeTimes, spikeTemplates, ...
        templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx, channelPositions] = bc_loadEphysData(ksFolder);
    
    %% Set paths
    savePath = fullfile(ksFolder,'qualityMetrics.mat');;
    ephysRawFile = dir(fullfile(ephysFolder,'*ap.bin'));
    ephysap_path = fullfile(ephysRawFile.folder,ephysRawFile.name);

    %% quality metric parameters and thresholds
    param = bc_qualityParamValues;

    %% compute quality metrics
    [qMetric, unitType] = bc_runAllQualityMetrics(param, spikeTimes, spikeTemplates, ...
        templateWaveforms, templateAmplitudes,pcFeatures,pcFeatureIdx,channelPositions, savePath);
    
    %% save
    save(savePath,'qMetric','unitType','param', '-v7.3')