function runBombcell(varargin)
    %% Runs Bombcell on a list of experiments.
    %
    % Parameters:
    % -------------------
    % Classic PinkRigs inputs (optional).
    % recompute (optional): bool
    %   Whether to recompute 
    % KSversion (optional): str
    %   Version of kilosort to look at (usually PyKS).
    % decompressDataLocal (optional): str
    %   Path to local folder for decompression.
    
    %% Get parameters
    varargin = ['recompute', {false}, varargin];
    varargin = ['KSversion', 'PyKS', varargin];
    varargin = ['decompressDataLocal', 'C:\Users\Experiment\Documents\KSworkfolder', varargin];
    params = csv.inputValidation(varargin{:});
    
    %% Get exp list

    exp2checkList = csv.queryExp(params);

    % Remove the ones with repeated recordings
    recList = cat(1,exp2checkList.ephysPathProbe0,exp2checkList.ephysPathProbe1);
    recList = unique(recList);
    recList(strcmp(recList,'NaN')) = [];

    %% Run Bombcell on all exp
    decompressDataLocal = params.decompressDataLocal{1};
    if ~exist(decompressDataLocal, 'dir')
        mkdir(decompressDataLocal)
    end

    for rec = 1:numel(recList)
        % Set paths
        ephysKilosortPath = fullfile(recList{rec},'PyKS','output');
        ephysDirPath = recList{rec};
        ephysRawDir = dir(fullfile(ephysDirPath,'*.*bin'));
        ephysMetaDir = dir(fullfile(ephysDirPath,'*.meta')); % used in bc_qualityParamValues
        savePath = fullfile(ephysKilosortPath,'qMetrics');

        % Load data
        [spikeTimes_samples, spikeTemplates, ...
            templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx, channelPositions] = bc_loadEphysData(ephysKilosortPath);

        % Detect whether data is compressed, decompress locally if necessary
        rawFile = bc_manageDataCompression(ephysRawDir, decompressDataLocal);

        % Which quality metric parameters to extract and thresholds
        param = bc_qualityParamValuesForUnitMatch(ephysMetaDir, rawFile);
        
        % Compute quality metrics
        rerun = 0;
        qMetricsExist = ~isempty(dir(fullfile(savePath, 'templates._bc_qMetrics.parquet')));

        if qMetricsExist == 0 || rerun
            [qMetric, unitType] = bc_runAllQualityMetrics(param, spikeTimes_samples, spikeTemplates, ...
                templateWaveforms, templateAmplitudes,pcFeatures,pcFeatureIdx,channelPositions, savePath);
        else
            [param, qMetric] = bc_loadSavedMetrics(savePath);
            unitType = bc_getQualityUnitType(param, qMetric);
        end

        % Delete local file if ran fine
        delete(rawFile);
    end

end


