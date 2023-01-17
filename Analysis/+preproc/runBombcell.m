function runBombcell(varargin)
    %%% This function will run Bombcell on each recording.
    
    %% Get parameters
    varargin = ['recompute', {false}, varargin];
    varargin = ['KSversion', 'PyKS', varargin];
    varargin = ['decompressDataLocal', 'C:\Users\Experiment\Documents\KSworkfolder', varargin];
    params = csv.inputValidation(varargin{:});
    
    %% Get all recordings
    
    recList = csv.getRecordingPathFromExp(params);

    %% Loop over all ephys files

    for ep = 1:numel(recList)
        %% set paths
        ephysKilosortPath = recList{ep};
        ephysDirPath = fileparts(fileparts(recList{ep}));
        ephysRawDir = dir(fullfile(ephysDirPath,'*.*bin'));
        ephysMetaDir = dir(fullfile(ephysDirPath,'*.meta')); % used in bc_qualityParamValues
        decompressDataLocal = params.decompressDataLocal{1};
        if ~exist(decompressDataLocal, 'dir')
            mkdir(decompressDataLocal)
        end
        savePath = fullfile(ephysKilosortPath,'qMetrics');

        %% load data
        [spikeTimes_samples, spikeTemplates, ...
            templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx, channelPositions] = bc_loadEphysData(ephysKilosortPath);

        %% which quality metric parameters to extract and thresholds
        bc_qualityParamValues;

        %% detect whether data is compressed, decompress locally if necessary
        decompDataFile = dir([decompressDataLocal, filesep, ephysRawDir.name(1:end-14), '_bc_decompressed', ephysRawDir.name(end-13:end-8),'.ap.bin']);
        if strcmp(ephysRawDir.name(end-4:end), '.cbin') && isempty(decompDataFile)
            fprintf('Decompressing ephys data file %s locally to %s... \n', ephysRawDir.name, decompressDataLocal)

            decompDataFile = bc_extractCbinData([ephysRawDir.folder, filesep, ephysRawDir.name],...
                [], [], [], decompressDataLocal);
            param.rawFile = decompDataFile;
        elseif strcmp(ephysRawDir.name(end-4:end), '.cbin') && ~isempty(decompDataFile)
            fprintf('Using previously decompressed ephys data file in %s ... \n', decompressDataLocal)

            param.rawFile = decompDataFile;
        else
            param.rawFile = [ephysRawDir.folder, filesep, ephysRawDir.name];
        end

        %% compute quality metrics
        qMetricsExist = ~isempty(dir(fullfile(savePath, 'qMetric*.mat'))) || ~isempty(dir(fullfile(savePath, 'templates._bc_qMetrics.parquet')));

        if qMetricsExist == 0 || params.recompute{1}
            bc_runAllQualityMetrics(param, spikeTimes_samples, spikeTemplates, ...
                templateWaveforms, templateAmplitudes,pcFeatures,pcFeatureIdx,channelPositions, savePath);
        end
    end

end


