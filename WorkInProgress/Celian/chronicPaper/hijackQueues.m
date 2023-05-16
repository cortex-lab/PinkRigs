queuePath = '\\zinu.cortexlab.net\Subjects\PinkRigs\Helpers';

%% Get recordings
folder = '\\zaru.cortexlab.net\Subjects\EB019';
d = dir(fullfile(folder,'**','*ap.*bin'));

alreadySorted = false(1,numel(d));
for dd = 1:numel(d)
    if exist(fullfile(d(dd).folder,'pyKS'))
        alreadySorted(dd) = true;
    end
end
dunsorted = d(~alreadySorted);
dsorted = d(alreadySorted);

%% Update pyKS queue

M = csv.readTable(fullfile(queuePath, 'pykilosort_queue.csv'));
s = size(M,1);
for dd = 1:numel(dunsorted)
    M(s+dd,1) = {fullfile(dunsorted(dd).folder, dunsorted(dd).name)};
    M(s+dd,2) = {'0.0'};
end
csv.writeTable(M,fullfile(queuePath, 'pykilosort_queue.csv'));

%% Update IBL format queue

M = csv.readTable(fullfile(queuePath, 'ibl_formatting_queue.csv'));
s = size(M,1);
for dd = 1:numel(dsorted)
    pyKSpath = fullfile(dsorted(dd).folder, 'pyKS');
    M(s+dd,1) = {pyKSpath};
    M(s+dd,2) = {'0.0'};
end
csv.writeTable(M,fullfile(queuePath, 'ibl_formatting_queue.csv'));

%% Run bombcell

decompressDataLocal = 'C:\Users\Experiment\Documents\KSworkfolder';
if ~exist(decompressDataLocal, 'dir')
    mkdir(decompressDataLocal)
end

for rec = 1:numel(dsorted)
    % Set paths
    ephysKilosortPath = fullfile(dsorted(dd).folder,'PyKS','output');
    ephysDirPath = dsorted(dd).folder;
    ephysRawDir = dir(fullfile(ephysDirPath,'*.*bin'));
    if numel(ephysRawDir)>1
        idx = find(contains({ephysRawDir.name},'.cbin'));
        if ~isempty(idx) && numel(idx)==1
            ephysRawDir = ephysRawDir(idx);
        end
    end
    ephysMetaDir = dir(fullfile(ephysDirPath,'*ap.meta')); % used in bc_qualityParamValues
    savePath = fullfile(ephysKilosortPath,'qMetrics');

    qMetricsExist = ~isempty(dir(fullfile(savePath, 'templates._bc_qMetrics.parquet')));
    sortingExist = ~isempty(dir(fullfile(ephysKilosortPath,'spike_templates.npy')));

    if sortingExist && (qMetricsExist == 0 || recompute)
        % Load data
        [spikeTimes_samples, spikeTemplates, ...
            templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx, channelPositions] = bc_loadEphysData(ephysKilosortPath);

        % Detect whether data is compressed, decompress locally if necessary
        rawFile = bc_manageDataCompression(ephysRawDir, decompressDataLocal);

        % Which quality metric parameters to extract and thresholds
        param = bc_qualityParamValuesForUnitMatch(ephysMetaDir, rawFile);

        % Compute quality metrics
        param.plotGlobal = 0;
        bc_runAllQualityMetrics(param, spikeTimes_samples, spikeTemplates, ...
            templateWaveforms, templateAmplitudes,pcFeatures,pcFeatureIdx,channelPositions, savePath);

        % Delete local file if ran fine
        delete(rawFile);
    end
end