function bc_saveQMetrics(qMetric,param,savePath)

    if ~exist(savePath,'dir')
        mkdir(fullfile(savePath))
    end
    disp([newline, 'saving quality metrics to ', savePath])

    qMetricSummary = table('Size',[length(qMetric.clusterID), 10],'VariableTypes',...
        {'double', 'double', 'double', 'double', 'double', 'double', 'double',...
        'double','double','double'},'VariableNames',...
        {'percSpikesMissing', 'clusterID', 'Fp', 'nSpikes', 'nPeaks', 'nTroughs', 'somatic', ...
        'waveformDuration', 'spatialDecaySlope', 'waveformBaseline'});
    qMetricSummary.clusterID = qMetric.clusterID';
    %qMetricSummary.percSpikesMissing = arrayfun(@(x) nanmean(qMetric.percSpikesMissing(qMetric.useTheseTimes{x})), 1:size(qMetric.percSpikesMissing,1));

    qMetricSummary.percSpikesMissing = arrayfun(@(x) nanmean(qMetric.percSpikesMissing(x, qMetric.percSpikesMissing(x, :) <= param.maxPercSpikesMissing)), ...
        1:size(qMetric.percSpikesMissing, 1))';
    qMetricSummary.Fp = arrayfun(@(x) nanmean(qMetric.Fp(x, qMetric.Fp(x, :) <= param.maxRPVviolations)), ...
        1:size(qMetric.percSpikesMissing, 1))';

    qMetricSummary.percSpikesMissing = arrayfun( @(x) nanmean(qMetric.percSpikesMissing(x, qMetric.percSpikesMissing(x,:) <= param.maxPercSpikesMissing)), ...
        1:size(qMetric.percSpikesMissing,1))';
    qMetricSummary.Fp = arrayfun( @(x) nanmean(qMetric.Fp(x, qMetric.Fp(x,:) <= param.maxRPVviolations)), ...
        1:size(qMetric.percSpikesMissing,1))';
    qMetricSummary.nSpikes = qMetric.nSpikes';
    qMetricSummary.nPeaks = qMetric.nPeaks';
    qMetricSummary.nTroughs = qMetric.nTroughs';
    qMetricSummary.somatic = qMetric.somatic';
    qMetricSummary.waveformDuration = qMetric.waveformDuration';
    qMetricSummary.spatialDecaySlope = qMetric.spatialDecaySlope';
    qMetricSummary.waveformBaseline = qMetric.waveformBaseline';

    % Save matlab file
    % save(fullfile(savePath, 'qMetric.mat'), 'qMetric', 'param', '-v7.3')
    
    % Save parquet files
    parquetwrite([fullfile(savePath, '_jf_parameters._jf_qMetrics.parquet')], struct2table(param))
    parquetwrite([savePath, filesep, 'templates._jf_qMetrics.parquet'], qMetricSummary)
    parquetwrite([savePath filesep 'templates._jf_qMetrics.parquet'],qMetricSummary)
end