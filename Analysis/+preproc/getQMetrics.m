function tab = getQMetrics(KSFolder)
    %%% Will load and compute quality metrics

    tab = readtable(fullfile(KSFolder,'ibl_format','cluster_metrics.csv'));

    % Maybe add some waveform based qmetrics?
end