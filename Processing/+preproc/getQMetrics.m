function tab = getQMetrics(KSFolder,type)
    %% Loads the IBL quality metrics.
    %
    % Parameters:
    % -------------------
    % KSFolder: str
    %   Path to kilosort folder.
    %
    % Returns: 
    % -------------------
    % tab: table
    %   Contains all the quality metrics for that specific recording.

    switch type
        case 'IBL'
            tab = readtable(fullfile(KSFolder,'ibl_format','cluster_metrics.csv'));
        case 'bombcell'
            tab = parquetread(fullfile(KSFolder,'qMetrics','templates._bc_qMetrics.parquet'));
    end

end