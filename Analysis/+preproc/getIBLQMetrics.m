function tab = getIBLQMetrics(KSFolder)
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

    tab = readtable(fullfile(KSFolder,'ibl_format','cluster_metrics.csv'));
end