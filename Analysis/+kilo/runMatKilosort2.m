function runMatKilosort2(rootZ,rootH,chanMapFile,KSconfig)
    %%% This function will run the matlab version of kilosort 2.
    % rootZ is the raw data binary file is in this folder
    % rootH is the path to temporary binary file (same size as data, should be on fast SSD)
    % pathToYourConfigFile: take from Github folder and put it somewhere else (together with the master_file)
    
    %% Get options
    
    pathToYourConfigFile = KSconfig;

    ops.trange = [0 Inf]; % time range to sort
    ops.NchanTOT    = 385; % total number of channels in your recording % CHANGED THIS TO 385
    
    run(fullfile(pathToYourConfigFile, 'configFile384.m'))
    ops.fproc       = fullfile(rootH, 'temp_wh.dat'); % proc file on a fast SSD
    ops.chanMap = chanMapFile;

    %% This block runs all the steps of the algorithm
    
    fprintf('Looking for data inside %s \n', rootZ)
    
    % find the binary file
    fs          = [dir(fullfile(rootZ, '*.bin')) dir(fullfile(rootZ, '*.dat'))];
    ops.fbinary = fullfile(rootZ, fs(1).name);
    
    % preprocess data to create temp_wh.dat
    rez = preprocessDataSub(ops);
    
    % time-reordering as a function of drift
    rez = clusterSingleBatches(rez);
    
    % saving here is a good idea, because the rest can be resumed after loading rez
    save(fullfile(rootZ, 'rez.mat'), 'rez', '-v7.3');
    
    % main tracking and template matching algorithm
    rez = learnAndSolve8b(rez);
    
    % final merges
    rez = find_merges(rez, 1);
    
    % final splits by SVD
    rez = splitAllClusters(rez, 1);
    
    % final splits by amplitudes
    rez = splitAllClusters(rez, 0);
    
    % decide on cutoff
    rez = set_cutoff(rez);
    
    fprintf('found %d good units \n', sum(rez.good>0))
    
    % write to Phy
    fprintf('Saving results to Phy  \n')
    rezToPhy(rez, rootZ);
    
    %% if you want to save the results to a Matlab file...
    %%% Won't be needed for now?
    
%     % discard features in final rez file (too slow to save)
%     rez.cProj = [];
%     rez.cProjPC = [];
%     
%     % final time sorting of spikes, for apps that use st3 directly
%     [~, isort]   = sortrows(rez.st3);
%     rez.st3      = rez.st3(isort, :);
%     
%     % Ensure all GPU arrays are transferred to CPU side before saving to .mat
%     rez_fields = fieldnames(rez);
%     for i = 1:numel(rez_fields)
%         field_name = rez_fields{i};
%         if(isa(rez.(field_name), 'gpuArray'))
%             rez.(field_name) = gather(rez.(field_name));
%         end
%     end
%     
%     % save final results as rez2
%     fprintf('Saving final results in rez2  \n')
%     fname = fullfile(rootZ, 'rez2.mat');
%     save(fname, 'rez', '-v7.3');
end