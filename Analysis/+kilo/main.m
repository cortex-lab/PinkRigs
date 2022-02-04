function main(varargin)
    %%% This function will run the main kilosorting code, and save the
    %%% results in the sorting folder.
    
    %% Get parameters and list of mice to check
    % Parameters for processing (can be inputs in varargin{1})
    recompute = 0;
    
    % This is not ideal
    if ~isempty(varargin)
        params = varargin{1};
        
        if ~isempty(params) && isfield(params, 'recompute')
            recompute = params.recompute;
        end
        
        if nargin > 1
            rec2sortList = varargin{2};
            % could check if in CSV and if not update CSV?
        end
    end
    
    if ~exist('rec2sortList', 'var')
        % Get all the recordings in the queue
        KSqueueCSVLoc = getCSVLocation('kilosort_queue');
        recList = readtable(KSqueueCSVLoc,'Delimiter',',');
        if ~recompute
            compIdx = find(recList.sortedTag == 0);
        else
            compIdx = 1:numel(recList.sortedTag);
        end
        rec2sortList = recList.ephysName(compIdx);
    end
    
    %% Get folders path etc.
    % These things won't be used anywhere else, so should be alright.
    
    % Local Kilosort working folder (rootH)
    % On SSD, where we process the data for whitening
    KSWorkFolder = 'C:\Users\Experiment\Documents\KSworkfolder';      
    if ~exist(KSWorkFolder, 'dir')
        mkdir(KSWorkFolder)
    end
        
    % Local Kilosort output folder (rootZ)
    KSOutFolder = 'C:\Users\Experiment\Documents\kilosort'; % local temporal folder for output
    if ~exist(KSOutFolder, 'dir')
        mkdir(KSOutFolder)
    end
        
    % KS2 config file
    pathToKSConfigFile = 'C:\Users\Experiment\Documents\Github\AV_passive\preprocessing\configFiles_kilosort2';
    if ~exist(pathToKSConfigFile, 'dir')
        error('Can''t find the path to the KS2 config files.')
    end
    
    % Path to the defaults P3B chan map
    defaultP3Bchanmap = 'C:\Users\Experiment\Documents\Github\AV_passive\preprocessing\configFiles_kilosort2\neuropixPhase3B2_kilosortChanMap.mat';

    %% Go through experiments to sort
    
    for rr = 1:numel(rec2sortList)
        recName = rec2sortList{rr};

        [ephysPath,b,c] = fileparts(recName);
        ephysFileName = strcat(b,c);
        
        % Get meta data 
        metaData = readMetaData_spikeGLX(ephysFileName,ephysPath);
        
        % Get channel map
        if contains(metaData.imDatPrb_type,'0')
            % phase 3B probe -- just load the default kilosort map
            chanMapPath = defaultP3Bchanmap;
        elseif contains(metaData.imDatPrb_type,'2')
            % create channelmap (good for all phase2, even single shank) or copy P3B map?
            fprintf('creating custom channelmap...')
            kilo.create_channelmapMultishank(recName,ephysPath,1);
            chanMapFile = dir(fullfile(ephysPath, '**','*_channelmap.mat*'));
            chanMapPath = fullfile(chanMapFile(1).folder, chanMapFile(1).name); % channelmap for the probe - should be in the same folder
        end
        
        %% Copying to local folder
        if ~exist(fullfile(KSOutFolder, ephysFileName),'file')
            fprintf('Copying data to local folder...')
            copyfile(recName,fullfile(KSOutFolder,ephysFileName));
            fprintf('Local copy done.\n')
        else
            disp('Data already copied.');
        end
        
        try
            %% Running the main algorithm
            kilo.runMatKilosort2(KSOutFolder,KSWorkFolder,chanMapPath,pathToKSConfigFile)
            
            %% Running quality metrics
            kilo.getQualityMetrics(KSOutFolder, KSOutFolder)
            
            %% Copying file to distant server
            delete([KSOutFolder '\' ephysFileName]); % delete .bin file from KS output
            movefile(KSOutFolder,fullfile(ephysPath,'kilosort2')) % copy KS output back to server
            
            %% Overwrite the queue
            recList.sortedTag(compIdx(rr)) = 1;
        catch
            % Sorting was not successful: write a permanent tag indicating that
            recList.sortedTag(compIdx(rr)) = -1;
            
            % Save error message.
            errorMsge = jsonencode(lasterror);
            fid = fopen([ephysPath '\KSerror.json'], 'w');
            fprintf(fid, '%s', errorMsge);
            fclose(fid);
        end
        
        % Save the updated queue
        writetable(recList,KSqueueCSVLoc,'Delimiter',',');
    end
    close all
    
    