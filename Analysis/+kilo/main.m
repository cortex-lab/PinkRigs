function main(varargin)
    %%% This function will run the main kilosorting code, and save the
    %%% results in the sorting folder.
    %%% Inputs can be a set of parameters (input 1) and/or a list of
    %%% recordings (input 2), given by their paths.
    
    %% Get parameters and list of mice to check
    % Parameters for processing (can be inputs in varargin{1})
    params.recomputeKilo = 0;
    params.recomputeQMetrics = 0; % made the two independent
    params.checkTime = 0;
    
    % This is not ideal
    if ~isempty(varargin)
        paramsIn = varargin{1};
        params = parseInputParams(params,paramsIn);
        
        if numel(varargin) > 1
            rec2sortList = varargin{2};
            %%% Could check if in CSV and if not update CSV?
            %%% Might also need to generate recList here?
        end
    end
    
    if ~exist('rec2sortList', 'var')
        % Get all the recordings in the queue
        KSqueueCSVLoc = getCSVLocation('kilosort_queue');
        recList = readtable(KSqueueCSVLoc,'Delimiter',',');
        if ~params.recomputeKilo
            compIdx = find(recList.sortedTag == 0);
        else
            compIdx = 1:numel(recList.sortedTag);
        end
        rec2sortList = recList.ephysName(compIdx);
    end
    
    % get day on which the script has been started
    startClock = datetime('now');
    
    %% Get folders path etc.
    % These things won't be used anywhere else, so should be alright.
    
    % Local Kilosort working folder (rootH)
    % On SSD, where we process the data for whitening
    KSWorkFolder = 'C:\Users\Experiment\Documents\KSworkfolder';      
    if ~exist(KSWorkFolder, 'dir')
        mkdir(KSWorkFolder)
    end
        
    % Local Kilosort output folder (rootZ)
    KSOutFolderLocGen = 'C:\Users\Experiment\Documents\kilosort'; % local temporal folder for output
    if ~exist(KSOutFolderLocGen, 'dir')
        mkdir(KSOutFolderLocGen)
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
        
        KSOutFolderLoc = fullfile(KSOutFolderLocGen,regexprep(ephysFileName(1:end-7),'\.','_'));
        KSOutFolderServer = fullfile(ephysPath,'kilosort2');
        
        if params.checkTime
            % To avoid running too long. Will stop after ~20h + 1 last 
            % processing.
            nowClock = datetime('now');
            if nowClock > startClock + 20/24
                return
            end
        end
        
        if exist(KSOutFolderServer,'dir') && ~isempty(dir(fullfile(KSOutFolderServer,'rez.mat'))) && ~params.recomputeKilo
            fprintf('Ephys %s already sorted.\n', ephysFileName)
            successKS = 1;
        else
            try
                %% Getting meta data
                % Get meta data
                metaData = readMetaData_spikeGLX(ephysFileName,ephysPath);
                
                % Get channel map
                if contains(metaData.imDatPrb_type,'0')
                    % phase 3B probe -- just load the default kilosort map
                    chanMapPath = defaultP3Bchanmap;
                elseif contains(metaData.imDatPrb_type,'2')
                    % create channelmap (good for all phase2, even single shank) or copy P3B map?
                    fprintf('Creating custom channelmap...\n')
                    kilo.create_channelmapMultishank(recName,ephysPath,1);
                    chanMapFile = dir(fullfile(ephysPath, '**','*_channelmap.mat*'));
                    chanMapPath = fullfile(chanMapFile(1).folder, chanMapFile(1).name); % channelmap for the probe - should be in the same folder
                end
                
                %% Copy data to local folder.
                if ~exist(fullfile(KSOutFolderLoc, ephysFileName),'file')
                    fprintf('Copying data to local folder...')
                    
                    if ~exist(KSOutFolderLoc, 'dir')
                        mkdir(KSOutFolderLoc)
                    end
                    success = copyfile(recName,fullfile(KSOutFolderLoc,ephysFileName));
                    if success
                        fprintf('Local copy done.\n')
                    end
                else
                    fprintf('Data already copied.\n');
                    success = 1;
                end
                
                if ~success
                    error('Couldn''t copy data to local folder.')
                else
                    %% Running the main algorithm
                    fprintf('Running kilosort...\n')
                    kilo.runMatKilosort2(KSOutFolderLoc,KSWorkFolder,chanMapPath,pathToKSConfigFile)
                    fprintf('Kilosort done.\n')
                    
                    %% Copying file to distant server
                    fprintf('Copying to server (and deleting local copy)...\n')
                    delete(fullfile(KSOutFolderLoc, ephysFileName)); % delete .bin file from KS output
                    successKS = movefile(fullfile(KSOutFolderLoc,'*'),KSOutFolderServer); % copy KS output back to server
                    
                    if ~successKS
                        error('Error when copying data to server.')
                    else
                        fprintf('Copying done.\n')
                        
                        % Delete any error file related to KS
                        if exist(fullfile(ephysPath, 'KSerror.json'))
                            delete(fullfile(ephysPath, 'KSerror.json'));
                        end
                    end
                end
                
            catch me
                successKS = 0;
                
                % Save error message.
                errorMsgeKS = jsonencode(me.message);
                fid = fopen(fullfile(ephysPath, 'KSerror.json'), 'w');
                fprintf(fid, '%s', errorMsgeKS);
                fclose(fid);
            end
                    
            if exist(KSOutFolderLoc,'dir')
                % Delete data otherwise will crowd up
                rmdir(KSOutFolderLoc); % delete whole folder whatever happens
            end
        end
        
        if successKS
            if ~exist(fullfile(KSOutFolderServer,'qMetrics.m')) || params.recomputeQMetrics
                %% Running quality metrics (directly on the server)
                % Independent of previous block to be able to run one or the
                % other (it's likely we might want to recompute only the qM)
                fprintf('Running quality metrics...\n')
                try
                    kilo.getQualityMetrics(KSOutFolderServer, ephysPath)
                    successFinal = 1;
                    
                    if exist(fullfile(ephysPath, 'QMerror.json'),'file')
                        delete(fullfile(ephysPath, 'QMerror.json'))
                    end
                catch me
                    successFinal = -2; % fails at quality metrics stage
                    
                    % Save error message.
                    errorMsgeQM = jsonencode(me.message);
                    fid = fopen(fullfile(ephysPath, 'QMerror.json'), 'w');
                    fprintf(fid, '%s', errorMsgeQM);
                    fclose(fid);
                end
            else
                successFinal = 1;
            end
        else 
            successFinal = -1;
        end
                
        if exist('recList','var')
            %% Overwrite the queue
            recList.sortedTag(compIdx(rr)) = successFinal; % 1 for all done / -1 for KS failed / -2 for qMetrics failed / 0 for not done
            
            % Save the updated queue
            try
                writetable(recList,KSqueueCSVLoc,'Delimiter',',');
            catch
                warning('Can''t write the csv (it must be open somewhere?). Will do next time?')
            end
        end
    end
    close all
    
    