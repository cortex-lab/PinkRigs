function main(varargin)
    %%% This function will run the main kilosorting code, and save the
    %%% results in the sorting folder.
    %%% Inputs can be a set of parameters (input 1) and/or a list of
    %%% recordings (input 2), given by their paths.
    
    %% Get parameters and list of mice to check
    % Parameters for processing (can be inputs in varargin{1})
    params.recomputeKilo = 0;
    params.recomputeQMetrics = 0; % made the two independent
    params.runFor = inf; % in hour
    
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
        KSqueueCSVLoc = csv.getLocation('kilosort_queue');
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
    pathToKSConfigFile = 'C:\Users\Experiment\Documents\GitHub\PinkRigs\Analysis\helpers\ephys\KSconfig';
    if ~exist(pathToKSConfigFile, 'dir')
        error('Can''t find the path to the KS2 config files.')
    end
    
    % Path to the defaults P3B chan map
    defaultP3Bchanmap = 'C:\Users\Experiment\Documents\GitHub\PinkRigs\Analysis\helpers\ephys\KSconfig\neuropixPhase3B2_kilosortChanMap.mat';

    %% Go through experiments to sort
    
    for rr = 1:numel(rec2sortList)
        recName = rec2sortList{rr};

        [ephysPath,b,c] = fileparts(recName);
        ephysFileName = strcat(b,c);
        
        % Plot and save recording sites
        ephysParentFolderName = fileparts(ephysPath);
        plotRecordingSites({ephysParentFolderName},1)
        
        KSOutFolderLoc = fullfile(KSOutFolderLocGen,regexprep(ephysFileName(1:end-7),'\.','_'));
        KSOutFolderServer = fullfile(ephysPath,'kilosort2');
        
        % To avoid running too long. 
        nowClock = datetime('now');
        if nowClock > startClock + params.runFor/24
            return
        end
        
        if exist(KSOutFolderServer,'dir') && ~isempty(dir(fullfile(KSOutFolderServer,'rez.mat'))) && ~params.recomputeKilo
            fprintf('Ephys %s already sorted.\n', ephysFileName)
            successKS = 1;
            successQM = nan;
        else            
            try
                if exist('recList','var')
                    % Indicate it's being processed
                    recList.sortedTag(compIdx(rr)) = 0.5; % for 'currently processing'
                    writetable(recList,KSqueueCSVLoc,'Delimiter',',');
                end
                
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
                    %% Decompress it if needed.
                    if strcmp(ephysFileName(end-3:end),'cbin')
                        cbinFile = ephysFileName;
                        chFile = regexprep(ephysFileName,'.cbin','.ch');
                        
                        % Copy locally the ch file that hasn't yet been
                        % copied
                        copyfile(regexprep(recName,'.cbin','.ch'),fullfile(KSOutFolderLoc,chFile));
                        
                        % Decompress 
                        fprintf('Decompressing the data...\n')
                        decompressPath = which('decompress_data.py');
                        [statusDecomp,resultDecomp] = system(['conda activate PinkRigs && ' ...
                            'python ' decompressPath ' ' ...
                            fullfile(KSOutFolderLoc,cbinFile) ' ' ...
                            fullfile(KSOutFolderLoc,chFile) ' && ' ...
                            'conda deactivate']);
                        if statusDecomp > 0 
                            error('Issue with decompression.')
                        end
                        
                        ephysFileName = regexprep(ephysFileName,'.cbin','.bin');
                    end
                    
                    %% Running the main algorithm
                    fprintf('Running kilosort...\n')
                    kilo.runMatKilosort2(KSOutFolderLoc,KSWorkFolder,chanMapPath,pathToKSConfigFile)
                    fprintf('Kilosort done.\n')
                    
                    %% Running quality metrics
                    % Have to do in while the raw data is decompressed
                    fprintf('Running quality metrics...\n')
                    try
                        kilo.getQualityMetrics(KSOutFolderServer, fullfile(KSOutFolderLoc,ephysFileName))
                        
                        if exist(fullfile(ephysPath, 'QMerror.json'),'file')
                            delete(fullfile(ephysPath, 'QMerror.json'))
                        end
                        successQM = 1;
                    catch me
                        successQM = 0;
                        
                        % Save error message locally
                        saveErrMess(me.message,fullfile(ephysPath, 'QMerror.json'))
                    end
                    
                    %% Copying file to distant server
                    fprintf('Copying to server (and deleting local copy)...\n')
                    delete(fullfile(KSOutFolderLoc, ephysFileName)); % delete .bin file from KS output
                    successKS = movefile(fullfile(KSOutFolderLoc,'*'),KSOutFolderServer); % copy KS output back to server
                    
                    if ~successKS
                        error('Error when copying data to server.')
                    else
                        fprintf('Copying done.\n')
                        
                        % Delete any error file related to KS
                        if exist(fullfile(ephysPath, 'KSerror.json'),'file')
                            delete(fullfile(ephysPath, 'KSerror.json'));
                        end
                    end
                end
                
            catch me
                successKS = 0;
                
                % Save error message locally
                saveErrMess(me.message,fullfile(ephysPath, 'KSerror.json'))
            end
                    
            if exist(KSOutFolderLoc,'dir')
                try
                    %%% Have to "try" for now because sometimes issue when
                    %%% there's a KS error above...
                    % Delete data otherwise will crowd up
                    rmdir(KSOutFolderLoc, 's'); % delete whole folder whatever happens
                catch
                    warning('Can''t delete KSout local folder.. Will crowd up.')
                end
            end
        end
        
        if successKS & isnan(successQM)
            if ~exist(fullfile(KSOutFolderServer,'qMetrics.m')) || params.recomputeQMetrics
                %% Running quality metrics (directly on the server)
                % Independent of previous block to be able to run this
                % without redoing the KSing.
                % Will crash if raw waveforms haven't been extracted.
                fprintf('Running quality metrics...\n')
                try
                    kilo.getQualityMetrics(KSOutFolderServer, ephysPath)
                    successFinal = 1;
                    
                    if exist(fullfile(ephysPath, 'QMerror.json'),'file')
                        delete(fullfile(ephysPath, 'QMerror.json'))
                    end
                catch me
                    successFinal = -2; % fails at quality metrics stage
                    
                    % Save error message locally
                    saveErrMess(me.message,fullfile(ephysPath, 'QMerror.json'))
                end
            else
                successFinal = 1;
            end
        else 
            if successKS
                if successQM
                    successFinal = 1;
                else
                    successFinal = -2;
                end
            else
                successFinal = -1;
            end
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
        
        %% Update the csv for associated experiments

        % Get experments
        [partmp.mice2Check, partmp.days2Check, ~] = parseExpPath(ephysPath);
        expList = csv.queryExp(partmp);
        
        % Update
        for ee = 1:numel(expList)
            [subject, expDate, expNum] = parseExpPath(expList(1,:).expFolder{1});
            csv.updateRecord(subject, expDate, expNum);
        end
    end
    close all
    
    