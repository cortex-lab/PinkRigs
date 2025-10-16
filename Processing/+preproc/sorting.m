function sorting(varargin)
    %% Runs sorting/Bombcell/ibl formatting on a list of experiments.
    %
    % Parameters:
    % -------------------
    % Classic PinkRigs inputs (optional).
    % recompute (optional): bool
    %   Whether to recompute 
    % KSversion (optional): str
    %   Version of kilosort to look at
    % tmpDataFolder (optional): str
    %   Path to local folder for decompression and KS output.
    % recList (optional): cell
    %   List of recordings to process

    %% Get parameters
    varargin = ['recompute', {'none'}, varargin];
    varargin = ['KSversion', 'kilosort4', varargin];
    varargin = ['tmpDataFolder', 'C:\Users\Experiment\Documents\KSworkfolder', varargin];
    varargin = ['recList', {[]}, varargin];
    params = csv.inputValidation(varargin{:});

    recompute = params.recompute{1};
    KSversion = params.KSversion{1};
    
    %% Define paths

    tmpDataFolder = params.tmpDataFolder{1};

    %% Get recording list

    if isempty(params.recList{1})
        exp2checkList = csv.queryExp(params);

        % Remove the ones with repeated recordings
        recList = cat(1,exp2checkList.ephysPathProbe0,exp2checkList.ephysPathProbe1);
        recList = unique(recList);
        recList(strcmp(recList,'NaN')) = [];
    else
        recList = params.recList{1};
    end

    %% Loop across all recordings

    for rec = 1:numel(recList)
        if exist('exp2checkList', 'var')
            expRefList = exp2checkList(contains(exp2checkList.ephysPathProbe0, recList{rec}) | ...
                contains(exp2checkList.ephysPathProbe1, recList{rec}), :);
        end

        %% Check if has been computed

        % Set paths
        switch KSversion
            case 'kilosort4'
                KSFolder = 'kilosort4';
            case 'pyKS-IBL'
                KSFolder = 'pyKS';
        end

        serverKilosortPath = fullfile(recList{rec},KSFolder,'output');
%         serverKilosortPath = fullfile(recList{rec},KSFolder);
        ephysDirPath = recList{rec};
        ephysRawDir = dir(fullfile(ephysDirPath,'*.*bin'));
        if numel(ephysRawDir)>1
            idx = find(contains({ephysRawDir.name},'.cbin'));
            if ~isempty(idx) && numel(idx)==1
                ephysRawDir = ephysRawDir(idx);
            end
        end
        ephysMetaDir = dir(fullfile(ephysDirPath,'*ap.meta')); % used in bc_qualityParamValues
        metaFile = fullfile(ephysMetaDir.folder, ephysMetaDir.name);
        bombcellPath = fullfile(serverKilosortPath,'qMetrics');
        iblformatPath = fullfile(serverKilosortPath,'ibl_format');

        % check if exists
        sortingExist = ~isempty(dir(fullfile(serverKilosortPath,'spike_templates.npy'))) | exist(fullfile(serverKilosortPath, 'Kilosort_error.json'),'file');
        qMetricsExist = ~isempty(dir(fullfile(bombcellPath, 'templates._bc_qMetrics.parquet'))) | exist(fullfile(serverKilosortPath, 'Bombcell_error.json'),'file');
        iblformatExist = ~isempty(dir(fullfile(iblformatPath, 'cluster_metrics.csv'))) | exist(fullfile(serverKilosortPath, 'IBLformat_error.json'),'file');

        switch recompute
            % could also delete the old ones
            case {'all','sorting'}
                sortingExist = 0;
                qMetricsExist = 0;
                iblformatExist = 0;
            case 'qMetrics'
                qMetricsExist = 0;
            case 'iblFormat'
                iblformatExist = 0;
        end

        %% Run

        if ~sortingExist || ~qMetricsExist || ~iblformatExist
            
            tmpDataFolderRec = strrep(fullfile(tmpDataFolder, ephysRawDir.name),'.','_');

            if ~sortingExist || ~qMetricsExist
                %% Detect whether data is compressed, decompress locally if necessary

                
                if ~exist(tmpDataFolderRec,'dir')
                    mkdir(tmpDataFolderRec)
                end
                ephysRawFile = bc.dcomp.manageDataCompression(ephysRawDir, tmpDataFolderRec);

                % move meta file
                copyfile(fullfile(ephysMetaDir.folder, ephysMetaDir.name), regexprep(ephysRawFile,'\.bin','.meta'));
            end
%             ephysRawFile = fullfile(ephysRawDir.folder, ephysRawDir.name);

            if ~sortingExist
                %% Extract channel map

                [channelPos, probeSN, recordingduration] = ChannelIMROConversion(fullfile(ephysMetaDir.folder, ephysMetaDir.name),1,0);

                %% Run kilosort
                try
                    binFile = strrep(ephysRawFile,'\','/');
                    probeFile = strrep(strrep(metaFile,'.ap.meta','_kilosortChanMap.mat'),'\','/');
                    tic
                    switch KSversion
                        case 'kilosort4'
                            fprintf('Running kilosort4 on %s  (%s)...', ephysRawDir.name, datestr(now))
                            runpyKS = which("RunPyKS4_FromMatlab.py");
                            [statusKS,resultKS] = system(['activate kilosort && ' ...
                                'python ' runpyKS ' ' binFile ' ' probeFile ' && ' ...
                                'conda deactivate']);
                        case 'pyKS-IBL'
                            fprintf('Running pyKS-IBL on %s  (%s)...', ephysRawDir.name, datestr(now))
                            runpyKS = which("RunPyKS-IBL_FromMatlab.py");
                            [statusKS,resultKS] = system(['activate pyks2-ibl && ' ...
                                'python ' runpyKS ' ' binFile(1:end-4) ' && ' ...
                                'conda deactivate']);
                        otherwise
                            error('This version of kilosort isn''t supported yet.')
                    end
                    toc

                    if statusKS == 0
                        % Move files to server
                        localKilosortPath = fullfile(tmpDataFolderRec, KSFolder);
                        copyfile(localKilosortPath, serverKilosortPath);

                        % Rename the params.py
                        paramPath = fullfile(serverKilosortPath,'params.py');
                        fid = fopen(paramPath);
                        C = textscan(fid,'%s','delimiter','\n');
                        fclose(fid);
                        C{1}{1} = ['dat_path = ''' fullfile(ephysRawDir.folder, ephysRawDir.name) ''''];

                        fid = fopen(paramPath,'w');
                        for kk = 1:numel(C{1})
                            fprintf(fid,'%s\n',C{1}{kk});
                        end
                        fclose(fid);
                        sortingExist = 1;
                    else
                        error(sprintf('Spikesorting failed %s.', resultKS))
                    end
                catch me
                    msgText = getReport(me);
                    warning('Couldn''t run spikesorting: threw an error (%s)',msgText)

                    % Save error message locally
                    mkdir(serverKilosortPath)
                    saveErrMess(msgText,fullfile(serverKilosortPath, 'Kilosort_error.json'))

                    statusKS = 1;
                end

                %% Update csvs
                if exist('exp2checkList', 'var')
                    csv.updateRecord(expRefList);
                end
            else
                statusKS = 0;
                sortingExist = 1;
            end

            if ~qMetricsExist && sortingExist == 1
                %% Run Bombcell
                try
                    fprintf('Running bombcell (%s)...',datestr(now));

                    % Which quality metric parameters to extract and thresholds
                    param = bc.qm.qualityParamValuesForUnitMatch(ephysMetaDir, ephysRawFile, serverKilosortPath, nan, 4);

                    % Load data
                    [spikeTimes_samples, spikeTemplates, ...
                        templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx, channelPositions] = bc.load.loadEphysData(serverKilosortPath, bombcellPath);

                    % Compute quality metrics
                    param.plotGlobal = 0;
                    bc.qm.runAllQualityMetrics(param, spikeTimes_samples, spikeTemplates, ...
                        templateWaveforms, templateAmplitudes,pcFeatures,pcFeatureIdx,channelPositions, bombcellPath);

                    statusBombcell = 0;
                    fprintf('Done.\n');
                catch me
                    msgText = getReport(me);
                    warning('Couldn''t run bombcell: threw an error (%s)',msgText)

                    % Save error message locally
                    saveErrMess(msgText,fullfile(serverKilosortPath, 'Bombcell_error.json'))

                    statusBombcell = 1;
                end
            else
                statusBombcell = 0;
            end

            if ~iblformatExist && sortingExist == 1
                %% Run IBL formatting
                try
                    fprintf('Creating the ibl format (%s)...',datestr(now));
                    checkScriptPath = which('convert_to_ibl_format_single_file.py');
                    [statusIBL,resultIBL] = system(['activate iblenv && ' ...
                        'python ' checkScriptPath ' ' fileparts(fileparts(serverKilosortPath)) ' ' KSFolder '/output && ' ...
                        'conda deactivate']);
    
                    if statusIBL ~= 0
                        error('IBL conversion didn''t work.')
                    end
                    fprintf('Done.\n');
                catch me
                    msgText = getReport(me);
                    warning('Couldn''t run IBL formatting: threw an error (%s)',msgText)

                    % Save error message locally
                    saveErrMess(msgText,fullfile(serverKilosortPath, 'IBLformat_error.json'))
                end
            else
                statusIBL = 0;
            end

            %% Delete things

            if statusKS == 0 && statusBombcell == 0 && statusIBL == 0 && exist(tmpDataFolderRec','dir')
                % Delete the whole folder
                try
                    rmdir(tmpDataFolderRec,'s');
                end
            end
        end
    end