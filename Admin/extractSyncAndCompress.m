function log = extractSyncAndCompress(localFolder, ignoreSubjectMismatch)

    if ~exist('localFolder', 'var'); localFolder = 'D:\ephysData'; end
    if ~exist('ignoreSubjectMismatch', 'var'); ignoreSubjectMismatch = 0; end

    % Get path for Python compression script
    compressPath = which('compress_data.py');

    % Get log
    log = '';
    
    %% Build list of files to sync-extract and compress

    %% Get files that have been modified more than 1h ago
    localEphysFiles = dir(fullfile(localFolder, '/**/*.ap.bin'));
    localEphysFilesAgeInMins = (now-[localEphysFiles.datenum]')*24*60;
    localEphysFiles(localEphysFilesAgeInMins < 60) = []; 

    % Get their metadata
    metaData = arrayfun(@(x) readMetaData_spikeGLX(x.name, x.folder), localEphysFiles, 'uni', 0);

    %% Check subject mismatches
    subjectFromBinName = arrayfun(@(x) x.name(1:5), localEphysFiles, 'uni', 0);
    dateFromBinName = arrayfun(@(x) cell2mat(regexp(x.name, '\d\d\d\d-\d\d-\d\d', 'match')), localEphysFiles, 'uni', 0);

    log = append(log, 'Checking dates in file name against file creation dates... \n');
    dateFromFileInfo = cellfun(@(x) datestr(x, 'yyyy-mm-dd'), {localEphysFiles.date}', 'uni', 0);
    dateMismatch = cellfun(@(x,y) ~strcmp(x,y), dateFromFileInfo, dateFromBinName);

    if any(dateMismatch)
        tmplog = cellfun(@(x) sprintf('Date mismatch for %s. Skipping... \n', x), {localEphysFiles(dateMismatch).name}, 'uni', 0);
        log = append(log, strcat(tmplog{:}));
    else
        log = append(log, 'All dates match file names. Nice! \n');
    end

    log = append(log, 'Checking subject in file name against probe serials in CSV... \n');
    serialsFromMeta = cellfun(@(x) str2double(x.imDatPrb_sn), metaData);

    [uniqueProbes, ~, uniIdx] = unique(serialsFromMeta);
    probeInfo = csv.checkProbeUse(uniqueProbes,'last');
    matchedSubjects = probeInfo.implantedSubjects;

    expectedSubject = matchedSubjects(uniIdx);
    if any(cellfun(@isempty, expectedSubject))
        log = append(log, 'At least one probe failed to match in CSV and will give subject mismatches.\n');
    end

    subjectMismatch = cellfun(@(x,y) ~strcmpi(x,y), subjectFromBinName, expectedSubject);
    if any(subjectMismatch)
        tmplog = cellfun(@(x) sprintf('Subject mismatch for %s. Skipping... \n', x), {localEphysFiles(subjectMismatch).name}, 'uni', 0);
        log = append(log, strcat(tmplog{:}));
    else
        log = append(log, 'All expected subjects match file names. Nice! \n');
    end

    %% Get valid files

    if ignoreSubjectMismatch && any(subjectMismatch)
        log = append(log, 'Ignoring subject mismatch..?!?!');
        subjectMismatch = subjectMismatch*0;
    end
    validIdx = ~subjectMismatch; %& ~dateMismatch;
    validEphysFiles = localEphysFiles(validIdx);
    
    %% Loop through files to extract sync and compress them
    for i = 1:length(validEphysFiles)
        binFileName = fullfile(validEphysFiles(i).folder, validEphysFiles(i).name);
        
        % Extracting the sync
        syncPath = fullfile(validEphysFiles(i).folder, 'sync.mat');
        if ~exist(syncPath, 'file')
            log = append(log, 'Couldn''t find the sync file for %s. Computing it.\n', ...
                validEphysFiles(i).name);
            metaS = readMetaData_spikeGLX(validEphysFiles(i).name,validEphysFiles(i).folder);
            extractSync(binFileName, str2double(metaS.nSavedChans));
        end
        
        % Comnpressing the .bin file
        log = append(log, sprintf('Compressing file %s...',binFileName));
        [statusComp,resultComp] = system(['conda activate PinkRigs && ' ...
            'python ' compressPath ' ' ...
            binFileName ' && ' ...
            'conda deactivate']);
        if statusComp > 0
            log = append(log, sprintf('Failed with error: %s', resultComp));
        else
            log = append(log, sprintf('%s.\n',regexprep(resultComp,'\','/')));
            % Deleting .bin
            delete(binFileName)
        end

    end
    
    % Just clean up the log 
    log = sprintf(regexprep(log,'Skipping...','Skipping...\n'));
end