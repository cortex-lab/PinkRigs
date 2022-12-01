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

    if isempty(localEphysFiles)
        log = appendAndPrint(log, 'There are no ephys files that are ready in the local directory. Returning... \n');
        pause(1);
        return;
    end

    % Get their metadata
    metaData = arrayfun(@(x) readMetaData_spikeGLX(x.name, x.folder), localEphysFiles, 'uni', 0);

    %% Check subject mismatches
    subjectFromBinName = arrayfun(@(x) x.name(1:5), localEphysFiles, 'uni', 0);
    dateFromBinName = arrayfun(@(x) cell2mat(regexp(x.name, '\d\d\d\d-\d\d-\d\d', 'match')), localEphysFiles, 'uni', 0);

    log = appendAndPrint(log, 'Checking dates in file name against file creation dates... \n');
    dateFromFileInfo = cellfun(@(x) datestr(x, 'yyyy-mm-dd'), {localEphysFiles.date}', 'uni', 0);
    dateMismatch = cellfun(@(x,y) ~strcmp(x,y), dateFromFileInfo, dateFromBinName);

    if any(dateMismatch)
        tmplog = cellfun(@(x) sprintf('Date mismatch for %s. Skipping... \n', x), {localEphysFiles(dateMismatch).name}, 'uni', 0);
        log = appendAndPrint(log, strcat(tmplog{:}));
    else
        log = appendAndPrint(log, 'All dates match file names. Nice! \n');
    end

    log = appendAndPrint(log, 'Checking subject in file name against probe serials in CSV... \n');
    serialsFromMeta = cellfun(@(x) str2double(x.imDatPrb_sn), metaData);

    [uniqueProbes, ~, uniIdx] = unique(serialsFromMeta);
    probeInfo = csv.checkProbeUse(uniqueProbes,'last');
    matchedSubjects = probeInfo.implantedSubjects;

    expectedSubject = matchedSubjects(uniIdx);
    if any(cellfun(@isempty, expectedSubject))
        log = appendAndPrint(log, 'At least one probe failed to match in CSV and will give subject mismatches.\n');
    end

    subjectMismatch = cellfun(@(x,y) ~strcmpi(x,y), subjectFromBinName, expectedSubject);
    if any(subjectMismatch)
        tmplog = cellfun(@(x) sprintf('Subject mismatch for %s. Skipping... \n', x), {localEphysFiles(subjectMismatch).name}, 'uni', 0);
        log = appendAndPrint(log, strcat(tmplog{:}));
    else
        log = appendAndPrint(log, 'All expected subjects match file names. Nice! \n');
    end

    %% Get valid files

    if ignoreSubjectMismatch && any(subjectMismatch)
        log = appendAndPrint(log, 'Ignoring subject mismatch..?!?!');
        subjectMismatch = subjectMismatch*0;
    end
    validIdx = ~subjectMismatch; %& ~dateMismatch;
    validEphysFiles = localEphysFiles(validIdx);
    
    %% Loop through files to extract sync and compress them
    for i = 1:length(validEphysFiles)
        binFileName = fullfile(validEphysFiles(i).folder, validEphysFiles(i).name);
        log = appendAndPrint(log, sprintf('Looking at file %s.\n', ...
                regexprep(validEphysFiles(i).name,'\','/')));
            
        % Extracting the sync
        syncPath = fullfile(validEphysFiles(i).folder, 'sync.mat');
        if ~exist(syncPath, 'file')
            log = appendAndPrint(log, 'Couldn''t find the sync file. Computing it.\n');
            metaS = readMetaData_spikeGLX(validEphysFiles(i).name,validEphysFiles(i).folder);
            extractSync(binFileName, str2double(metaS.nSavedChans));
        end
        
        % Comnpressing the .bin file
        cbinFileName = regexprep(binFileName,'.bin','.cbin');
        chFileName = regexprep(binFileName,'.bin','.ch');
        if ~exist(cbinFileName, 'file') || ~exist(chFileName, 'file') 
            log = appendAndPrint(log, 'Compressing it...\n');
            [statusComp,resultComp] = system(['conda activate PinkRigs && ' ...
                'python ' compressPath ' ' ...
                binFileName ' && ' ...
                'conda deactivate']);
            if statusComp > 0
                log = appendAndPrint(log, sprintf('Failed with error: %s.\n', regexprep(resultComp,'\','/')));
            else
                log = appendAndPrint(log, sprintf('%s.\n',regexprep(resultComp,'\','/')));
                % Deleting .bin
                if exist(cbinFileName, 'file') && exist(chFileName, 'file')
                    delete(binFileName)
                    log = appendAndPrint(log, 'Deleted bin file.');
                else
                    log = appendAndPrint(log, 'Something weird happened: compression ran fine but the .cbin or the .ch file is missing. Recheck?\n');
                end
            end
        else
            log = appendAndPrint(log, 'Already compressed! Check why .bin file hasn''t been deleted?\n');
        end

    end
    
    % Just clean up the log 
    log = regexprep(log,'Skipping...','Skipping...\n');
end