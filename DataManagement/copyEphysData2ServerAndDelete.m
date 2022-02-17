function copyEphysData2ServerAndDelete(ignoreSubjectMismatch)
%% This funtion will need to be run at the end of each experiment/day? and
if ~exist('ignoreSubjectMismatch', 'var'); ignoreSubjectMismatch = 0; end 
%% identify data
localFolder ='D:\ephysData'; % the localExpData folder where data is held
% find all folders with bin files
localEphysFiles = cell2mat(cellfun(@(x) dir([localFolder '\**\*' x]), {'.ap.bin'}, 'uni', 0));
if isempty(localEphysFiles)
    fprintf('There are no ephys files in the local directory. Returning... \n');
    return;
end
metaData = arrayfun(@(x) readMetaData_spikeGLX(x.name, x.folder), localEphysFiles, 'uni', 0);
%%
subjectFromBinName = arrayfun(@(x) x.name(1:5), localEphysFiles, 'uni', 0);
dateFromBinName = arrayfun(@(x) cell2mat(regexp(x.name, '\d\d\d\d-\d\d-\d\d', 'match')), localEphysFiles, 'uni', 0);

fprintf('Checking dates in file name against file creation dates... \n')
dateFromFileInfo = cellfun(@(x) datestr(x, 'yyyy-mm-dd'), {localEphysFiles.date}', 'uni', 0);
dateMismatch = cellfun(@(x,y) ~strcmp(x,y), dateFromFileInfo, dateFromBinName);

if any(dateMismatch)
    cellfun(@(x) fprintf('Date mismatch for %s. Skipping... \n', x), {localEphysFiles(dateMismatch).name});
else
    fprintf('All dates match file names. Nice! \n');
end

fprintf('Checking subject in file name against probe serials in CSV... \n')
serialsFromMeta = cellfun(@(x) str2double(x.imDatPrb_sn), metaData);

[uniqueProbes, ~, uniIdx] = unique(serialsFromMeta);
matchedSubjects = getCurrentSubjectFromProbeSerial(uniqueProbes);

expectedSubject = matchedSubjects(uniIdx);
if any(cellfun(@isempty, expectedSubject))
    warning('At least one probe failed to match in CSV and will give subject mismatches');
end

subjectMismatch = cellfun(@(x,y) ~strcmpi(x,y), subjectFromBinName, expectedSubject);
if any(subjectMismatch)
    cellfun(@(x) fprintf('Subject mismatch for %s. Skipping... \n', x), {localEphysFiles(subjectMismatch).name});
else
    fprintf('All expected subjects match file names. Nice! \n');
end

%%
if ignoreSubjectMismatch && any(subjectMismatch)
    warning('Ignoring subject mismatch..?!?!')
    subjectMismatch = subjectMismatch*0;
    expectedSubject = subjectFromBinName;
end
validIdx = ~subjectMismatch & ~dateMismatch;
validEphysFiles = localEphysFiles(validIdx);
validSubjects = expectedSubject(validIdx);
validDates = dateFromBinName(validIdx);

% NOTE: This is specific to SpikeGLX output... maybe there is a better way
splitFolders = arrayfun(@(x) regexp(x.folder,'\','split'), validEphysFiles, 'uni', 0);
serverFolders = cellfun(@(x,y,z) getExpPath(x,y), validSubjects, validDates, 'uni', 0);
serverFolders = cellfun(@(x,y) fullfile(x, 'ephys', y(end-1), y(end)), serverFolders, splitFolders);

allLocalFiles = arrayfun(@(x) dir(x.folder), validEphysFiles, 'uni', 0);
serverFolders = cellfun(@(x,y) num2cell(repmat(x,length(y),1),2), serverFolders, allLocalFiles, 'uni', 0);
serverFolders = vertcat(serverFolders{:});
allLocalFiles = cell2mat(allLocalFiles);
%%
allLocalFilePaths = arrayfun(@(x) fullfile(x.folder, x.name), allLocalFiles, 'uni', 0);
allServerFilePaths = arrayfun(@(x,y) fullfile(y{1}, x.name), allLocalFiles, serverFolders, 'uni', 0);
copyFiles2ServerAndDelete(allLocalFilePaths, allServerFilePaths, 1)
%%
cleanEmptyFoldersInDirectory(localFolder);
end
