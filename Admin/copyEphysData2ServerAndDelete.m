function copyEphysData2ServerAndDelete(localFolder)
%% This funtion will need to be run at the end of each experiment/day? and
if ~exist('localFolder', 'var'); localFolder = 'D:\ephysData'; end

fprintf('Starting now %s...\n',datestr(now))
localEphysFolders = dir([localFolder '\**\*']);
localEphysFolders = localEphysFolders([localEphysFolders.isdir]' & ...
    contains({localEphysFolders.name}', 'imec') & ...
    ~contains({localEphysFolders.name}', '.'));

<<<<<<< HEAD
% find all folders with both ap.cbin files and sync.mat
localCompressed = cell2mat(cellfun(@(x) dir([localFolder '\**\*' x]), {'.ap.cbin'}, 'uni', 0));
localSync = cell2mat(cellfun(@(x) dir([localFolder '\**\*' x]), {'sync.mat'}, 'uni', 0));
completeFolders = intersect({localCompressed.folder}', {localSync.folder}');



localEphysFiles = localCompressed(contains({localCompressed.folder}', completeFolders));
=======
localEphysPaths = arrayfun(@(x) [x.folder filesep x.name], localEphysFolders, 'uni', 0);
localEphysPathContents = cellfun(@dir, localEphysPaths, 'uni', 0);
localEphysFolders(cellfun(@(x) isempty(x) | all([x.isdir]), localEphysPathContents)) = [];
>>>>>>> 64bb2eded05282cfe33aaa0d3687af529dc3fe00

subjectFromFolder = arrayfun(@(x) x.name(1:5), localEphysFolders, 'uni', 0);
dateFromFolder = arrayfun(@(x) cell2mat(regexp(x.name, '\d\d\d\d-\d\d-\d\d', 'match')), localEphysFolders, 'uni', 0);

% NOTE: This is specific to SpikeGLX output... maybe there is a better way
splitFolders = arrayfun(@(x) regexp([x.folder filesep x.name],'\','split'), localEphysFolders, 'uni', 0);
serverFolders = cellfun(@(x,y) getExpPath(x,y), subjectFromFolder, dateFromFolder, 'uni', 0);
serverFolders = cellfun(@(x,y) fullfile(x, 'ephys', y(end-1), y(end)), serverFolders, splitFolders);
localFolders = arrayfun(@(x) [x.folder filesep x.name], localEphysFolders, 'uni', 0);

% Check that sync and compressed files exist either on server or locally
localCompressed = cell2mat(cellfun(@(x) ~isempty(dir([x '\*.ap.cbin'])), localFolders, 'uni', 0));
serverCompressed = cell2mat(cellfun(@(x) ~isempty(dir([x '\*.ap.cbin'])), serverFolders, 'uni', 0));
localSync = cell2mat(cellfun(@(x) ~isempty(dir([x '\*sync.mat'])), localFolders, 'uni', 0));
serverSync = cell2mat(cellfun(@(x) ~isempty(dir([x '\*sync.mat'])), serverFolders, 'uni', 0));
readyFolders = (localCompressed | serverCompressed) & (localSync | serverSync);

if isempty(readyFolders)
    fprintf('There are no ephys files in the local directory. Returning... \n');
    pause(1);
    return;
end 
localFolders = localFolders(readyFolders);
serverFolders = serverFolders(readyFolders);

%% 
allLocalFiles = cellfun(@(x) dir(x), localFolders, 'uni', 0);
serverFolders = cellfun(@(x,y) num2cell(repmat(x,length(y),1),2), serverFolders, allLocalFiles, 'uni', 0);
serverFolders = vertcat(serverFolders{:});
allLocalFiles = cell2mat(allLocalFiles);

allLocalFilePaths = arrayfun(@(x) fullfile(x.folder, x.name), allLocalFiles, 'uni', 0);
allServerFilePaths = arrayfun(@(x,y) fullfile(y{1}, x.name), allLocalFiles, serverFolders, 'uni', 0);

% put .cbin files first
cbinIdx = find(arrayfun(@(x) contains(x,'.cbin'), allLocalFilePaths));
otherIdx = find(arrayfun(@(x) ~contains(x,'.cbin'), allLocalFilePaths));
allLocalFilePaths = allLocalFilePaths([cbinIdx; otherIdx]);
allServerFilePaths = allServerFilePaths([cbinIdx; otherIdx]);

%Sanity check to make sure that the files are in the correct order
allLocalFilePathsTest = cellfun(@(x) x(max(strfind(x, '\'))+1:end), allLocalFilePaths, 'uni', 0);
allServerFilePathsTest = cellfun(@(x) x(max(strfind(x, '\'))+1:end), allServerFilePaths, 'uni', 0);
matchTest = all(cellfun(@(x,y) strcmp(x,y), allLocalFilePathsTest, allServerFilePathsTest));
if ~matchTest
    error('File paths names do not correspond..?')
end

copyFiles2ServerAndDelete(allLocalFilePaths, allServerFilePaths, 1)
%%
cleanEmptyFoldersInDirectory(localFolder);

fprintf('Stopping now %s.',datestr(now))

end
