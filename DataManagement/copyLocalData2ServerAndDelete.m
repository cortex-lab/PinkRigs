function copyLocalData2ServerAndDelete
%% This funtion will need to be run at the end of each experiment/day? and
%% identify data
localFolder ='D:\LocalExpData'; % the localExpData folder where data is held
% find all folders with a relevant file like timeline
files2Check = {'Timeline.mat';'block.mat';'eyeCam*';'frontCam*';'sideCam*';'mic.mat'};
localDat = cell2mat(cellfun(@(x) dir([localFolder '\**\*' x]), files2Check, 'uni', 0));

%% push the data to server
% check whether it has already been copied
localFolders = unique({localDat.folder})';

splitFolders = cellfun(@(x) regexp(x,'\','split'), localFolders, 'uni', 0)';
subjects = cellfun(@(x) x{end-2}, splitFolders, 'uni', 0)';
expDates = cellfun(@(x) x{end-1}, splitFolders, 'uni', 0)';
expNums = cellfun(@(x) x{end-0}, splitFolders, 'uni', 0)';

serverFolders = cellfun(@(x,y,z) getExpPath(x,y,z), subjects, expDates, expNums, 'uni', 0);
allLocalFiles = cellfun(@dir, localFolders, 'uni', 0);
serverFolders = cellfun(@(x,y) repmat(x,length(y),1), serverFolders, allLocalFiles, 'uni', 0);
serverFolders = num2cell(cell2mat(serverFolders),2);

allLocalFiles = cell2mat(allLocalFiles);
allServerFilePaths = arrayfun(@(x,y) fullfile(y{1}, x.name), allLocalFiles, serverFolders, 'uni', 0);

isDirectory = [allLocalFiles.isdir];
allLocalFiles = allLocalFiles(~isDirectory);
allServerFilePaths = allServerFilePaths(~isDirectory);

copiedAlready = cellfun(@(x) exist(x,'file'), allServerFilePaths)>0;
if all(copiedAlready)
    fprintf('All data is already copied .. \n')
else
    files2copy = find(~copiedAlready);
    for i = 1:length(files2copy)
        cIdx = files2copy(i);
        fprintf('Copying %s ... \n', allLocalFiles(cIdx).name);
        data2Copy = fullfile(allLocalFiles(cIdx).folder, allLocalFiles(cIdx).name);, 
        serverTarget = fileparts(allServerFilePaths{cIdx}); 
    try
        copyfile(data2Copy,serverTarget);
    catch
        fprintf('WARNING: Problem copying file %s. Skipping.... \n', data2Copy);
    end
    end
end

allServerFiles = cellfun(@dir, allServerFilePaths, 'uni', 0);
failedCopy = cellfun(@isempty, allServerFiles);
allLocalFiles(failedCopy) = []; 
allServerFiles = cell2mat(allServerFiles);

%% Deletions
% delete files that have been copied correctly
oldIdx = ([allLocalFiles(:).datenum]<=now-2)';
sizeMismatch = ([allLocalFiles(:).bytes]~=[allServerFiles(:).bytes])';

toDelete = allLocalFiles(oldIdx & ~sizeMismatch);
arrayfun(@(x) delete(fullfile(x.folder, x.name)), toDelete);

% Clean up empty folders
folderList = dir([localFolder '\**\*']);
folderList = folderList(~ismember({folderList(:).name} ,{'.','..'}));

emptyFolders = folderList([folderList(:).isdir] & [folderList(:).bytes]<5);
emptyFolders = arrayfun(@(x) fullfile(x.folder, x.name), emptyFolders, 'uni', 0);
emptyFolders = flipud(unique(emptyFolders));    
cellfun(@rmdir, emptyFolders);
end
