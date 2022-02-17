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

if isempty(allLocalFiles)
    fprintf('NOTE: No files in local folder ... will clean any empty folders \n');
else
    serverFolders = cellfun(@(x,y) num2cell(repmat(x,length(y),1),2), serverFolders, allLocalFiles, 'uni', 0);
    serverFolders = vertcat(serverFolders{:});
    
    allLocalFiles = cell2mat(allLocalFiles);
    allServerFilePaths = arrayfun(@(x,y) fullfile(y{1}, x.name), allLocalFiles, serverFolders, 'uni', 0);
    allLocalFilePaths = arrayfun(@(x) fullfile(x.folder, x.name), allLocalFiles, 'uni', 0);
    
    copyFiles2ServerAndDelete(allLocalFilePaths, allServerFilePaths)
end
cleanEmptyFoldersInDirectory(localFolder);
end
