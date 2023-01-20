function log = copyEphysData2ServerAndDelete(localFolder,fid)
%% Copies the ephys folder to the server and then deletes it locally
% 
% NOTE: At time of writing, this function first checkes that the local
% folder contains both the "sync.mat" and the and the copressed apbin files
% before copying any of the folder contents to the server. The purpose of
% this function is to check this, create a list of what should be copied,
% and pass that on to "copyFiles2ServerAndDelete"
%
% Parameters:
% ------------
% localFolder (default='D:\ephysData'): string
%   the folder where ephys data should be copied from
%
% fid (default = []): string
%   This is (I think) the ID of the current log... written by Celian?
%
% Returns: 
% -----------
% log: string
%   A log of the various timings and other useful information during run


if ~exist('localFolder', 'var'); localFolder = 'D:\ephysData'; end
    if ~exist('fid', 'var'); fid = []; end
    
    log = '';  % Save log in string in case needs to output
    
    log = appendAndPrint(log, sprintf('Starting now %s... \n',datestr(now)), fid);
    localEphysFolders = dir([localFolder '\**\*']);
    localEphysFolders = localEphysFolders([localEphysFolders.isdir]' & ...
        contains({localEphysFolders.name}', 'imec') & ...
        ~contains({localEphysFolders.name}', '.'));
    
    localEphysPaths = arrayfun(@(x) [x.folder filesep x.name], localEphysFolders, 'uni', 0);
    localEphysPathContents = cellfun(@dir, localEphysPaths, 'uni', 0);
    localEphysFolders(cellfun(@(x) isempty(x) | all([x.isdir]), localEphysPathContents)) = [];
    
    subjectFromFolder = arrayfun(@(x) x.name(1:5), localEphysFolders, 'uni', 0);
    dateFromFolder = arrayfun(@(x) cell2mat(regexp(x.name, '\d\d\d\d-\d\d-\d\d', 'match')), localEphysFolders, 'uni', 0);
    
    % NOTE: This is specific to SpikeGLX output... maybe there is a better way
    splitFolders = arrayfun(@(x) regexp([x.folder filesep x.name],'\','split'), localEphysFolders, 'uni', 0);
    serverFolders = cellfun(@(x,y) getExpPath(x,y), subjectFromFolder, dateFromFolder, 'uni', 0);
    serverFolders = cellfun(@(x,y) fullfile(x, 'ephys', y(end-1), y(end)), serverFolders, splitFolders);
    if isempty(serverFolders); serverFolders = num2cell(serverFolders); end % otherwise crashes?
    localFolders = arrayfun(@(x) [x.folder filesep x.name], localEphysFolders, 'uni', 0);
    
    % Check that sync and compressed files exist either on server or locally
    localCompressed = cell2mat(cellfun(@(x) ~isempty(dir([x '\*.ap.cbin'])), localFolders, 'uni', 0));
    serverCompressed = cell2mat(cellfun(@(x) ~isempty(dir([x '\*.ap.cbin'])), serverFolders, 'uni', 0));
    localCh = cell2mat(cellfun(@(x) ~isempty(dir([x '\*.ap.ch'])), localFolders, 'uni', 0));
    serverCh = cell2mat(cellfun(@(x) ~isempty(dir([x '\*.ap.ch'])), serverFolders, 'uni', 0));
    localSync = cell2mat(cellfun(@(x) ~isempty(dir([x '\*sync.mat'])), localFolders, 'uni', 0));
    serverSync = cell2mat(cellfun(@(x) ~isempty(dir([x '\*sync.mat'])), serverFolders, 'uni', 0));
    readyFolders = (localCompressed | serverCompressed) & ...
        (localCh | serverCh) & ...
        (localSync | serverSync);
    
    if isempty(readyFolders)
        log = appendAndPrint(log, sprintf('There are no ephys files in the local directory. Returning... \n'), fid);
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
    
    % Do not copy bin files over to the server
    isBinFile = cell2mat(arrayfun(@(x) contains(x.name,'.ap.bin'),allLocalFiles,'uni',0));
    allLocalFiles(isBinFile) = [];
    serverFolders(isBinFile) = [];

    % Copy all the peripherals (.ch, .meta, sync) first, then .cbin
    isCbinFile = cell2mat(arrayfun(@(x) contains(x.name,'.ap.cbin'),allLocalFiles,'uni',0));
    allLocalFiles = cat(1,allLocalFiles(~isCbinFile), allLocalFiles(isCbinFile));
    serverFolders = cat(1,serverFolders(~isCbinFile), serverFolders(isCbinFile));
    
    allLocalFilePaths = arrayfun(@(x) fullfile(x.folder, x.name), allLocalFiles, 'uni', 0);
    allServerFilePaths = arrayfun(@(x,y) fullfile(y{1}, x.name), allLocalFiles, serverFolders, 'uni', 0);
    
    %Sanity check to make sure that the files are in the correct order
    allLocalFilePathsTest = cellfun(@(x) x(max(strfind(x, '\'))+1:end), allLocalFilePaths, 'uni', 0);
    allServerFilePathsTest = cellfun(@(x) x(max(strfind(x, '\'))+1:end), allServerFilePaths, 'uni', 0);
    matchTest = all(cellfun(@(x,y) strcmp(x,y), allLocalFilePathsTest, allServerFilePathsTest));
    if ~matchTest
        error('File paths names do not correspond..?')
    end
    
    log_copy = copyFiles2ServerAndDelete(allLocalFilePaths, allServerFilePaths, 1, fid);
    log = append(log, log_copy);
    
    %%
    cleanEmptyFoldersInDirectory(localFolder);
    
    log = appendAndPrint(log, sprintf('Stopping now %s.',datestr(now)), fid);

end