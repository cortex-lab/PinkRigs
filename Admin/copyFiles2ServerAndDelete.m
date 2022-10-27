function copyFiles2ServerAndDelete(localFilePaths, serverFilePaths, makeMissingDirs)
if ~exist('makeMissingDirs', 'var'); makeMissingDirs = 0; end

serverList = getServersList;
serverList = cellfun(@(x) x(1:10), serverList, 'uni', 0);

if any(cellfun(@(x) contains(x, serverList), localFilePaths))
    error('It seems like the localFolder is actually on the zerver?!?!')
end

isDirectory = cellfun(@isfolder, localFilePaths);
localFilePaths = localFilePaths(~isDirectory);
serverFilePaths = serverFilePaths(~isDirectory);
copiedAlready = cellfun(@(x) exist(x,'file'), serverFilePaths)>0;

if any(contains(serverFilePaths, 'ephys'))    
    slashIdx = cellfun(@(x) strfind(x, filesep), serverFilePaths, 'uni', 0);
    serverFilePathsCelian = cellfun(@(x,y) [x(1:y(end-2)-1) x(y(end-1):end)], serverFilePaths, slashIdx, 'uni', 0);
    copiedAlreadyCelian = cellfun(@(x) exist(x,'file'), serverFilePathsCelian)>0;
    serverFilePaths(copiedAlreadyCelian) = serverFilePathsCelian(copiedAlreadyCelian);
    copiedAlready = copiedAlready | copiedAlreadyCelian;
end

if all(copiedAlready)
    fprintf('All data is already copied .. \n')
else
    files2copy = find(~copiedAlready);
    for i = 1:length(files2copy)
        cIdx = files2copy(i);
        fprintf('Copying %s ...\n', localFilePaths{cIdx});
        tic;
        if ~isfolder(fileparts(serverFilePaths{cIdx}))
            if makeMissingDirs
                mkdir(fileparts(serverFilePaths{cIdx}));
            else
                fprintf('WARNING: Directory missing for: %s. Skipping.... \n', localFilePaths{cIdx});
            end
        end
        try
            copyfile(localFilePaths{cIdx},fileparts(serverFilePaths{cIdx}));
        catch
            fprintf('WARNING: Problem copying file %s. Skipping.... \n', localFilePaths{cIdx});
        end
        elapsedTime = toc;
        d = dir(localFilePaths{cIdx});
        rate = d.bytes/(10^6)/elapsedTime;
        fprintf('Done in %d sec (%d MB/s).\n',elapsedTime,rate)
    end
end

localFileMD5 = cellfun(@(x) GetMD5(x, 'File'), localFilePaths, 'ErrorHandler', @md5Error, 'uni', 0);
serverFileMD5 = cellfun(@(x) GetMD5(x, 'File'), serverFilePaths, 'ErrorHandler', @md5Error, 'uni', 0);
failedCopy = cellfun(@(x,y) ~strcmp(x,y), localFileMD5, serverFileMD5);

%% Deletions
% delete local files that have been copied correctly
if any(~failedCopy)
    fprintf('Deleting local files... \n')
    cellfun(@(x) delete(x), localFilePaths(~failedCopy));
end

% delete server files that have been copied correctly
if any(failedCopy)
    fprintf('Deleting "bad" server files... \n')
    cellfun(@(x) delete(x), serverFilePaths(failedCopy));
end

fprintf('Done! \n')
end


function outPut = md5Error(~,~,~)
%function to handle errors when getting the md5 hash
outPut = 'md5Error';
end