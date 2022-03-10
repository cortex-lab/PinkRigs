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
        fprintf('Copying %s ... \n', localFilePaths{cIdx});
        
        if ~isfolder(fileparts(serverFilePaths{cIdx}))
            if makeMissingDirs
                mkdir(fileparts(serverFilePaths{cIdx}));
            else
                fprintf('WARNING: Directory missing for: %s. Skipping.... \n', data2Copy);
            end
        end
        try
            copyfile(localFilePaths{cIdx},fileparts(serverFilePaths{cIdx}));
        catch
            fprintf('WARNING: Problem copying file %s. Skipping.... \n', data2Copy);
        end
    end
end

serverFileDetails = cellfun(@dir, serverFilePaths, 'uni', 0);
localFileDetails = cell2mat(cellfun(@dir, localFilePaths, 'uni', 0));

failedCopy = cellfun(@isempty, serverFileDetails);
localFileDetails(failedCopy) = []; 
serverFileDetails = cell2mat(serverFileDetails);

%% Deletions
% delete files that have been copied correctly
oldIdx = ([localFileDetails(:).datenum]<=now-1)';
sizeMismatch = ([localFileDetails(:).bytes]~=[serverFileDetails(:).bytes])';

toDelete = localFileDetails(oldIdx & ~sizeMismatch);
arrayfun(@(x) delete(fullfile(x.folder, x.name)), toDelete);
end
