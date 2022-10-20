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
    
serverFileDetails = cellfun(@dir, serverFilePaths, 'uni', 0);
localFileDetails = cell2mat(cellfun(@dir, localFilePaths, 'uni', 0));

failedCopy = cellfun(@isempty, serverFileDetails);
vid2Check = contains(cellfun(@(x) x.name, serverFileDetails, 'uni',0), '.mj2') & ~failedCopy;
for i = find(vid2Check)'
    corruptLocal = 0;
    corruptServer = 0;
    localVid = fullfile(localFileDetails(i).folder, localFileDetails(i).name);
    serverVid = fullfile(serverFileDetails{i}.folder, serverFileDetails{i}.name);

    try VideoReader(localVid); catch, corruptLocal = 1; end %#ok<*TNMLP>
    try VideoReader(serverVid); catch, corruptServer = 1; end

    if corruptServer == corruptLocal && corruptLocal == 1
        fprintf('%s corrupted locally \n',serverFileDetails{i}.name);
    elseif corruptServer ~= corruptLocal && corruptLocal == 0
        fprintf('%s corrupted when copying. Deleting and will retry next time \n', serverFileDetails{i}.name);
        delete(serverVid);
        failedCopy(i) = 1;
    elseif corruptServer ~= corruptLocal && corruptLocal == 1
        fprintf('%s is corrupted locally but not on server?!?!?! \n', serverFileDetails{i}.name);
        failedCopy(i) = 1;
    elseif corruptServer == corruptLocal && corruptLocal == 0
%         fprintf('%s safely copied. No corruption :) \n', serverFileDetails{i}.name);
    end
end

serverFileDetails = cell2mat(serverFileDetails(~failedCopy));
localFileDetails = localFileDetails(~failedCopy); 

%% Deletions
oldIdx = ([localFileDetails(:).datenum]<=now-0)';
sizeMismatch = ([localFileDetails(:).bytes]~=[serverFileDetails(:).bytes])';

% delete local files that have been copied correctly
local2Delete = localFileDetails(oldIdx & ~sizeMismatch);
fprintf('Deleting local files... \n')
arrayfun(@(x) delete(fullfile(x.folder, x.name)), local2Delete);

% delete server files that have been copied correctly
if any(sizeMismatch)
    server2Delete = serverFileDetails(oldIdx & sizeMismatch);
    fprintf('Deleting "bad" server files... \n')
    arrayfun(@(x) delete(fullfile(x.folder, x.name)), server2Delete);
end

fprintf('Done! \n')
end
