function copyFiles2ServerAndDelete(localFilePaths, serverFilePaths, makeMissingDirs)
if ~exist('makeMissingDirs', 'var'); makeMissingDirs = 0; end
serverList = getServersList;
serverList = cellfun(@(x) x(1:10), serverList, 'uni', 0);

% Priotitize timeline to copy
timelineIdx = contains(localFilePaths, 'timeline', 'ignorecase', 1);
localFilePaths = [localFilePaths(timelineIdx); localFilePaths(~timelineIdx)];
serverFilePaths = [serverFilePaths(timelineIdx); serverFilePaths(~timelineIdx)];

if any(cellfun(@(x) contains(x, serverList), localFilePaths))
    error('It seems like the localFolder is actually on the zerver?!?!')
end

isDirectory = cellfun(@isfolder, localFilePaths);
localFilePaths = localFilePaths(~isDirectory);
serverFilePaths = serverFilePaths(~isDirectory);
localFilePaths(contains(localFilePaths, '.bin')) = [];
serverFilePaths(contains(serverFilePaths, '.bin')) = [];
copiedAlready = cellfun(@(x) exist(x,'file'), serverFilePaths)>0;

%% Loop to copy/check/delete files
failedCopy = 0*copiedAlready>0;
for i = 1:length(copiedAlready)
    localFileMD5 = GetMD5(localFilePaths{i}, 'File');
    fprintf('Processing %s ...\n', localFilePaths{i});
    
    %This exception deals with the fact that we expect timeline to be
    %different, so we only "copy" if we can't open the server version
    if contains(localFilePaths{i}, 'timeline', 'ignorecase', 1) && exist(serverFilePaths{i}, 'file')
        try load(serverFilePaths{i})
            serverFilePaths{i} = localFilePaths{i};
        catch
            fprintf('Server timeline appears corrupt for %s ...\n', localFilePaths{i});
        end
    end
    
    if ~copiedAlready(i)
        fprintf('Copying %s ...\n', localFilePaths{i});
        tic;
        if ~isfolder(fileparts(serverFilePaths{i}))
            if makeMissingDirs
                mkdir(fileparts(serverFilePaths{i}));
            else
                fprintf('WARNING: Directory missing for: %s. Skipping.... \n', localFilePaths{i});
            end
        end
        try
            copyfile(localFilePaths{i},fileparts(serverFilePaths{i}));
            serverFileMD5 = GetMD5(serverFilePaths{i}, 'File');
            if ~strcmp(localFileMD5, serverFileMD5)
                fprintf('WARNING: Problem copying file %s. Skipping.... \n', localFilePaths{i});
                failedCopy(i) = 1;
            else
                elapsedTime = toc;
                d = dir(localFilePaths{i});
                rate = d.bytes/(10^6)/elapsedTime;
                fprintf('Done in %d sec (%d MB/s).\n',elapsedTime,rate)               
            end
        catch
            fprintf('WARNING: Problem copying file %s. Skipping.... \n', localFilePaths{i});
            failedCopy(i) = 1;
        end
        if failedCopy(i) == 0
            fprintf('Copy successful. Will delete local file... \n')
        end
    else
        serverFileMD5 = GetMD5(serverFilePaths{i}, 'File');
        failedCopy(i) = ~strcmp(localFileMD5, serverFileMD5);
        if failedCopy(i) == 0
            fprintf('Copied already. Will delete local file... \n')
        end
    end
    if failedCopy(i) == 0
        delete(localFilePaths{i});
    elseif exist(serverFilePaths{i}, 'file')
        movefile(serverFilePaths{i}, [serverFilePaths{i} '_FAILEDCOPY']);
    end
end
%% TODO--email list of bad copies to users

fprintf('Done! \n')
end
