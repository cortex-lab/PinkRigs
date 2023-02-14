function log = copyFiles2ServerAndDelete(localFilePaths, serverFilePaths, makeMissingDirs, fid)
%% Copies a list of files to the server and then deletes them locally
%
% Parameters:
% ------------
% localFilePaths (required): cell array of strings
%   A list of all files (full paths) that should be copied
%
% serverFilePaths (required): cell array of strings
%   A list of all destinations (full paths) that should be copied to
%
% makeMissingDirs (default=0): logical
%   A logical to indicate whether file directories should be created if
%   they don't already exist.
%
% fid (default = []): string
%   This is (I think) the ID of the current log... written by Celian?
%
% Returns:
% -----------
% log: string
%   A log of the various timings and other useful information during run


if ~exist('makeMissingDirs', 'var'); makeMissingDirs = 0; end
if ~exist('fid', 'var'); fid = []; end
serverList = getServersList;
serverList = cellfun(@(x) x(1:10), serverList, 'uni', 0);

log = ''; % Save log in case in string in case needs to output

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
localFilePaths(contains(localFilePaths, 'ap.bin')) = [];
serverFilePaths(contains(serverFilePaths, 'ap.bin')) = [];
copiedAlready = cellfun(@(x) exist(x,'file'), serverFilePaths)>0;

%% Loop to copy/check/delete files
failedCopy = 0*copiedAlready>0;
for i = 1:length(copiedAlready)
    log = appendAndPrint(log, sprintf('Processing %s (%s)...\n', localFilePaths{i}, datestr(now)), fid);

    %         tic;
    %         log = appendAndPrint(log, sprintf('Running GetMD5 on local file (%s).\n',datestr(now)), fid);
    %         localFileMD5 = GetMD5(localFilePaths{i}, 'File');
    %         elapsedTime = toc;
    %         log = appendAndPrint(log, sprintf('Done running GetMD5 on local file in %d sec.\n',elapsedTime), fid);

    %This exception deals with the fact that we expect timeline to be
    %different, so we only "copy" if we can't open the server version
    if contains(localFilePaths{i}, 'timeline', 'ignorecase', 1) && exist(serverFilePaths{i}, 'file')
        try load(serverFilePaths{i})
            serverFilePaths{i} = localFilePaths{i};
        catch
            log = appendAndPrint(log, sprintf('Server timeline appears corrupt for %s ...\n', localFilePaths{i}), fid);
        end
    end

    if ~copiedAlready(i)
        log = appendAndPrint(log, sprintf('Copying %s (%s)...\n', localFilePaths{i}, datestr(now)), fid);
        if ~isfolder(fileparts(serverFilePaths{i}))
            if makeMissingDirs
                mkdir(fileparts(serverFilePaths{i}));
            else
                log = appendAndPrint(log, sprintf('WARNING: Directory missing for: %s. Skipping.... \n', localFilePaths{i}), fid);
            end
        end
        try
            tic;
            copyfile(localFilePaths{i},fileparts(serverFilePaths{i}));
            elapsedTime = toc;
            d = dir(localFilePaths{i});
            rate = d.bytes/(10^6)/elapsedTime;
            log = appendAndPrint(log, sprintf('Ran copying in %d sec (%d MB/s).\n',elapsedTime,rate), fid);
            tic;
            %                 log = appendAndPrint(log, sprintf('Running GetMD5 on server file (%s).\n',datestr(now)), fid);
            %                 serverFileMD5 = GetMD5(serverFilePaths{i}, 'File');
            %                 elapsedTime = toc;
            %                 log = appendAndPrint(log, sprintf('Done running GetMD5 on server file in %d sec.\n',elapsedTime), fid);
            %                 if ~strcmp(localFileMD5, serverFileMD5)
            %                     log = appendAndPrint(log, sprintf('WARNING: MD5 checksum doesn''t match for file %s. Skipping.... \n', localFilePaths{i}), fid);
            %                     failedCopy(i) = 1;
            %                 end
        catch
            log = appendAndPrint(log, sprintf('WARNING: Problem copying file %s. Skipping.... \n', localFilePaths{i}), fid);
            failedCopy(i) = 1;
        end
        if failedCopy(i) == 0
            log = appendAndPrint(log, sprintf('Copy successful. Will delete local file... \n'), fid);
        end
    else
        dserver = dir(serverFilePaths{i});
        if (now - dserver.datenum)*24 > 1
            %                 serverFileMD5 = GetMD5(serverFilePaths{i}, 'File');
            %                 failedCopy(i) = ~strcmp(localFileMD5, serverFileMD5);
            failedCopy(i) = 0; % force it for now
            if failedCopy(i) == 0
                log = appendAndPrint(log, sprintf('Copied already. Will delete local file... \n'), fid);
            end
        else
            log = appendAndPrint(log, sprintf('Must be copying. Skip... \n'), fid);
        end
    end
    if failedCopy(i) == 0
        delete(localFilePaths{i});
    elseif exist(serverFilePaths{i}, 'file')
        movefile(serverFilePaths{i}, [serverFilePaths{i} '_FAILEDCOPY']);
    end
end
%% TODO--email list of bad copies to users

log = appendAndPrint(log, sprintf('Done (%s)! \n', datestr(now)), fid);
end