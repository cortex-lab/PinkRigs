%% this funtion will need to be run at the end of each experiment/day? and 
function pipe2Server_tl()
    %% identify data
    localFolder ='D:\LocalExpData'; % the localExpData folder where data is held
    % find all folders with a relevant file like timeline
    localDat = [dir([localFolder '\**\*Timeline.mat']);...
        dir([localFolder '\**\*block.mat']); ...
        dir([localFolder '\**\*eyeCam*'])];  % checks what camera data is there
        
    %% push the data to server
    % check whether it has already been copied
    [folders, idxRef] = unique({localDat.folder});
    localDat = localDat(idxRef);
    splitFolders = cellfun(@(x) regexp(x,'\','split'), folders, 'uni', 0);

    subjects = cellfun(@(x) x{end-2}, splitFolders, 'uni', 0)';
    expDates = cellfun(@(x) x{end-1}, splitFolders, 'uni', 0)';
    expNums = cellfun(@(x) x{end-0}, splitFolders, 'uni', 0)';
    serverPath = cellfun(@(x,y,z) getExpPath(x,y,z), subjects, expDates, expNums, 'uni', 0);
    
    moveDat = assessCopy(localDat,serverPath);
    
    % copy the ones that haven't been 
    toCopy = moveDat([moveDat(:).copied]==false); 
    
    for datidx = 1:numel(toCopy)
        disp(datidx)
        data2Copy = toCopy(datidx).localFile;
        serverTarget = toCopy(datidx).serverTarget;
        try
            copyfile(data2Copy,serverTarget);
        catch
            fprintf('WARNING: Problem copying file %s. Skipping.... \n', data2Copy);
        end
    end
    
    
    %% deletions
    % delete files that have been copied correctly
    oldIdx = [localDat(:).datenum]<=now-2;
    oldDataStatus = assessCopy(localDat(oldIdx),serverPath(oldIdx));
    
    % copy the ones that haven't been 
    toDelete = oldDataStatus([oldDataStatus(:).copied]==true); 
    for i = 1:numel(toDelete)
        localFile = toDelete(i).localFile;
        checkFolder = dir(localFile);
        folders = regexp(checkFolder.folder,'\','split');
        subject = folders{numel(folders)-2}; 
        
        mfold = [localFolder '\' subject];
        
        
        delete(localFile);
        
        checkmouse=dir([mfold '\**\*.*']); 
        if sum([checkmouse(:).bytes])<10
            rmdir(mfold,'s');
        end    
    end
end

function moveDat = assessCopy(localDat, serverPath)
% for each camera data check all the relevant other datafiles that are
% there
% select the relevant target foler
% check whether copying has occured

ct=0; 
for tlidx = 1:numel(localDat)
    % check any files that contain info relating to that camera 
    subd = dir([localDat(tlidx).folder]); 
    subd = subd([subd(:).isdir]==0);
        for fileidx=1:numel(subd)
            ct=ct+1; 
            % data source local
            localFile = [subd(fileidx).folder '\' subd(fileidx).name];
            % need to get expnum out to be able to identify relevant target folder

            % check if file already exists at the target
            serverFile = [serverPath{tlidx} '\' subd(fileidx).name];
            if isfile(serverFile)
                % check if file sizes are the same 
                dOnServer = dir(serverFile);
                dOnLocal = dir(localFile);
                samesize = dOnLocal.bytes==dOnServer.bytes;
            else
                samesize = 0;
            end

            if samesize==1
                copiedOK=true;
            else
                copiedOK=false; 
            end
            
            % write the movedat struct
            moveDat(ct).localFile=localFile;
            moveDat(ct).serverTarget=serverPath{tlidx};
            moveDat(ct).copied=copiedOK;
        end     


end
if ~exist('moveDat', 'var')
    moveDat.localFile=nan;
    moveDat.serverTarget=nan;
    moveDat.copied=nan;
end
end


