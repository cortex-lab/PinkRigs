%% this funtion will need to be run at the end of each experiment/day? and 
function pipe2Server_tl()
    %% identify data
    ops.localsource = 'D:\LocalExpData'; % the localExpData folder where data is held
    % find all folders with a relevant file like timeline
    d = dir([ops.localsource '\**\*Timeline.mat']);  % checks what camera data is there
        
    %% push the data to server
    %day2check=40;
    %dToday=d([d(:).datenum]>=now-day2check & [d(:).datenum]<=now-day2check+1);
    ops.serversource = '\\128.40.224.65\Subjects';
    dToday=d([d(:).datenum]>=now-1);
    % check whether it has already been copied
    [moveDat]=assessCopy(dToday,ops);
    
    % copy the ones that haven't been 
    toCopy=moveDat([moveDat(:).copied]==false); 
 
    for datidx=1:numel(toCopy)
        mydat=toCopy(datidx).localfile;
        servertarget=toCopy(datidx).servertarget;
        copyfile(mydat,servertarget);
    end
    
    
    %% deletions 
    % delete files that have been copied correctly
    dBefore=d([d(:).datenum]<=now-2);
    [delDat]=assessCopy(dBefore,ops);
    
    % copy the ones that haven't been 
    toDelete=delDat([delDat(:).copied]==true); 
    for datidx=1:numel(toDelete)
        mydat=toDelete(datidx).localfile;
        checkfold=dir([mydat]);
        folders=regexp(checkfold.folder,'\','split');
        mname=folders{numel(folders)-2}; 
        
        mfold=[ops.localsource '\' mname];
        
        
        delete(mydat);
        
        checkmouse=dir([mfold '\**\*.*']); 
        if sum([checkmouse(:).bytes])<10
            rmdir(mfold,'s');
        end
        % if folder is empty delete parent?
        
        
    end
        
end

function [moveDat]=assessCopy(d,ops)
% for each camera data check all the relevant other datafiles that are
% there
% select the relevant target foler
% check whether copying has occured

ct=0; 
for tlidx=1:numel(d)
    folders=regexp(d(tlidx).folder,'\','split');
    mname=folders{numel(folders)-2}; 
    date=folders{numel(folders)-1}; 
    expnum=str2double(folders{numel(folders)}); 

    % target folder 
    servertarget=[ops.serversource sprintf('\\%s\\%s\\%.d',mname,date,expnum)]; 

    % check any files that contain info relating to that camera 
    subd=dir([d(tlidx).folder]); 
    subd=subd([subd(:).isdir]==0);
        for fileidx=1:numel(subd)
            ct=ct+1; 
            % data source local
            mydat=[subd(fileidx).folder '\' subd(fileidx).name];
            % need to get expnum out to be able to identify relevant target folder

            % check if file already exists at the target
            targetfile=[servertarget '\' subd(fileidx).name];
            if isfile(targetfile)
                % check if file sizes are the same 
                dOnServer=dir(targetfile);
                dOnLocal=dir(mydat);
                samesize=dOnLocal.bytes==dOnServer.bytes;
            end

            if samesize==1
                copiedOK=true;
            else
                copiedOK=false; 
            end
            
            % write the movedat struct
            moveDat(ct).localfile=mydat;
            moveDat(ct).servertarget=servertarget;
            moveDat(ct).copied=copiedOK;
        end     


end
if exist('moveDat')==0
    moveDat.localfile=nan;
    moveDat.servertarget=nan;
    moveDat.copied=nan;
end
end


