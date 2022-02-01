function commitChanges2PinkRigRepo(updateMessage)
checkAndUpdatePinkRigRepo;
if ~exist('updateMessage', 'var')
    updateMessage = '"No update message provieded"';
end
if ~strcmp(updateMessage(1), '"')
    updateMessage = ['"' updateMessage];
end
if ~strcmp(updateMessage(end), '"')
    updateMessage = [updateMessage '"'];
end
    
status = system('git add --all');
if status == 1
    warning('Could not add files for commit'); 
    return
end

status = system(['git commit -m' updateMessage]);
if status == 1
    warning('Could not commit files to GIT'); 
    return
end

status = system('git pull');
if status == 1
    warning('Could not commit pull current files from GIT'); 
    return
end

[status, ~] = system('git push');
if status == 1
    warning('Could not push current files from GIT'); 
    return
end
checkAndUpdatePinkRigRepo;
