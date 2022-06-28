function commitChanges2PinkRigRepo(updateMessage)
%% Function to updated PinkRig repo AND commit changes 
startFolder = cd;
checkAndUpdatePinkRigRepo;

if ~exist('updateMessage', 'var')
    updateMessage = '"No update message provided"';
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

[~, commitInfo] = system(['git commit -m' updateMessage]);
fprintf([commitInfo '\n']);

[status, ~] = system('git pull');
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
cd(startFolder);
