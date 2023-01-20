function commitChanges2PinkRigRepo(updateMessage)
%% Update the PinkRig repo AND commit changes 
% 
% NOTE: This function calls "checkAndUpdatePinkRigRepo" to pull the latest
% version of the PinkRigs repo, but then ADDITIONALLY commits any changes
% to the PinkRigs repo from the current local version. If there are
% conflicts, these will have to be manually resolved.
%
%
% Parameters:
% ------------
% updateMessage (default='No update message provided')
%   The message that will be committed with the update to the PinkRig repo

startFolder = cd;
cd(fileparts(which('zeldaStartup')));
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
