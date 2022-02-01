[~, computerName] = system('hostname');

%% Change to old version of psychtoolbox
if contains(lower(computerName), 'stim')
    changeToOldPTB;
    fprintf('Changed to old version of PTB \n');
end

%% Check PinkRig repo and update if needed
checkAndUpdatePinkRigRepo
