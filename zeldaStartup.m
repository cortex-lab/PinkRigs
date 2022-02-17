%% Change to old version of psychtoolbox
if strcmpi(getComputerType, 'stim')
    changeToOldPTB;
    fprintf('Changed to old version of PTB \n');
end

%% Check PinkRig repo and update if needed
checkAndUpdatePinkRigRepo
