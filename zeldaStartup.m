%% Check PinkRig repo and update if needed
checkAndUpdatePinkRigRepo
addpath(genpath('C:\Users\Experiment\Documents\Github\PinkRigs'));

%% Change to old version of psychtoolbox
if strcmpi(getComputerType, 'stim')
    changeToOldPTB;
    fprintf('Changed to old version of PTB \n');

    addpath('\\zserver.cortexlab.net\code\Rigging\ExpDefinitions\PinkRigs');
end