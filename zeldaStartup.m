%% Runs at startup of MATLAB on all pink-rig associated computers
%  Adds the repo to the path, and sets the vesion of psychtoolbox

%% Update PinkRig repo and update if needed
checkAndUpdatePinkRigRepo
addpath(genpath('C:\Users\Experiment\Documents\Github\PinkRigs'));

%% Change to old version of psychtoolbox
if strcmpi(getComputerType, 'stim')
    changeToOldPTB;
    fprintf('Changed to old version of PTB \n');

    addpath('\\znas.cortexlab.net\code\Rigging\ExpDefinitions\PinkRigs');
end