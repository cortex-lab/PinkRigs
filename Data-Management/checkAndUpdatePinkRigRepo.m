function checkAndUpdatePinkRigRepo
%% Function to check and update the local copy of the PinkRig repo 

% Go to the pink rig repo folder and get git status
cd(fileparts(which('zeldaStartup')));
[status, cmdout] = system('git status');

% Some warning messages to use
warnMessage1 = 'Cannot connect with GIT. PinkRig repo may be outdated!!!';
warnMessage2 = 'Cannot pull from GIT... PinkRig repo may be outdated!!!';
warnMessage3 = 'It looks like you have uncommited changes in the PinkRig repo... why?!';

% Check status of repository and pull if not up to date
if status == 1
    warning(warnMessage1);
end
if status == 0
    if contains(cmdout, regexprep('Your branch is up to date','-',' '))
        fprintf('PinkRig repo is up to date \n');
    else
        fprintf('PinkRig repo is outdated... will pull \n');
        pullStatus = system('git pull');
        if pullStatus == 1
            warning(warnMessage2);
        end
    end
end

% Check status of repository alert used to uncomitted changes
[status, cmdout] = system('git status');
if status == 1
    warning(warnMessage1);
end
if status == 0
    if contains(cmdout, 'working tree clean')
        fprintf('Work tree for Pink Rig repo is clean \n');
    else
        warning(warnMessage3);
    end
end
end
