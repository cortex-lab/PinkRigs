function checkAndUpdatePinkRigRepo

cd(fileparts(which('zeldaStartup')));
[status, cmdout] = system('git status');
warnMessage1 = 'Cannot connect with GIT. PinkRig repo may be outdated!!!';
warnMessage2 = 'Cannot pull from GIT... PinkRig repo may be outdated!!!';
warnMessage3 = 'It looks like you have uncommited changes in the PinkRig repo... why?!';

if status == 1
    warning(warnMessage1);
end
if status == 0
    if contains(cmdout, 'Your branch is up to date')
        fprintf('PinkRig repo is up to date \n');
    else
        fprintf('PinkRig repo is outdated... will pull \n');
        pullStatus = system('git pull');
        if pullStatus == 1
            warning(warnMessage2);
        end
    end
end

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
