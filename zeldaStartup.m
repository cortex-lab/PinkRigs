[~, computerName] = system('hostname');

%% Change to old version of psychtoolbox
if contains(lower(computerName), 'stim')
    changeToOldPTB;
    fprintf('Changed to old version of PTB \n');
end

%% Check PinkRig repo and update if needed
cd(fileparts(which('zeldaStartup')));
[status, cmdout] = system('git status');
warnMessage = 'Cannot connect with GIT. PinkRig repo may be outdated!!!';
if status == 1
    warning(warnMessage);
end

if status == 0
    if contains(cmdout, 'Your branch is up to date')
        fprintf('PinkRig repo is up to date \n');
    else
        fprintf('PinkRig repo is outdated... will pull \n');
        pullStatus = system('git pull');
        if pullStatus == 1
            warning(warnMessage);
        end
    end
end

if status == 0
end

