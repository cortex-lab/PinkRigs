function computerType = getComputerType
%% Automatically detect the type of computer

[~, computerName] = system('hostname');
if contains(lower(computerName), 'ephys')
    computerType = 'ephys';
elseif contains(lower(computerName), 'stim')
    computerType = 'stim';
elseif contains(lower(computerName), 'time')
    computerType = 'time';
elseif contains(lower(computerName), 'mc')
    computerType = 'mc';
elseif contains(lower(computerName), 'kilo1')
    computerType = 'kilo1';
elseif contains(lower(computerName), 'kilo2')
    computerType = 'kilo2';
elseif contains(lower(computerName), 'zippy')
    computerType = 'pips';
elseif contains(lower(computerName), 'zestylemon')
    computerType = 'celians';    
else
    warning('Computer type not recognized?!')
    computerType = 'Unrecognized';
end
