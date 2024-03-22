% check if new psychtoolbox is in path 
p_array = strsplit(path(), pathsep); % array of paths
old_ptb_folder = 'C:\Users\Experiment\Documents\MATLAB\Psychtoolbox'; 
new_ptb_folder = 'C:\toolbox\Psychtoolbox';
if any(strcmp(p_array, old_ptb_folder))
    % rmpath(genpath(new_ptb_folder))
    for pathIdx = 1:length(p_array)
        if contains(p_array{pathIdx}, old_ptb_folder)
            rmpath(p_array{pathIdx});
        end 
    end 
end 


addpath(genpath(new_ptb_folder));

% Screen('Preference', 'SkipSyncTests', 1); 

% TODO: put the MEX thing before the other thing
top_priority_path = [new_ptb_folder filesep 'PsychBasic\MatlabWindowsFilesR2007a'];
addpath(top_priority_path)

% the main folder needs to be top
addpath(new_ptb_folder)

PsychStartup; 
Screen('Preference', 'SkipSyncTests', 1); 