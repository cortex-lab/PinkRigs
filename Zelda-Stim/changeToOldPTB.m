% check if new psychtoolbox is in path 
p_array = strsplit(path(), pathsep); % array of paths
new_ptb_folder = 'C:\toolbox\Psychtoolbox';
if any(strcmp(p_array, new_ptb_folder))
    % rmpath(genpath(new_ptb_folder))
    for pathIdx = 1:length(p_array)
        if contains(p_array{pathIdx}, new_ptb_folder)
            rmpath(p_array{pathIdx});
        end 
    end 
end 

old_ptb_folder = 'C:\Users\Experiment\Documents\MATLAB\Psychtoolbox'; 
addpath(genpath(old_ptb_folder));

% TODO: put the MEX thing before the other thing
top_priority_path = [old_ptb_folder filesep 'PsychBasic\MatlabWindowsFilesR2007a'];
addpath(top_priority_path)

% the main folder needs to be top
addpath(old_ptb_folder)

PsychStartup; 