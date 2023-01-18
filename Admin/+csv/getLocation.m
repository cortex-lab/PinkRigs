function csvLocation = getLocation(subject)
    %% Find the location of the CSV for a subject. 
    % Note: can also be used to find the main CSV if input is 'main'
    %
    % Parameters:
    % -------------------
    % subject: str
    %   Name of the subject to look for, or 'main'.
    %
    % Returns: 
    % -------------------
    % csvLocation: str
    %   Path to the csv.
    
    if iscell(subject); subject = subject{1}; end
    csvPath = '\\zinu.cortexlab.net\Subjects\PinkRigs\';
    
    if strcmp(subject,'main')
        subject = '!MouseList';
    end

    csvLocation = fullfile(csvPath, [subject '.csv']);