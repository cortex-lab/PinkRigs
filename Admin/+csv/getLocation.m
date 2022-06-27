function csvLocation = getLocation(subject)
    %%% This function will find the location of the CSV for a subject (or
    %%% the main CSV if input is 'main', or the 'kilosort_queue').
    
    if iscell(subject); subject = subject{1}; end
    csvPath = '\\zserver.cortexlab.net\Code\AVrig\';
    
    if strcmp(subject,'main')
        subject = '!MouseList';
    end
        
    if strcmp(subject,'kilosort_queue')
        subject = 'Helpers\kilosort_queue';
    end
    csvLocation = fullfile(csvPath, [subject '.csv']);