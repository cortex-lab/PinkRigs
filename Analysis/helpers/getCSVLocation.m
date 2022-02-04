function csvLocation = getCSVLocation(subject)
    %%% This function will find the location of the CSV for a subject (or
    %%% the main CSV if input is 'main', or the 'kilosort_queue').
    
    csvPath = '\\zserver.cortexlab.net\Code\AVrig\';
    
    if strcmp(subject,'main')
        subject = '!MouseList';
    end
        
    csvLocation = fullfile(csvPath, [subject '.csv']);