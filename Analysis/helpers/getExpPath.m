function expPath = getExpPath(subject, expDate, expNum)
    %% Outputs the exact path of an experiment.
    % This is intended to be used outside of the .csv, e.g., when checking 
    % if files have been pushed to the server, etc.
    %
    % Parameters:
    % -------------------
    % subject: str
    %   Name of the subject
    % expDate: str
    %   Date of the experiment
    % expNum: str
    %   Experiment number
    %
    % Returns: 
    % -------------------
    % expPath: str
    %   Path of the experiment folder
    
    % get current list of servers
    servers = getServersList;  
    
    % this allows to search for folder without expNum (useful for ephys)
    if ~exist('expNum', 'var')
        expNum = ''; 
    end
    
    % convert if input as double
    if isa(expNum,'double')
        expNum = num2str(expNum);
    end
    
    % loop through the servers to get the experiment's folder
    found = 0;
    n = 1;
    while found == 0 && n<=numel(servers)
        expPath = fullfile(servers{n},subject,expDate,expNum);
        if exist(expPath,'dir')
            found = 1;
        else
            n = n+1;
        end
    end
    
    if ~found
        expPath = [];
        warning('Couldn''t find experiment %s for mouse %s and date %s',expNum,subject,expDate')
    end
end