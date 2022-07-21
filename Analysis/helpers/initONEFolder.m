function initONEFolder(ONEFolderName)
    % Will empty the folder if exists, or create one.
    
    if exist(eventsONEFolder,'dir')
        filesName = dir(ONEFolderName);
        for k = 1:length(filesName)
            fullFileName = fullfile(ONEFolderName, filesName(k).name);
            delete(fullFileName);
        end
    else
        mkdir(eventsONEFolder);
    end 
end