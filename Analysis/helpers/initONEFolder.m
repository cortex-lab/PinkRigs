function initONEFolder(ONEFolderName)
    % Will empty the folder if exists, or create one.
    if exist(ONEFolderName,'dir')
        filesName = dir(ONEFolderName);
        filesName = filesName(cellfun(@(x) ~strcmp(x(1),'.'),{filesName.name}'));
        for k = 1:length(filesName)
            fullFileName = fullfile(ONEFolderName, filesName(k).name);
            delete(fullFileName);
        end
    else
        mkdir(ONEFolderName);
    end 
end