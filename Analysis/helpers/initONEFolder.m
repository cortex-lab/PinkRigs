function initONEFolder(ONEFolderName,fileStr)
    %% Initializes the ONE folder.
    % Will create it if doesn't exist, or remove the files that are in it
    % (and which can be selected with a specific string 'fileStr')
    %
    % Parameters:
    % -------------------
    % ONEFolderName: str
    %   Path where to check the ONE folder.
    % fileStr: str
    %   Specific string to select the files to delete.

    if ~exist('fileStr','var')
        fileStr = '';
    end

    % Will empty the folder if exists, or create one.
    if exist(ONEFolderName,'dir')
        filesName = dir(ONEFolderName);
        filesName = filesName(cellfun(@(x) ~strcmp(x(1),'.'),{filesName.name}'));
        filesName = filesName(contains({filesName.name},fileStr));
        for k = 1:length(filesName)
            fullFileName = fullfile(ONEFolderName, filesName(k).name);
            delete(fullFileName);
        end
    else
        mkdir(ONEFolderName);
    end 
end