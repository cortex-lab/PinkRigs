function changePinkRigsFunction(oldName, newName, fileExt)

if ~exist('fileExt', 'var'); fileExt = '.m'; end

%% Parameters
% The directory in which to replace files. Currently this code does not modify files in
% sub-directories
pinkRigsDir = fileparts(which('zeldaStartup'));
% The string that will be replaced
oldString = sprintf(oldName);
% The replacement string
newString = sprintf(newName);

%% Determine files to update, and update them as necessary
% Put the details of all files and folders in that current directory into a structure
fileList = dir([pinkRigsDir '\**\*' fileExt]);
% Initialise indexes for files that do and do not contain oldString
changedIdx = zeros(length(fileList),1)>0;

filePaths = arrayfun(@(x) fullfile(x.folder, x.name), fileList, 'uni', 0);

% For the number of files and folders in the directory
for idx = 1 : length(fileList)
    
    % Open the file for reading
    fileIdRead  = fopen(filePaths{idx}, 'r');
    
    % Extract the text
    fileText = fscanf(fileIdRead,'%c');
    
    % Close the file
    fclose(fileIdRead);
    
    splitFile = splitlines(fileText);
    
    if any(contains(splitFile, 'function'))
        splitFile = splitFile(~contains(splitFile, 'function'));
    end
        
    % If an occurrence is found...
    if any(contains(splitFile, oldString))
        
        % Replace any occurrences of oldString with newString
        fileTextNew = strrep(fileText, oldString, newString);
        
        % Open the file for writing
        fileIdWrite = fopen(filePaths{idx}, 'w');
        
        % Write the modified text
        fprintf(fileIdWrite, '%c', fileTextNew);
        
        % Close the file
        fclose(fileIdWrite);
        
        % Update the index for files that contained oldString
        changedIdx(idx) = 1;
    end
end
%% Display what files were changed, and what were not
% If the variable filesWithString exists in the workspace
if any(changedIdx)
    disp('Files that contained the target string that were updated:');
    % Display their names
    cellfun(@(x) fprintf('%s \n', x), filePaths(changedIdx), 'uni', 0);
else
    disp('No files contained the target string');
end