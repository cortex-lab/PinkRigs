function checkForNewPinkRigRecordings(varargin)
%% Function to check for any new recordings on the pink rigs and update csvs
% NOTE: This function uses csv.inputValidate to parse inputs

% Add default values for extra inputs:
% expDate: integer--number of days into the past to check for new data
% recompute: logical--whether csv should be deleted and remade anew
% NOTE "expDate" can be 'all'--faster than a bit integer
varargin = ['expDate', {0}, varargin];
varargin = ['recompute', {0}, varargin];
params = csv.inputValidation(varargin{:});

% Take first value since these inputs cannot differ between mice
recompute = params.recompute{1};
expDate = params.expDate{1};

% Get the server locations and load the main mouse csv
serverLocations = getServersList;
csvData = params.mainCSV{1};

% Checks that all the "active" mice are at the top of the main csv.  
% Sorts mice alphabetically. Any changes are saved.
activeMice = cellfun(@(x) x==1 || strcmp(x, '1'),csvData.IsActive);
csvData.IsActive = num2cell(num2str(activeMice));
csvDataSort = sortrows(csvData, 'Subject', 'ascend');
csvDataSort = sortrows(csvDataSort, 'IsActive', 'descend');
if any(~strcmp(csvDataSort.Subject, csvData.Subject))
    csv.createBackup(csvLocation);
    csv.writeClean(csvDataSort, csvLocation, 1)
end

% For each mouse that needs to be updated (default is 'active' mice),
% generate a list of folders to check. 
mice2Update = params.subject;
if recompute || (ischar(expDate) && strcmpi(expDate, 'all'))
    % If recompute is true, generate the list of "base" folders for each
    % mouse (because the entire folder needs to be checked since the csv 
    % is being remade).
    cycles = 2;
    paths2Check = cellfun(@(y) cellfun(@(x) [y x], mice2Update, 'uni', 0), serverLocations, 'uni', 0);
    paths2Check = vertcat(paths2Check{:});
else
    % If recompute is faulse, generate a list of folders with dates
    % corresponding to the "expDate" for each mouse. It is much quicker
    % to check all folders than to first check which of them exist
    cycles = 1;
    paths2Check = cellfun(@(y) cellfun(@(x) [y x], mice2Update, 'uni', 0), serverLocations, 'uni', 0);
    
    pastXDays = arrayfun(@(x) datestr(x, 'yyyy-mm-dd'), now-expDate:now, 'uni', 0)';
    paths2Check = cellfun(@(y) cellfun(@(x) [y filesep x], pastXDays, 'uni', 0), vertcat(paths2Check{:}), 'uni', 0);
    paths2Check = vertcat(paths2Check{:});
end

% Repeatedly get a list of all files withing the "paths2Check" list and
% ehck the contexts of those folders. There is an extra "cycle" if
% recompute is true because need to check subject/expDate/expNum rather
% than just checking expDate/expNum.
% Note, here I use java.io.File rather than MATLAB's dir for speed
for i = 1:cycles
    if isempty(paths2Check); continue; end
    fprintf('Detecting folder level %d ... \n', i);
    % Get list of all files within the each "paths2Check" cell
    paths2Check = cellfun(@(x) java.io.File(x), paths2Check, 'uni', 0);
    paths2Check = cellfun(@(x) arrayfun(@char,x.listFiles,'uni',0), paths2Check, 'uni', 0);
    paths2Check = vertcat(paths2Check{:});

    % NOTE: below are some checks that save time by removing paths/files
    % that don't need to be checked. Some of this may be redundant at
    % times, but since it's quick, that shouldn't matter.

    % Remove any files with certain keywords to save time
    paths2Check = paths2Check(~contains(paths2Check, {'Lightsheet';'ephys';'Backup';'g0'},'IgnoreCase',true));
    
    % Remove files that don't have the right number of digits at the end to
    % save time (this should be 10 if 2nd to last cycle (because it should
    % be a date) or <2 in the final cycle (an expNum)
    finalDigits = cellfun(@(x) length(x)-max(strfind(x, filesep)), paths2Check);
    if cycles - i == 1
        paths2Check(finalDigits~=10) = [];
    elseif cycles-i == 0
        paths2Check(finalDigits>2) = [];
    end

    % Remove paths where the final digit isn't a number
    paths2Check(isnan(cellfun(@(x) str2double(x(end)), paths2Check))) = [];
end

% Identify any duplicate entries (same data on multiple servers)
[~,uniIdx,pathIdx] = ...
    unique(cellfun(@(x) x(strfind(x, 'Subjects'):end), paths2Check, 'uni', 0));
duplicateEntries = unique(pathIdx(setdiff(1:numel(paths2Check), uniIdx)));
paths2Remove = zeros(length(paths2Check),1)>0;
% Loop over duplicate entries and remove the smallers one
for i = duplicateEntries'
    dupIdx = find(pathIdx==i);
    pathSize = cellfun(@(x) dir([fileparts(x) '/**/*']), paths2Check(dupIdx), 'uni', 0);
    pathSize = cellfun(@(x) sum([x.bytes])+rand, pathSize);
    paths2Remove(dupIdx(pathSize~=max(pathSize))) = 1;
end
% Remove duplicate entries
paths2Check(paths2Remove) = [];

%% Below is the loop that actually checks the paths and updates the csv
% Split each path at file separators and get "subList" which comprises the
% subject in every path
pathInfo = cellfun(@(x) split(x,filesep), paths2Check, 'uni', 0);
subList = cellfun(@(x) x{end-2}, pathInfo, 'uni', 0);

% Loop over each mouse and update the corresponding csv
for i = 1:length(mice2Update)

    % Get current subject and the corresponding indices from subList
    currSub = mice2Update{i};
    currIdx = contains(subList, currSub);

    % Get the dateList, expNumList, and csv path for the current mouse
    dateList = cellfun(@(x) x{end-1}, pathInfo(currIdx), 'uni', 0);
    expNumList = cellfun(@(x) x{end}, pathInfo(currIdx), 'uni', 0);
    csvPathMouse = csv.getLocation(currSub);
   
    if recompute && exist(csvPathMouse, 'file') && ~isempty(dateList)
        % If recompute is true, backup the old mouse csv and delete since 
        % this file will be completely recomputed
        csv.createBackup(csvPathMouse);
        delete(csvPathMouse);
    elseif exist(csvPathMouse, 'file') && ~isempty(dateList)
        % If recompute is not true, then don't rerun experiments that are
        % already enetered in the csv
        mouseCSV = csv.readTable(csvPathMouse);
        csvRef = strcat(mouseCSV.expDate, mouseCSV.expNum);
        procRef = strcat(dateList, expNumList);
        doneIdx = ismember(procRef, csvRef);
        dateList(doneIdx) = [];
        expNumList(doneIdx) = [];
    end

    % If dateList is empty, it indicates that no "new" paths exist for that
    % mouse, so continue to next loop
    if isempty(dateList)
        fprintf('No new data for %s. Skipping... \n', currSub);
        continue;
    end

    % Run "csv.updateRecord" on each detected experiment path to generate a
    % row of data in each case. Remove empty cells (where that row was
    % skipped for some reason, e.g. the experiment was <2 mins)
    newRecords = cellfun(@(x,y) csv.updateRecord('subject', currSub, ...
        'expDate', x, 'expNum', y, 'saveData', 0, 'queryExp', 0, 'mainCSV', csvData), dateList, expNumList, 'uni', 0);
    newRecords = newRecords(~cellfun(@isempty, newRecords));
    newRecords = vertcat(newRecords{:});
    
    if isempty(newRecords)
        fprintf('No new data for %s. Skipping... \n', currSub);
        continue;
    end

    % Concatenate the new records with the current subjects existing csv
    combinedData = csv.insertNewData(newRecords, currSub);    
    % Overwrite the old csv with the combined data
    csv.writeClean(combinedData, csvPathMouse, 0);
end