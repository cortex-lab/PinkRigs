function checkForNewPinkRigRecordings(days2Check, recompute, selectedMouse)
serverLocations = getServersList;

if ~exist('recompute', 'var'); recompute = 0; end
if ~exist('days2Check', 'var'); days2Check = 1; end

csvLocation = csv.getLocation('main');
csvData = csv.readTable(csvLocation);

activeMice = cellfun(@(x) x==1 || strcmp(x, '1'),csvData.IsActive);

csvData.IsActive = num2cell(num2str(activeMice));
csvDataSort = sortrows(csvData, 'Subject', 'ascend');
csvDataSort = sortrows(csvDataSort, 'IsActive', 'descend');
if any(~strcmp(csvDataSort.Subject, csvData.Subject))
    csv.createBackup(csvLocation);
    csv.writeClean(csvDataSort, csvLocation)
    csvData = csvDataSort;
end
mice2Update = csvData.Subject(activeMice);

if exist('selectedMouse', 'var') && ~isempty(selectedMouse)
    if ~iscell(selectedMouse); selectedMouse = {selectedMouse}; end
    mice2Update = selectedMouse;
elseif recompute
    mice2Update = csvData.Subject;
end

if recompute
    cycles = 2;
    paths2Check = cellfun(@(y) cellfun(@(x) [y x], mice2Update, 'uni', 0), serverLocations, 'uni', 0);
    paths2Check = vertcat(paths2Check{:});
else
    cycles = 1;
    paths2Check = cellfun(@(y) cellfun(@(x) [y x], mice2Update, 'uni', 0), serverLocations, 'uni', 0);
    
    pastXDays = arrayfun(@(x) datestr(x, 'yyyy-mm-dd'), now-days2Check:now, 'uni', 0)';
    paths2Check = cellfun(@(y) cellfun(@(x) [y filesep x], pastXDays, 'uni', 0), vertcat(paths2Check{:}), 'uni', 0);
    paths2Check = vertcat(paths2Check{:});
end

for i = 1:cycles
    if isempty(paths2Check); continue; end
    fprintf('Detecting folder level %d ... \n', i);
    paths2Check = cellfun(@(x) java.io.File(x), paths2Check, 'uni', 0);
    paths2Check = cellfun(@(x) arrayfun(@char,x.listFiles,'uni',0), paths2Check, 'uni', 0);
    paths2Check = vertcat(paths2Check{:});
    paths2Check = paths2Check(~contains(paths2Check, {'Lightsheet';'ephys';'Backup';'g0'},'IgnoreCase',true));
    finalDigits = cellfun(@(x) length(x)-max(strfind(x, filesep)), paths2Check);
    if cycles - i == 1
        paths2Check(finalDigits~=10) = [];
    elseif cycles-i == 0
        paths2Check(finalDigits>2) = [];
    end
    paths2Check(isnan(cellfun(@(x) str2double(x(end)), paths2Check))) = [];
end

pathInfo = cellfun(@(x) split(x,filesep), paths2Check, 'uni', 0);
subList = cellfun(@(x) x{end-2}, pathInfo, 'uni', 0);
%%
for i = 1:length(mice2Update)
    currSub = mice2Update{i};
    currIdx = contains(subList, currSub);
    dateList = cellfun(@(x) x{end-1}, pathInfo(currIdx), 'uni', 0);
    expNumList = cellfun(@(x) x{end}, pathInfo(currIdx), 'uni', 0);
    csvPathMouse = csv.getLocation(currSub);
    
    if isempty(dateList)
        fprintf('No new data for %s. Skipping... \n', currSub);
        continue;
    end
    %% If recompute, backup and delete
    if recompute && exist(csvPathMouse, 'file')
        csv.createBackup(csvPathMouse);
        delete(csvPathMouse);
    end
    
    newRecords = cellfun(@(x,y) csv.updateRecord(currSub, x, y, 0), dateList, expNumList, 'uni', 0);
    newRecords = newRecords(~cellfun(@isempty, newRecords));
    newRecords = vertcat(newRecords{:});
    
    combinedData = csv.insertNewData(newRecords, currSub);    
    csv.writeClean(combinedData, csvPathMouse, 0);
end