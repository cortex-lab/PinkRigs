function removeDataRow(subject, expDate, expNum)
%% Function to remove row from a csv 
%  NOTE: At time of writing, this funciton is only called within 
%  csv.updateRecord but I made it external as it could be of general use

% "subject", "expDate", and "expNum" are all strings that indicate the csv
% (subject) and row (expDate/expNum) that needs to be removed

% Make sure expDate and expNum are both cells
if ~iscell(expDate); expDate = {expDate}; end
if ~iscell(expNum); expNum = {expNum}; end

% Locate mouse csv, and skip rest of function if it doesn't exist
csvPathMouse = csv.getLocation(subject);
if ~exist(csvPathMouse, 'file'); return; end

% Read data from mouse csv and identify the idx of the row to be removed
csvData = csv.readTable(csvPathMouse);
csvRef = strcat(csvData.expDate, csvData.expNum);
removeRef = strcat(expDate, expNum);

% Remove the row corresponding to function inputs and overwrite the csv
csvData(contains(csvRef,removeRef),:) = [];
csv.writeClean(csvData, csvPathMouse);
end