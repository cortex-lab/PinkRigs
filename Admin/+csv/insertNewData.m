function combinedTable = insertNewData(newData, subject)
%% Function to insert a table containing new data into a mouse CSV

% "newData" is a table with fields matching the existing  mouse csv. 
% "subject" is the name of the mouse that the new data corresponds to

% Check whether each field of "newData" is a cell. If not, then convert it
% to a cell
newDataTypes = varfun(@class,newData,'OutputFormat','cell');
newDataFields = newData.Properties.VariableNames;
for i = 1:length(newDataFields)
    if strcmpi(newDataTypes{i}, 'cell'); continue; end
    newData.(newDataFields{i}) = num2cell(newData.(newDataFields{i}));
end
newDataTypes = varfun(@class,newData,'OutputFormat','cell');

% If it wasn't possible to convert each field of "newData" into cells then
% generate an error
if ~all(contains(newDataTypes, 'cell'))
    error('Could not convert new data to cells for some reason...');
end

% Load the existing csv (if it exists) and remove and colums that have the
% save expDate/expNum as the "newData" (this prevents row duplication)
csvLocation = csv.getLocation(subject);
if ~exist(csvLocation, 'file')
    csvData = [];
else
    csvData = csv.readTable(csvLocation);
    csvRef = cellfun(@(x,y) [x,num2str(y)], csvData.expDate, csvData.expNum, 'uni', 0);
    insertRef = cellfun(@(x,y) [x,num2str(y)], newData.expDate, newData.expNum, 'uni', 0);
    csvData(contains(csvRef,insertRef),:) = [];
end

% Concatenate "newData" with modified "csvData" and sort according to 
% expNum and expDate 
csvData = [csvData;newData];

%%% NEEDS TEST: SHOULD MAKE IT SO '10' IS NOT BEFORE '2' %%%
% [~, sortRef] = sort(str2double(csvData.expNum));
% csvData = sortrows(csvData(sortRef,:), 'expDate', 'ascend');

csvData = sortrows(csvData, 'expNum', 'ascend');
csvData = sortrows(csvData, 'expDate', 'ascend');
combinedTable = csvData;
end