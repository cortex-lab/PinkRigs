function combinedTable = insertNewData(newData, subject)
%% Insert a table containing new data into an existing mouse CSV
%
% Parameters:
% ------------
% newData (required): table
%   a table with fields matching the existing  mouse csv
%
% subject (required): string
%   the subjected where the new data should be inserted
%
% Returns: 
% ------------
% combinedTable: table 
%   A new, sorted table with data inserted
%
% Examples: 
% ------------
% csv.insertNewData(newData, 'AV008');


% If no subject provided, then error
if ~exist('subject', 'var') || isempty(subject)
    error('No subject provided for data insertion... \n')
end

% If "newData" is empty then "return"
if ~exist('newData', 'var') || isempty(newData)
    fprintf('Asked to write empty data for %s, returning... \n', subject)
    return
end

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
    csvData(ismember(csvRef,insertRef),:) = [];
end

% Concatenate "newData" with modified "csvData" and sort according to 
% expNum and expDate 
csvData = [csvData;newData];

% Sort csv by date, and expNum within each date
[~, sortRef] = sort(str2double(csvData.expNum));
csvData = sortrows(csvData(sortRef,:), 'expDate', 'ascend');

% Assign output
combinedTable = csvData;
end