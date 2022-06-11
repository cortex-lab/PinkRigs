function combinedTable = insertNewData(newData, subject)
newDataTypes = varfun(@class,newData,'OutputFormat','cell');
newDataFields = newData.Properties.VariableNames;
for i = 1:length(newDataFields)
    if strcmpi(newDataTypes{i}, 'cell'); continue; end
    newData.(newDataFields{i}) = num2cell(newData.(newDataFields{i}));
end
newDataTypes = varfun(@class,newData,'OutputFormat','cell');
if ~all(contains(newDataTypes, 'cell'))
    error('Could not convert new data to cells for some reason...)');
end

csvLocation = csv.getLocation(subject);
if ~exist(csvLocation, 'file')
    csvData = [];
else
    csvData = csv.readTable(csvLocation);
    csvRef = cellfun(@(x,y) [x,num2str(y)], csvData.expDate, csvData.expNum, 'uni', 0);
    insertRef = cellfun(@(x,y) [x,num2str(y)], newData.expDate, newData.expNum, 'uni', 0);
    csvData(contains(csvRef,insertRef),:) = [];
end

csvData = [csvData;newData];
csvData = sortrows(csvData, 'expNum', 'ascend');
csvData = sortrows(csvData, 'expDate', 'ascend');
combinedTable = csvData;
end