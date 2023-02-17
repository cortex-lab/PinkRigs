function csvData = readTable(csvPath)
%% Reads CSV files with all cells being "characters"
%
% NOTE: This is imporant for consistency, as otherwise data types can  
% change depending on, for example, if there are NaN's
%
% Parameters:
% ---------------
% csvPath (required): string
%   Path of the csv file to read
%
% Returns: 
% ---------------
% csvData: table 
%   Table with all the loaded csv data

% These lines set the reading options of the csv to "char" for all columns
opts = detectImportOptions(csvPath, 'Delimiter',',');
opts = setvartype(opts, 'char');

% Read the csv
if ~contains(csvPath, 'docs.google.com')
    csvData = readtable(csvPath, opts');

    dateColumns = find(contains(opts.VariableNames,'Date'));
    for dd = 1:numel(dateColumns)
        csvData.(opts.VariableNames{dateColumns(dd)}) = cellfun(@(x) strrep(x, '_', '-'), csvData.(opts.VariableNames{dateColumns(dd)}), 'uni', 0);
    end
else
    docID = csvPath(strfind(csvPath, 'spreadsheets/d/')+15:strfind(csvPath, '/edit?')-1);
    csvData = csv.getGoogleSpreadsheet(docID);
    variableNames = csvData(1,:);
    csvData = cell2table(csvData(2:end,:), 'VariableNames', variableNames);
    dVars = variableNames(contains(variableNames, 'Date'));
    for i  =1:length(dVars)
        csvData.(dVars{i}) = cellfun(@(x) strrep(x, '_', '-'), csvData.(dVars{i}), 'uni', 0);
    end
end
end