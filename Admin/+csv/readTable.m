function csvData = readTable(csvPath)
%% Reads CSV files with all cells being "characters"
%
% NOTE: This is imporant for consistency, as otherwise data types can  
% change depending on, for example, if there are NaN's
%
% Parameters:
% ------------
% csvPath (required): string
% ----Path of the csv file to read
% ----
%
% Returns: 
% ---------------
%
% csvData: table 
% ----Table with all the loaded csv data

% These lines set the reading options of the csv to "char" for all columns
opts = detectImportOptions(csvPath, 'Delimiter',',');
opts = setvartype(opts, 'char');

% Read the csv
csvData = readtable(csvPath, opts');

dateColumns = find(contains(opts.VariableNames,'Date'));
for dd = 1:numel(dateColumns)
    csvData.(opts.VariableNames{dateColumns(dd)}) = cellfun(@(x) strrep(x, '_', '-'), csvData.(opts.VariableNames{dateColumns(dd)}), 'uni', 0);
end
end