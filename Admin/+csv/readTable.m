function csvData = readTable(csvPath)
%% This function reads CSV files with all cells being "characters"
%  This is generally imporant for consistency, as otherwise data types can
%  potentially change depending on, for example, if there are NaN's

% "csvPath" is the path to the csv file 

% These lines set the reading options of the csv to "char" for all columns
opts = detectImportOptions(csvPath);
dateFields = find(contains(opts.VariableNames, 'date','IgnoreCase',true));
opts = setvartype(opts, 'char');

% Read the csv
csvData = readtable(csvPath, opts');

% This loop should never be used anymore.. it was to deal with cases where
% the dates had been entered with backslash format instead of the standard
% format. If that happens, they are changed to standard format and the csv
% is overwritten. 
reWrite = 0;
for i = 1:dateFields
    if any(contains(csvData.(opts.VariableNames{i}), {'\';'/'}))
        reWrite = 1;
        badStr = '(?<day>\d+)/(?<month>\d+)/(?<year>\d+)';       
        badFormat = ~cellfun(@isempty,regexp(csvData.(opts.VariableNames{i}),badStr));
        badDates = csvData.(opts.VariableNames{i})(badFormat);
        newDates = datetime(badDates,'InputFormat','dd/MM/yyyy');
        newDates = datestr(newDates, 'yyyy-mm-dd');
        csvData.(opts.VariableNames{i})(badFormat) = num2cell(newDates,2);
    end
end
if reWrite == 1
    fprintf('WARNING: Check data format in %s. Should be uuuu-MM-dd!', csvPath)
    fprintf('WARNING: Updating Main .csv with correct date formats... \n')
    csv.createBackup(csvPath);
    csv.writeClean(csvData, csvPath)
end
end