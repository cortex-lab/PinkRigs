function csvData = readTable(csvPath)
%% This function reads CSV files with all cells being "characters"
%  This is generally imporant for consistency, as otherwise data types can
%  potentially change depending on, for example, if there are NaN's

% "csvPath" is the path to the csv file 

% These lines set the reading options of the csv to "char" for all columns
opts = detectImportOptions(csvPath, 'Delimiter',',');
dateFields = find(contains(opts.VariableNames, 'date','IgnoreCase',true));
opts = setvartype(opts, 'char');

% Read the csv
csvData = readtable(csvPath, opts');

% This loop should never be used anymore.. it was to deal with cases where
% the dates had been entered with backslash format instead of the standard
% format. If that happens, they are changed to standard format and the csv
% is overwritten. 
reWrite = 0;
for i = dateFields
    if any(contains(csvData.(opts.VariableNames{i}), {'\';'/';'-'}))
        badStr = '(?<year>\d+)-(?<month>\d+)-(?<day>\d+)';
        badFormat = ~cellfun(@isempty,regexp(csvData.(opts.VariableNames{i}),badStr));
        if ~isempty(badFormat) && any(badFormat)
            badDates = csvData.(opts.VariableNames{i})(badFormat);
            reWrite = 1;
            newDates = datetime(badDates,'InputFormat','yyyy-MM-dd');
            newDates = datestr(newDates, 'yyyy_mm_dd');
            csvData.(opts.VariableNames{i})(badFormat) = num2cell(newDates,2);
        end
    end
end
if reWrite == 1
    fprintf('WARNING: Check data format in %s. Should be uuuu_MM_dd!\n', csvPath)
    fprintf('WARNING: Updating %s with correct date formats... \n', csvPath)
    csv.createBackup(csvPath);
    csv.writeClean(csvData, csvPath);
end

if contains('expDate', opts.VariableNames)
    csvData.expDate = cellfun(@(x) strrep(x, '_', '-'), csvData.expDate, 'uni', 0);
end
end