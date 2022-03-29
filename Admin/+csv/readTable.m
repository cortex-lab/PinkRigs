function csvData = readTable(csvPath)
%% Writes a "clean" csv file from "csvData"--meaning no "NaN" or "NaT" 
opts = detectImportOptions(csvPath);

dateFields = find(contains(opts.VariableNames, 'date','IgnoreCase',true));
opts = setvartype(opts, 'char');

% opts.VariableNamingRule = 'preserve';

csvData = readtable(csvPath, opts');

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