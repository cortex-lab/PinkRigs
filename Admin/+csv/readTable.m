function csvData = readTable(csvPath)
%% Writes a "clean" csv file from "csvData"--meaning no "NaN" or "NaT" 
opts = detectImportOptions(csvPath);

dateFields = opts.VariableNames(strcmp(opts.VariableTypes, 'datetime'));
opts = setvartype(opts, 'char');

csvData = readtable(csvPath, opts);

for i = 1:length(dateFields)
    if any(contains(csvData.(dateFields{i}), {'\';'/'}))
        sprintf('WARNING: Check data format in %s. Should be uuuu-MM-dd!', csvPath)
    end
end
end