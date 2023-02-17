function writeTable(csvData, csvLocation)
%% Writes a csv file from "csvData"--meaning no "NaN" or "NaT"
%
% Parameters:
% ------------
% csvData (required): table
%   the data (table) to be written
%
% csvLocation (required): string
%   the location where the csv will be saved

% Set default value for "removeNaN" (which is 0)

% We convert dates to have underscores for saving as this prevents
% instability in opening/closing of the excel files
if contains('expDate', csvData.Properties.VariableNames)
    csvData.expDate = cellfun(@(x) strrep(x, '-', '_'), csvData.expDate, 'uni', 0);
end

try
    % Write table with csv
    writetable(csvData,csvLocation,'Delimiter',',');
catch
    % If there is an issue with writing, the csv is probably open somewhere
    fprintf('Issue writing new exps for %s. May be in use. Skipping... \n', csvLocation);
end