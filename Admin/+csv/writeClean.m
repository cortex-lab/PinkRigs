function writeClean(csvData, csvLocation, removeNaN)
%% Writes a "clean" csv file from "csvData"--meaning no "NaN" or "NaT"
% 
% NOTE: It's likely the "clean" aspect of this function will only ever be
% used when writing to the "main" mouse csv. If it isn't used, all the
% empy cells in that csv will have NaN or NaT. This is difficult to read
%
% Parameters:
% ------------
% csvData (required): table
%   the data (table) to be written
%
% csvLocation (required): string
%   the location where the csv will be saved
%
% removeNaN (default = 0): logical
%   if 1 all cases of NaN and NaT should be removed

% Set default value for "removeNaN" (which is 0)
if ~exist('removeNaN', 'var'); removeNaN = 0; end

% We convert dates to have underscores for saving as this prevents
% instability in opening/closing of the excel files
if contains('expDate', csvData.Properties.VariableNames)
    csvData.expDate = cellfun(@(x) strrep(x, '-', '_'), csvData.expDate, 'uni', 0);
end

try
    % Write table with csv
    writetable(csvData,csvLocation,'Delimiter',',');
    if removeNaN
        % Open newly written csv
        fid = fopen(csvLocation,'rt');
        backupDat = fread(fid);
        fclose(fid);
        backupDat = char(backupDat.');

        % Replace all cases of NaN and NaT with empty strings
        replaceData = strrep(backupDat, 'NaN', '') ;
        replaceData = strrep(replaceData, 'NaT', '') ;
        
        % Open csv for writing and replace data with "clean" entries
        fid = fopen(csvLocation,'wt') ;
        fwrite(fid,replaceData) ;
        fclose (fid) ;
    end
catch
    % If there is an issue with writing, the csv is probably open somewhere
    fprintf('Issue writing new exps for %s. May be in use. Skipping... \n', csvLocation);
end