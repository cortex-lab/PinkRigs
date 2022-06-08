function writeClean(csvData, csvLocation, removeNaN)
%% Writes a "clean" csv file from "csvData"--meaning no "NaN" or "NaT"
if ~exist('removeNaN', 'var'); removeNaN = 0; end
csv.createBackup(csvLocation);
try
    writetable(csvData,csvLocation,'Delimiter',',');
    if removeNaN
        fid = fopen(csvLocation,'rt');
        backupDat = fread(fid);
        fclose(fid);
        backupDat = char(backupDat.');

        % replace string S1 with string S2
        replaceData = strrep(backupDat, 'NaN', '') ;
        replaceData = strrep(replaceData, 'NaT', '') ;

        fid = fopen(csvLocation,'wt') ;
        fwrite(fid,replaceData) ;
        fclose (fid) ;
    end
catch
    fprintf('Issue writing new exps for %s. May be in use. Skipping... \n', csvLocation);
end