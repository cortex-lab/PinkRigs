function createBackup(csvLocation)
%% Well create a timestamped backup of the CSV in the "Backups" folder
if ~exist(csvLocation, 'file')
    fprintf('WARNING: Requested backup, but file does not exist %s \n', csvLocation)
    return;
end

timeStamp = [datestr(now, 'YYmmDD') '_' datestr(now, 'HHMM')];
[folderLoc, fName] = fileparts(csvLocation);
csv.createBackupLoc = [folderLoc '\Backups\' timeStamp '_' fName '.csv'];
copyfile(csvLocation, csv.createBackupLoc);
end