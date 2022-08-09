function createBackup(csvLocation)
%% Will create a timestamped backup of the CSV in the "Backups" folder
%  "csvLocation" is the full path of the .csv file to be backed up

% Check that file exists, and warn experimenter if it doesn't
if ~exist(csvLocation, 'file')
    fprintf('WARNING: Requested backup, but file does not exist %s \n', csvLocation)
    return;
end

% Get current date/times "timestamp" for backup filename 
timeStamp = [datestr(now, 'YYmmDD') '_' datestr(now, 'HHMM')];
[folderLoc, fName] = fileparts(csvLocation);

% Create backup file, named with appended timestamp, in a "Backups" folder
% which is placed in the same directory as the csv being backed up
backupLoc = [folderLoc '\Backups\' timeStamp '_' fName '.csv'];
copyfile(csvLocation, backupLoc);
end