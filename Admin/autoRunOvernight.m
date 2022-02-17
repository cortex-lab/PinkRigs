function autoRunOvernight
%% Functions that will run on timeline computers
computerType = getComputerType;

switch lower(computerType)
    case 'time'
        copyLocalData2ServerAndDelete;
    case 'ephys'
        copyLocalData2ServerAndDelete;
        copyEphysData2ServerAndDelete;
    case 'kilo1'
        checkForNewAVRecordings;
end

exit
