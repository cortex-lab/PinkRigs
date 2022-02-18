function autoRunOvernight
%% Functions that will run on timeline computers
computerType = getComputerType;

switch lower(computerType)
    case 'time'
        fprintf('Detected timeline computer... \n')
        
        fprintf('Running "copyLocalData2ServerAndDelete"... \n')
        copyLocalData2ServerAndDelete;
        
    case 'ephys'
        fprintf('Detected ephys computer... \n')
        fprintf('Running "copyLocalData2ServerAndDelete"... \n')
        copyLocalData2ServerAndDelete;
        
        fprintf('Running "copyEphysData2ServerAndDelete"... \n')
        copyEphysData2ServerAndDelete;
        
    case 'kilo1'
        fprintf('Detected kilo1 computer... \n')
        
        fprintf('Running "checkForNewAVRecordings"... \n')
        checkForNewAVRecordings;
        
        fprintf('Sending training summary... \n')
        % call batch script
        statusTraining = system('C:\Users\Experiment\Documents\Github\PinkRigs\Admin\+training\check_training.bat');
        
        fprintf('Getting kilosort queue... \n')
        % call batch script
        statusUpdateQueue = system('C:\Users\Experiment\Documents\Github\PinkRigs\Admin\updateKilosortQueue.bat');
        
        fprintf('Running "checkForNewAVRecordings"... \n')
        kilo.main()
end