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
        
        fprintf('Update on training... \n')
        % call batch script
        statusTraining = system(['conda activate PinkRigs && ' ...
            'python C:\Users\Experiment\Documents\Github\PinkRigs\Admin\check_training_mice.py ' ...
            '&& conda deactivate']);
        if statusTraining == 0
            fprintf('Update on training failed.\n')
        end
        
        fprintf('Getting kilosort queue... \n')
        % call batch script
        statusUpdateQueue = system(['activate PinkRigs && ' ...
            'python C:\Users\Experiment\Documents\Github\PinkRigs\Admin\stageKS.py && ' ...
            'conda deactivate']);
        if statusUpdateQueue == 0
            fprintf('Updating kilosort queue failed.\n')
        end
        
        fprintf('Running kilosort on the queue... \n')
        kilo.main()
end