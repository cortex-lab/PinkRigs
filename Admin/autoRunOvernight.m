function autoRunOvernight
%% Functions that will run on timeline computers
computerType = getComputerType;

switch lower(computerType)
    case 'time'
        fprintf('Detected timeline computer... \n')
        
        fprintf('Running "copyLocalData2ServerAndDelete"... \n')
        copyLocalData2ServerAndDelete('D:\LocalExpData');
        
        fprintf('Running "runFacemap" ... \n')
        eveningFacemapPath = which('evening_facemap.py');
        [statusFacemap resultFacemap] = system(['conda activate facemap && ' ...
            'python ' eveningFacemapPath ' &&' ...
            'conda deactivate']);
        if statusFacemap > 0
            fprintf('Facemap failed with error "%s".\n', resultFacemap)
        end
        
    case 'ephys'
        fprintf('Detected ephys computer... \n')
        
        fprintf('Running "copyLocalData2ServerAndDelete"... \n')
        copyLocalData2ServerAndDelete('D:\LocalExpData');
        
        fprintf('Running "copyEphysData2ServerAndDelete"... \n')
        copyEphysData2ServerAndDelete('D:\ephysData');
        
    case 'kilo1'
        fprintf('Detected kilo1 computer... \n')
        
        fprintf('Running "checkForNewAVRecordings"... \n')
        checkForNewAVRecordings;
        
        fprintf('Update on training... \n')
        checkTrainingPath = which('check_training_mice.py');
        [statusTrain,resultTrain] = system(['conda activate PinkRigs && ' ...
            'python ' checkTrainingPath ' &&' ...
            'conda deactivate']);
        if statusTrain > 0
            fprintf('Updating on training failed with error "%s".\n', resultTrain)
        end
        
        fprintf('Getting kilosort queue... \n')
        stageKSPath = which('stageKS.py');
        [statusQueue,resultQueue] = system(['activate PinkRigs && ' ...
            'python ' stageKSPath ' && ' ...
            'conda deactivate']);
        if statusQueue > 0
            fprintf('Updating the queue failed with error "%s".\n', resultQueue)
        end
        
        fprintf('Running kilosort on the queue... \n')
        param.checkTime = 1; % to stop it after about 20h
        kilo.main(param)
end