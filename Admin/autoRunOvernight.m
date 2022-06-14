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
        
        fprintf('Running "extractLocalSync"... \n')
        extractLocalSync('D:\ephysData');
        
        fprintf('Compressing local data... \n')     
        compressPath = which('compress_data.py');
        [statusComp,resultComp] = system(['conda activate PinkRigs && ' ...
            'python ' compressPath ' && ' ...
            'conda deactivate']);
        if statusComp > 0
            error('Compressing local data failed.')
        end
        
        fprintf('Running "copyEphysData2ServerAndDelete"... \n')
        copyEphysData2ServerAndDelete('D:\ephysData');
        
        fprintf('Running "runFacemap" ... \n')
        eveningFacemapPath = which('evening_facemap.py');
        [statusFacemap resultFacemap] = system(['conda activate facemap && ' ...
            'python ' eveningFacemapPath ' &&' ...
            'conda deactivate']);
        if statusFacemap > 0
            fprintf('Facemap failed with error "%s".\n', resultFacemap)
        end
        
    case {'kilo1','kilo2'}
      
        %%
        fprintf('Detected kilo computer... \n')
        fprintf('Starting now %s... \n',datestr(now))
        
        dbstop if error % temporarily, to debug
        
        fprintf('Running "csv.checkForNewPinkRigRecordings"... \n')
        csv.checkForNewPinkRigRecordings('days2check', 1);
        
        c = clock;
        if c(4) > 20
            fprintf('Update on training... \n')
            % Get plot of the mice trained today.
            expList = csv.queryExp('expDate', 0, 'expDef', 'training');
            if ~isempty(expList)
                plt.behaviour.boxPlots(expList, 'sepPlots', 1)
                saveas(gcf,fullfile('C:\Users\Experiment\Documents\BehaviorFigures',['Behavior_' datestr(datetime('now'),'dd-mm-yyyy') '.png']))
                close(gcf)
            end
            
            % Check status and send email.
            checkTrainingPath = which('check_training_mice.py');
            [statusTrain,resultTrain] = system(['conda activate PinkRigs && ' ...
                'python ' checkTrainingPath ' &&' ...
                'conda deactivate']);
            if statusTrain > 0
                fprintf('Updating on training failed with error "%s".\n', resultTrain)
            end
        end
        
        fprintf('Getting kilosort queue... \n')
        checkQueuePath = which('check_kilosort_queue.py');
        checkWhichMice = 'all';
        checkWhichDates = 'last7';
        [statusQueue,resultQueue] = system(['conda activate PinkRigs && ' ...
            'python ' checkQueuePath ' ' checkWhichMice ' ' checkWhichDates ' && ' ...
            'conda deactivate']);
        if statusQueue > 0
            fprintf('Updating the queue failed with error "%s".\n', resultQueue)
        end
        
        fprintf('Running kilosort on the queue... \n')
        if c(4) > 20 || c(4) < 2
            paramsKilo.runFor = 2; 
        else
            paramsKilo.runFor = 12;
        end
        kilo.main(paramsKilo)
        
        if c(4) < 20 && c(4) > 2
            %%% Bypassing preproc.main for now to go through experiments
            %%% that have been aligned but not preprocessed... Have to fix
            %%% it! Have to wait until it's a 0 and not a NaN when ephys
            %%% hasn't been aligned...
            
            fprintf('Running preprocessing...\n')
            
            % Alignment
            preproc.align.main('expDate', 7, 'align2Check', '(0,0,0,0,0,0)')
            
            % Extracting data
            preproc.extractExpData('expDate', 7, 'preproc2Check', '(0,0)')
        end
        fprintf('Stopping now %s. \n',datestr(now))
end