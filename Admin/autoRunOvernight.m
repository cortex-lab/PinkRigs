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
        
        fprintf('Running "runFacemap" ... \n')
        eveningFacemapPath = which('evening_facemap.py');
        [statusFacemap resultFacemap] = system(['conda activate facemap && ' ...
            'python ' eveningFacemapPath ' &&' ...
            'conda deactivate']);
        if statusFacemap > 0
            fprintf('Facemap failed with error "%s".\n', resultFacemap)
        end
        
    case 'kilo1'
        fprintf('Detected kilo1 computer... \n')
        fprintf('Starting now %s...',datestr(now))
        
        dbstop if error % temporarily, to debug
        
        fprintf('Running "csv.checkForNewPinkRigRecordings"... \n')
        csv.checkForNewPinkRigRecordings(1);
        
        c = clock;
        if c(4) > 20
            fprintf('Update on training... \n')
            % Get plot of the mice trained today.
            paramsQuery.days2Check = 0;
            paramsQuery.expDef2Check = 'multiSpaceWorld_checker_training';
            expList = csv.queryExp(paramsQuery);
            [mouseNames, dates, expNums] = cellfun(@(x) parseExpPath(x), expList.expFolder, 'UniformOutput', false);
            opt.expNum = expNums;
            if ~isempty(mouseNames)
                plt.behaviour.boxPlots(mouseNames,dates, 'res',expList.expDef, opt)
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
        stageKSPath = which('stageKS.py');
        [statusQueue,resultQueue] = system(['activate PinkRigs && ' ...
            'python ' stageKSPath ' && ' ...
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
            paramsPreproc.days2Check = 7; % anything older than a week will be considered as "normal", will have to be manually rechecked
            % paramsPreproc.mice2Check = 'active';
            paramsPreproc.mice2Check = {'AV006','AV007','AV008','AV009'}; % for now to avoid crashes
            
            % Alignment
            paramsPreproc.align2Check = '(0,0,0,0,0,0)'; % "any 0"
            paramsPreproc.preproc2Check = '(*,*)';
            exp2checkList = csv.queryExp(paramsPreproc);
            preproc.align.main([], exp2checkList)
            
            % Extracting data
            paramsPreproc.align2Check = '(*,*,*,*,*,*)'; % "any 0"
            paramsPreproc.preproc2Check = '(0,0)';
            exp2checkList = csv.queryExp(paramsPreproc);
            preproc.extractExpData([], exp2checkList)
        end
        fprintf('Stopping now %s.',datestr(now))
end