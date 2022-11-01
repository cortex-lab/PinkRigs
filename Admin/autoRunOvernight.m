function autoRunOvernight
%% Functions that will run on timeline computers
computerType = getComputerType;
githubPath = fileparts(fileparts(which('autoRunOvernight.m')));

log = '';
try
    switch lower(computerType)
        case 'time'
            log = append(log,'Detected timeline computer... \n');
    
            log = append(log,'Running "copyLocalData2ServerAndDelete"... \n');
            copyLocalData2ServerAndDelete('D:\LocalExpData');
    
            log = append(log,'Running "runFacemap" ... \n');
            % update environment
            eveningFacemapPath = [githubPath '\Analysis\\+vidproc\_facemap\run_facemap.py'];
            [statusFacemap, resultFacemap] = system(['activate facemap && ' ...
                'cd ' githubPath ' && ' ...
                'conda env update --file facemap_environment.yaml --prune' ' &&' ...
                'python ' eveningFacemapPath ' &&' ...
                'conda deactivate']);
            if statusFacemap > 0
                log = append(log,sprintf('Facemap failed with error "%s".\n', resultFacemap));
            end
    
            disp(resultFacemap);
    
        case 'ephys'
            log = append(log,'Detected ephys computer... \n');
    
            log = append(log,'Running "copyLocalData2ServerAndDelete"... \n');
            copyLocalData2ServerAndDelete('D:\LocalExpData');
    
            log = append(log,'Running "extractLocalSync"... \n');
            extractLocalSync('D:\ephysData');
    
            log = append(log,'Compressing local data... \n');
            compressPath = which('compress_data.py');
            [statusComp, resultComp] = system(['conda activate PinkRigs && ' ...
                'python ' compressPath ' && ' ...
                'conda deactivate']);
            if statusComp > 0
                error('Compressing local data failed with error: %s.', resultComp)
            end
    
            log = append(log,'Running "copyEphysData2ServerAndDelete"... \n');
            copyEphysData2ServerAndDelete('D:\ephysData');
    
            log = append(log,'Running "runFacemap" ... \n');
            % update environment
            eveningFacemapPath = [githubPath '\Analysis\\+vidproc\_facemap\run_facemap.py'];
            [statusFacemap, resultFacemap] = system(['activate facemap && ' ...
                'cd ' githubPath ' && ' ...
                'conda env update --file facemap_environment.yaml --prune' ' &&' ...
                'python ' eveningFacemapPath ' &&' ...
                'conda deactivate']);
            if statusFacemap > 0
                log = append(log,sprintf('Facemap failed with error "%s".\n', resultFacemap));
            end
    
            disp(resultFacemap);
    
        case {'kilo1'}
            %%
            log = append(log,'Detected kilo computer... \n');

            log = append(log,sprintf('Starting now %s... \n',datestr(now)));
    
            dbstop if error % temporarily, to debug
    
            log = append(log,'Running "csv.checkForNewPinkRigRecordings"... \n');
            csv.checkForNewPinkRigRecordings('expDate', 1);
    
            c = clock;
            if c(4) > 20 % trigger at 10pm 
                log = append(log,'Update on training... \n');
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
                    log = append(log,sprintf('Updating on training failed with error "%s".\n', resultTrain));
                end
            end
            
            c = clock;
            if c(4) > 20 || c(4) < 2
                Kilo_runFor = num2str(2); % 2hrs at 10pm/1am run 
            else
                Kilo_runFor = num2str(5); % 5 hrs at the 4am,10am & 4pm run 
            end

            log = append(log,sprintf('current hour is %.0f, running kilo for %s hours',c(4),Kilo_runFor));
            log = append(log,'Running pykilosort on the queue... \n');
            runpyKS = [githubPath '\Analysis\+kilo\python_\run_pyKS.py'];
            [statuspyKS,resultpyKS] = system(['activate pyks2 && ' ...
                'python ' runpyKS ' ' Kilo_runFor ' && ' ...
                'conda deactivate']);
            if statuspyKS > 0
                log = append(log,sprintf('Running pyKS failed... "%s".\n', resultpyKS));
            end
    
            disp(resultpyKS);
            log = append(log,resultpyKS);

            % run at all times 
            log = append(log,'creating the ibl format... \n');
            checkQueuePath = [githubPath '\Analysis\+kilo\python_\convert_to_ibl_format.py'];
            checkWhichMice = 'all';
            whichKS = 'pyKS';
            checkWhichDates = 'last300';
            [statusIBL,resultIBL] = system(['activate iblenv && ' ...
                'python ' checkQueuePath ' ' checkWhichMice ' ' whichKS ' ' checkWhichDates ' && ' ...
                'conda deactivate']);
            if statusIBL > 0
                log = append(log,sprintf('Updating the queue failed with error "%s".\n', resultIBL));
            end
            disp(resultIBL);
            log = append(log,sprintf('Stopping now %s. \n',datestr(now)));
    
            c = clock;
            if c(4) < 20 && c(4) > 2 % should be triggered at 4am,10am,4pm
                %%% Bypassing preproc.main for now to go through experiments
                %%% that have been aligned but not preprocessed... Have to fix
                %%% it! Have to wait until it's a 0 and not a NaN when ephys
                %%% hasn't been aligned...
    
                log = append(log,'Running preprocessing...\n');
    
                % Alignment
                preproc.align.main('expDate', 7, 'checkAlignAny', '0')
    
                % Extracting data
                preproc.extractExpData('expDate', 7, 'checkSpikes', '0')
            end
    
    
        case {'kilo2'}
            log = append(log,'Detected kilo2 computer... \n');

            log = append(log,sprintf('Starting now %s... \n',datestr(now)));
    
            c = clock;
            if c(4) > 20 || c(4) < 2
                Kilo_runFor = num2str(2);
            else
                Kilo_runFor = num2str(5);
            end
    
            dbstop if error % temporarily, to debug
            log = append(log,'Running pykilosort on the queue... \n');
            githubPath = fileparts(fileparts(which('autoRunOvernight.m')));
            runpyKS = [githubPath '\Analysis\+kilo\python_\run_pyKS.py'];
            [statuspyKS,resultpyKS] = system(['activate pyks2 && ' ...
                'python ' runpyKS ' ' Kilo_runFor ' && ' ...
                'conda deactivate']);
            if statuspyKS > 0
                log = append(log,sprintf('Running pyKS failed... "%s".\n', resultpyKS));
            end
    
            disp(resultpyKS);
            log = append(log,resultpyKS);
            log = append(log,sprintf('Stopping now %s. \n',datestr(now)));
    
        case {'celians'}
            log = append(log,'Detected kilo2 computer... \n');

            log = append(log,sprintf('Starting now %s... \n',datestr(now)));
    
            c = clock;
            if c(4) > 20 || c(4) < 2
                Kilo_runFor = num2str(2);
            else
                Kilo_runFor = num2str(5);
            end
    
            dbstop if error % temporarily, to debug
            log = append(log,'Running pykilosort on the queue... \n');
            runpyKS = 'Analysis\+kilo\python_\run_pyKS.py';
            [statuspyKS,resultpyKS] = system(['activate pyks2 && ' ...
                'cd C:\Users\Hamish\OneDrive - University College London\Documents\GitHub\PinkRigs &&' ...
                'python ' runpyKS ' ' Kilo_runFor ' && ' ...
                'conda deactivate']);
            if statuspyKS > 0
                log = append(log,sprintf('Running pyKS failed... "%s".\n', resultpyKS));
            end
    
            disp(resultpyKS);
    end


catch me
    log = append(log,me.message);
end

% Save log and close matlab session
logPath = 'C:\autoRunLog';
logFile = [regexprep(regexprep(datestr(now,31),' ','_'),':','-') '_log.txt'];
if ~exist(logPath,'dir')
    mkdir(logPath);
end
fid = fopen(fullfile(logPath,logFile),'wt');
fprintf(fid, log);
fclose(fid);
quit


end