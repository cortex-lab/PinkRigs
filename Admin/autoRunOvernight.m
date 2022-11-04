function autoRunOvernight
%% Functions that will run on all computers
computerType = getComputerType;
githubPath = fileparts(fileparts(which('autoRunOvernight.m')));

% Save log and close matlab session
logPath = 'C:\autoRunLog';
logFile = [regexprep(regexprep(datestr(now,31),' ','_'),':','-') '_log.txt'];
if ~exist(logPath,'dir')
    mkdir(logPath);
end
fid = fopen(fullfile(logPath,logFile),'wt');

fprintf(fid,'Starting now %s... \n',datestr(now));
try
    switch lower(computerType)
        case 'time'
            fprintf(fid,'Detected timeline computer... \n');
    
            fprintf(fid,'Running "copyLocalData2ServerAndDelete" (%s)... \n',datestr(now));
            copyLocalData2ServerAndDelete('D:\LocalExpData');
    
            fprintf(fid,'Running "runFacemap" (%s)... \n',datestr(now));
            % update environment
            eveningFacemapPath = [githubPath '\Analysis\\+vidproc\_facemap\run_facemap.py'];
            [statusFacemap, resultFacemap] = system(['activate facemap && ' ...
                'cd ' githubPath ' && ' ...
                'conda env update --file facemap_environment.yaml --prune' ' &&' ...
                'python ' eveningFacemapPath ' &&' ...
                'conda deactivate']);
            if statusFacemap > 0
                fprintf(fid,sprintf('Facemap failed with error "%s".\n', resultFacemap));
            end
    
            disp(resultFacemap);
            fprintf(fid,regexprep(resultFacemap,'\','/'));

            fprintf(fid,sprintf('Stopping now %s. \n',datestr(now)));
    
        case 'ephys'
            fprintf(fid,'Detected ephys computer... \n');
    
            fprintf(fid,'Running "copyLocalData2ServerAndDelete" (%s)... \n',datestr(now));
            copyLocalData2ServerAndDelete('D:\LocalExpData');
    
            fprintf(fid,'Running "extractLocalSync" (%s)... \n',datestr(now));
            extractLocalSync('D:\ephysData');
    
            fprintf(fid,'Compressing local data... (%s)... \n',datestr(now));
            compressPath = which('compress_data.py');
            [statusComp, resultComp] = system(['conda activate PinkRigs && ' ...
                'python ' compressPath ' && ' ...
                'conda deactivate']);
            if statusComp > 0
                error('Compressing local data failed with error: %s.', resultComp)
            end
    
            fprintf(fid,'Running "copyEphysData2ServerAndDelete" (%s)... \n',datestr(now));
            copyEphysData2ServerAndDelete('D:\ephysData');
    
            fprintf(fid,'Running "runFacemap" (%s)... \n',datestr(now));
            % update environment
            eveningFacemapPath = [githubPath '\Analysis\\+vidproc\_facemap\run_facemap.py'];
            [statusFacemap, resultFacemap] = system(['activate facemap && ' ...
                'cd ' githubPath ' && ' ...
                'conda env update --file facemap_environment.yaml --prune' ' &&' ...
                'python ' eveningFacemapPath ' &&' ...
                'conda deactivate']);
            if statusFacemap > 0
                fprintf(fid,sprintf('Facemap failed with error "%s".\n', resultFacemap));
            end
    
            disp(resultFacemap);
            fprintf(fid,regexprep(resultFacemap,'\','/'));

            fprintf(fid,sprintf('Stopping now %s. \n',datestr(now)));
    
        case {'kilo1'}
            %%
            fprintf(fid,'Detected kilo1 computer... \n');
    
            dbstop if error % temporarily, to debug
    
            fprintf(fid,'Running "csv.checkForNewPinkRigRecordings" (%s)... \n',datestr(now));
            csv.checkForNewPinkRigRecordings('expDate', 1);
    
            c = clock;
            if c(4) > 20 % trigger at 10pm 
                fprintf(fid,'Update on training (%s)... \n',datestr(now));
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
                    fprintf(fid,sprintf('Updating on training failed with error "%s".\n', resultTrain));
                end
            end
            
            c = clock;
            if c(4) > 20 || c(4) < 2
                Kilo_runFor = num2str(2); % 2hrs at 10pm/1am run 
            else
                Kilo_runFor = num2str(5); % 5 hrs at the 4am,10am & 4pm run 
            end

            fprintf(fid,'Running pykilosort on the queue for %s hours (%s)... \n',Kilo_runFor,datestr(now));
            runpyKS = [githubPath '\Analysis\+kilo\python_\run_pyKS.py'];
            [statuspyKS,resultpyKS] = system(['activate pyks2 && ' ...
                'python ' runpyKS ' ' Kilo_runFor ' && ' ...
                'conda deactivate']);
            if statuspyKS > 0
                fprintf(fid,sprintf('Running pyKS failed with error "%s".\n', resultpyKS));
            end
            disp(resultpyKS);
            fprintf(fid,regexprep(resultpyKS,'\','/'));

            % run at all times 
            fprintf(fid,'Creating the ibl format (%s)... \n',datestr(now));
            checkScriptPath = [githubPath '\Analysis\+kilo\python_\convert_to_ibl_format.py'];
            checkWhichMice = 'all';
            whichKS = 'pyKS';
            checkWhichDates = 'last300';
            [statusIBL,resultIBL] = system(['activate iblenv && ' ...
                'python ' checkScriptPath ' ' checkWhichMice ' ' whichKS ' ' checkWhichDates ' && ' ...
                'conda deactivate']);
            if statusIBL > 0
                fprintf(fid,sprintf('Creating IBL format failed with error "%s".\n', resultIBL));
            end
            disp(resultIBL);
            fprintf(fid,regexprep(resultIBL,'\','/'));
    
            c = clock;
            if c(4) < 20 && c(4) > 2 % should be triggered at 4am,10am,4pm
                %%% Bypassing preproc.main for now to go through experiments
                %%% that have been aligned but not preprocessed... Have to fix
                %%% it! Have to wait until it's a 0 and not a NaN when ephys
                %%% hasn't been aligned...
    
                fprintf(fid,'Running preprocessing (%s)... \n',datestr(now));
    
                % Alignment
                preproc.align.main('expDate', 7, 'checkAlignAny', '0')
    
                % Extracting data
                preproc.extractExpData('expDate', 7, 'checkSpikes', '0')
            end
            fprintf(fid,sprintf('Stopping now %s. \n',datestr(now)));
    
        case {'kilo2'}
            fprintf(fid,'Detected kilo2 computer... \n');
    
            c = clock;
            if c(4) > 20 || c(4) < 2
                Kilo_runFor = num2str(2);
            else
                Kilo_runFor = num2str(5);
            end
    
            dbstop if error % temporarily, to debug
            fprintf(fid,'Running pykilosort on the queue for %s hours (%s)... \n',Kilo_runFor,datestr(now));
            githubPath = fileparts(fileparts(which('autoRunOvernight.m')));
            runpyKS = [githubPath '\Analysis\+kilo\python_\run_pyKS.py'];
            [statuspyKS,resultpyKS] = system(['activate pyks2 && ' ...
                'python ' runpyKS ' ' Kilo_runFor ' && ' ...
                'conda deactivate']);
            if statuspyKS > 0
                fprintf(fid,sprintf('Running pyKS failed with error "%s".\n', resultpyKS));
            end
    
            disp(resultpyKS);
            fprintf(fid,regexprep(resultIBL,'\','/'));

            fprintf(fid,sprintf('Stopping now %s. \n',datestr(now)));
    
        case {'celians'}
            fprintf(fid,'Detected Celian''s computer... \n');
    
            c = clock;
            if c(4) > 20 || c(4) < 2
                Kilo_runFor = num2str(2);
            else
                Kilo_runFor = num2str(5);
            end
    
            dbstop if error % temporarily, to debug
            fprintf(fid,'Running pykilosort on the queue for %s hours (%s)... \n',Kilo_runFor,datestr(now));
            runpyKS = 'Analysis\+kilo\python_\run_pyKS.py';
            [statuspyKS,resultpyKS] = system(['activate pyks2 && ' ...
                'cd C:\Users\Hamish\OneDrive - University College London\Documents\GitHub\PinkRigs &&' ...
                'python ' runpyKS ' ' Kilo_runFor ' && ' ...
                'conda deactivate']);
            if statuspyKS > 0
                fprintf(fid,sprintf('Running pyKS failed with error "%s".\n', resultpyKS));
            end
    
            disp(resultpyKS);
            fprintf(fid,regexprep(resultIBL,'\','/'));

            fprintf(fid,sprintf('Stopping now %s. \n',datestr(now)));
    end


catch me
    fprintf(fid,sprintf('Global error: %s',me.message));
end
fclose(fid);

% Close matlab sessions
quit


end