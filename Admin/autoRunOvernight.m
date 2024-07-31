function autoRunOvernight
    %% Functions that will run on all computers
    % Each computer is going to run different sets of functions, depending
    % on its name (e.g., 'time', 'ephys', or 'kilo'). This function will be
    % called automatically at night, at various hours according to each
    % computer's schedule.

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
                copyLocalData2ServerAndDelete('D:\LocalExpData',fid);
                fprintf(fid,'Done (%s).\n',datestr(now));
        
                fprintf(fid,'Running "runFacemap" (%s)... \n',datestr(now));
                % update environment
                eveningFacemapPath = [githubPath '\Processing\_facemap\run_facemap.py'];
                [statusFacemap, resultFacemap] = system(['activate facemap && ' ...
                    'cd ' githubPath ' && ' ...
                    'start /affinity F python ' eveningFacemapPath ' &&' ...
                    'conda deactivate']);
                printMessage(statusFacemap,resultFacemap,fid)
                % 'conda env update --file facemap_environment.yaml --prune' ' &&' ...
    
                fprintf(fid,sprintf('Stopping now %s. \n',datestr(now)));
        
            case 'ephys'
                fprintf(fid,'Detected ephys computer... \n');
        
                fprintf(fid,'Running "copyLocalData2ServerAndDelete" (%s)... \n',datestr(now));
                copyLocalData2ServerAndDelete('D:\LocalExpData',fid);
                fprintf(fid,'Done (%s).\n',datestr(now));
        
                fprintf(fid,'Running "extractSyncAndCompress" (%s)... \n',datestr(now));
                extractSyncAndCompress('D:\ephysData',fid);
                fprintf(fid,'Done (%s).\n',datestr(now));
        
                fprintf(fid,'Running "copyEphysData2ServerAndDelete" (%s)... \n',datestr(now));
                copyEphysData2ServerAndDelete('D:\ephysData',fid);
                fprintf(fid,'Done (%s).\n',datestr(now));
        
                fprintf(fid,'Running "runFacemap" (%s)... \n',datestr(now));
                % update environment
                eveningFacemapPath = [githubPath '\Processing\_facemap\run_facemap.py'];
                [statusFacemap, resultFacemap] = system(['activate facemap && ' ...
                    'cd ' githubPath ' && ' ...
                    'python ' eveningFacemapPath ' &&' ...
                    'conda deactivate']);
                printMessage(statusFacemap,resultFacemap, fid)
                % 'conda env update --file facemap_environment.yaml --prune' ' &&' ...
    
                fprintf(fid,sprintf('Stopping now %s. \n',datestr(now)));
        
            case {'kilo1'}
                %%
                fprintf(fid,'Detected kilo1 computer... \n');
        
                dbstop if error % temporarily, to debug
        
                fprintf(fid,'Running "csv.checkForNewPinkRigRecordings" (%s)... \n',datestr(now));
                csv.checkForNewPinkRigRecordings('expDate', 1);
                fprintf(fid,'Done (%s).\n',datestr(now));
        
                c = clock;
                if c(4) > 20 % trigger at 10pm 
                    fprintf(fid,'Update on training (%s)... \n',datestr(now));
                    % Get plot of the mice trained today.
                    expList = csv.queryExp('expDate', 0, 'expDef', 'training');
                    if ~isempty(expList)
                        plts.behaviour.boxPlots(expList, 'sepPlots', 1)
                        saveas(gcf,fullfile('C:\Users\Experiment\Documents\BehaviorFigures',['Behavior_' datestr(datetime('now'),'dd-mm-yyyy') '.png']))
                        close(gcf)
                    end
        
                    % Check status and send email.
                    checkTrainingPath = which('check_training_mice.py');
                    [statusTrain,resultTrain] = system(['conda activate PinkRigs && ' ...
                        'python ' checkTrainingPath ' &&' ...
                        'conda deactivate']);
                    printMessage(statusTrain,resultTrain,fid)
                end
                
%                 c = clock;
%                 if c(4) > 20 || c(4) < 2
%                     Kilo_runFor = num2str(2); % 2hrs at 10pm/1am run 
%                 else
%                     Kilo_runFor = num2str(5); % 5 hrs at the 4am,10am & 4pm run 
%                 end
%
%                 fprintf(fid,'Running pykilosort on the queue for %s hours (%s)... \n',Kilo_runFor,datestr(now));
%                 runpyKS = [githubPath '\Processing\pykilo\run_pyKS.py'];
%                 [statuspyKS,resultpyKS] = system(['activate pyks2 && ' ...
%                     'python ' runpyKS ' ' Kilo_runFor ' && ' ...
%                     'conda deactivate']);
%                 printMessage(statuspyKS,resultpyKS,fid)
%     
%                 % run at all times 
%                 fprintf(fid,'Creating the ibl format (%s)... \n',datestr(now));
%                 checkScriptPath = [githubPath '\Processing\pykilo\convert_to_ibl_format.py'];
%                 checkWhichMice = 'all';
%                 whichKS = 'pyKS';
%                 checkWhichDates = 'previous1000';
%                 [statusIBL,resultIBL] = system(['activate iblenv && ' ...
%                     'python ' checkScriptPath ' ' checkWhichMice ' ' whichKS ' ' checkWhichDates ' && ' ...
%                     'conda deactivate']);
%                 printMessage(statusIBL,resultIBL,fid)
        
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
                    preproc.runBombcell('expDate', 30)

                    % Extracting data
                    preproc.extractExpData('expDate', 30, 'checkSpikes', '0')
    
                    
                    fprintf(fid,'Done (%s).\n',datestr(now));
                end
                fprintf(fid,sprintf('Stopping now %s. \n',datestr(now)));
        
            case {'kilo2'}
                fprintf(fid,'Detected kilo2 computer... \n');
%                 preproc.sorting()
                
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
                printMessage(statuspyKS,resultpyKS,fid)
    
                fprintf(fid,sprintf('Stopping now %s. \n',datestr(now)));
        end
    
    
    catch me
        fprintf(fid,sprintf('Global error: %s',regexprep(me.message,'\','/')));
    end
    fclose(fid);
    
    % Close matlab sessions
    quit
end

function printMessage(status,result,fid)
    %% Will print a message to file, and to terminal.

    result = regexprep(result,'\\','\\\\');
    result = regexprep(result,'%','%%');
    disp(result);
    if status > 0
        fprintf(fid,sprintf('Failed with error "%s".\n', result));
    else
        fprintf(fid,sprintf('%s.\n', result));
        fprintf(fid,'Done (%s).\n',datestr(now));
    end
end