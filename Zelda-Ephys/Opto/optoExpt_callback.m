function optoExpt_callback(eventObj, LGObj)
%callback function run by expServer, called with inputs:
    %eventObj: from expServer containing various information about a trial
    %LGObj: laserGalvo object containing functions for laser
    % i.e. function that listens to updates from stim computer and
    % triggers the laser accordingly.
   % this callback is designed the interct with the expDef
   % \\znas.cortexlab.net\Code\Rigging\ExpDefinitions\PinkRigs\multiSpaceWorld_checker_training.m
   % 

if iscell(eventObj.Data) && strcmp(eventObj.Data{2}, 'experimentInit') %Experiment started
    expRef = eventObj.Ref;
    disp(['Starting Experiment: ' expRef]);
    %START LOG FILE
    LGObj.log = []; LGObj.filepath = [];
    LGObj.filepath = [dat.expPath(expRef, 'main', 'm') '\' expRef '_optoLog.mat'];    
    LGObj.laser.registerTrigger;    
    
elseif isstruct(eventObj.Data) && any(strcmp({eventObj.Data.name},'events.newTrial'))
    
    tic;    
    allT = tic;
    names = {eventObj.Data.name};
    values = {eventObj.Data.value};  

    trialNum = values{strcmp(names,'events.trialNum')};
    laserOn = values{strcmp(names,'events.is_laserOn')};     
    powerLaser1 = values{strcmp(names,'events.laser_power1')};
    powerLaser2 = values{strcmp(names,'events.laser_power2')};

    ROW = struct;    
    ROW.laser1_hemisphere = LGObj.laser.hemispheres{1};
    ROW.laser2_hemisphere = LGObj.laser.hemispheres{2};
    ROW.delay_readVar = toc;
    tic;    
    fprintf(['trialNum %03d) '], trialNum);
    %Setup waveforms depending on the trial configurations
        if laserOn>0 %If laser ON
            VoltLaser1 = LGObj.laser.power2volt(powerLaser1,1); % later do this based on calibration.  
            VoltLaser2 = LGObj.laser.power2volt(powerLaser2,2);
            laserV = LGObj.laser.generateWaveform(VoltLaser1,VoltLaser2);    
            ROW.delay_preallocLaserWaveform = toc;
            tic;        
            LGObj.laser.issueWaveform(laserV);            
            ROW.delay_issueLaser = toc;
        else
            ROW.delay_preallocLaserWaveform = nan;
            ROW.delay_issueLaser = nan; % append nan to log when wf is not issued.. 

        end
       

    %Save these details to a log
    ROW.trialNum = trialNum;
    ROW.is_laserOn = laserOn;
    ROW.powerLaser1 = powerLaser1;
    ROW.powerLaser2 = powerLaser2;  
    
    
    ROW.tictoc = toc(allT);
%     disp(['    total time: ' num2str(ROW.tictoc)]);
    LGObj.appendToLog(ROW);
    fprintf('\n');
    
elseif iscell(eventObj.Data) && strcmp(eventObj.Data{2}, 'experimentEnded')
    LGObj.stop;
    
    %Remove triggers
    LGObj.laser.removeTrigger;
    
    %Save log 
    LGObj.saveLog;       
    fprintf('Experiment Ended.\n');
end

end
