function ev = multiSpaceTraining(timeline, block, alignmentBlock)
    %% Fetches all important information from the imageWorld protocols
    %
    % Parameters:
    % -------------------
    % timeline: struct
    %   Timeline structure.
    % block: struct
    %   Block structure
    % alignmentBlock: struct
    %   Alignement structure, containing fields "originTimes" and
    %   "timelineTimes"
    %
    % Returns: 
    % -------------------
    % ev: struct
    %   Structure containing all relevant events information.
    %   All fields should have the form [nxm] where n is the number of trials.
    %   FOR TIMES: all relative to trial start
    %   FOR DIRECTIONS: 2 =  rightward choice, 1 = leftward choice
    %       ev.is_blankTrial      %logical: indicating "blank" trials
    %       ev.is_visualTrial     %logical: indicating "visual" trials
    %       ev.is_auditoryTrial   %logical: indicating "auditory" trials
    %       ev.is_coherentTrial   %logical: indicating "coherent" trials
    %       ev.is_conflictTrial   %logical: indicating "conflict" trials
    %       ev.is_validTrial      %logical: indicating "valid" trials (used for analysis)
    % 
    %       ev.block_trialOnOff   %nx2 matrix: [starttime endtime]
    %       ev.block_stimOn       %nx1 matrix: start times for stimulus period
    % 
    %       ev.timeline_rewardOn  %nx1 cell: reward times (manual rewards included)
    %       ev.timeline_audOnOff  %nx1 cell: [on off] times for aud stimuli (each click)
    %       ev.timeline_visOnOff  %nx1 cell: [on off] times for vis stimuli (each flash)
    % 
    %       ev.timeline_audPeriodOnOff %nx2 matrix: [on off] times for the "whole" aud stimulus
    %       ev.timeline_visPeriodOnOff %nx2 matrix: [on off] times for the "whole" vis stimulus
    %       ev.timeline_firstMoveOn    %nx1 matrix: time for the first movement initiation
    %       ev.timeline_firstMoveDir   %nx1 matrix: direction of first movement initiation
    %       ev.timeline_choiceMoveOn   %nx1 matrix: time of "choice" movement initiation
    %       ev.timeline_choiceMoveDir  %nx1 matrix: direction of "choice" movement
    %       ev.timeline_choiceThreshOn %nx2 matrix: time that wheel crosses decision threshold
    %       ev.timeline_allMoveOn      %nx1 cell:   times for all movement onsets
    %       ev.timeline_allMoveDir     %nx1 cell:   direction for all movement onsets
    %       ev.timeline_wheelTimeValue %nx2 cell:   [times wheelPosition(deg)]
    % 
    %       ev.stim_correctResponse     %nx1 matrix: correct answer NOT mouse choice ev.stim_audAmplitude        %nx1 matrix: aud amplitude
    %       ev.stim_audAzimuth          %nx1 matrix: aud azimuth presented
    %       ev.stim_visContrast         %nx1 matrix: vis contrast
    %       ev.stim_visAzimuth          %nx1 matrix: vis azimuth presented
    % 
    %       ev.response_direction      %nx1 matrix. recorded response (1/2 for left/right)
    %       ev.response_feedback       %nx1 matrix. -1/0/1 for incorrect/timeout/reward
      
        
    %% Convert to shorter names for ease of use later
    e = block.events;                     %Event structure
    v = block.paramsValues;  %Parameter values at start of trial
    vIdx = e.repeatNumValues(1:length(e.endTrialTimes))==1;            %Indices of valid trials (0 for repeats)

    %% The number of repeats and timeouts for each trial type presented
    %Invalidate trials that are repeats following an incorrect choice (because the mouse knows which way to go based on the incorrect choice) and trials
    %where there were multiple repeats because of a timeout (i.e. only the initial timeout response is counted as valid)
    for i = find(vIdx)
        if i < length(vIdx) && e.responseTypeValues(i) == 0
            nextResponse = min([i+find(e.responseTypeValues(i+1:length(vIdx))~=0 | e.repeatNumValues(i+1:length(vIdx))==1,1), length(vIdx)+1]);
            if e.repeatNumValues(nextResponse)==1 || nextResponse >= length(vIdx); continue; end
            vIdx(nextResponse) = 1;
        end
    end

    %% Extract meaningful data from the block file
    %eIdx is just an logical for all trials that ended (if the experiment ends mid-trial, there may be an extra index for some events)
    eIdx = 1:length(e.endTrialTimes);
    vIdx = vIdx(eIdx);

    if isfield(v, 'audAmplitude')
        audAmplitude = [v(eIdx).audAmplitude]';               %Convert amplitudes to matrix. Assumes one value for each trial.
        visContrast = [v(eIdx).visContrast]';                 %Convert amplitudes to matrix. Assumes one value for each trial.
        %     correctResponse = [v(eIdx).correctResponse]';         %Convert correctResponse on each trial to matrix. Assumes one value for each trial.
        correctResponse = e.correctResponseValues(eIdx)';         %Convert correctResponse on each trial to matrix. Assumes one value for each trial.
        audInitialAzimuth = e.audInitialAzimuthValues(eIdx)';     %Convert audInitialAzimuth on each trial to matrix. Assumes one value for each trial.
        visInitialAzimuth = e.visInitialAzimuthValues(eIdx)';     %Convert visInitialAzimuth on each trial to matrix. Assumes one value for each trial.
        clickRate = block.paramsValues.clickRate;
        clickDuration = block.paramsValues.clickDuration;
    else
        audAmplitude = e.audAmplitudeValues(eIdx)';               %Convert amplitudes to matrix. Assumes one value for each trial.
        visContrast = e.visContrastValues(eIdx)';                 %Convert contrast to matrix. Assumes one value for each trial.
        correctResponse = e.correctResponseValues(eIdx)';         %Convert correctResponse on each trial to matrix. Assumes one value for each trial.
        audInitialAzimuth = e.audInitialAzimuthValues(eIdx)';     %Convert audInitialAzimuth on each trial to matrix. Assumes one value for each trial.
        visInitialAzimuth = e.visInitialAzimuthValues(eIdx)';     %Convert visInitialAzimuth on each trial to matrix. Assumes one value for each trial.
        clickRate = block.events.selected_paramsetValues.clickRate;
        clickDuration = block.events.selected_paramsetValues.clickDuration;
    end
    audInitialAzimuth(audAmplitude==0) = nan;             %Change case when audAmplitude was 0 to have nan azimuth (an indication of no azimuth value)
    visInitialAzimuth(visContrast==0) = nan;              %Change case when visContrast was 0 to have nan azimuth (an indication of no azimuth value)

    % Get block if switching task
    if isfield(block.events, 'currentBlockValues')
        noRepTrialIdx = [1 find(~diff(block.events.totalRepeatsValues))+1];
        currentBlockValues = interp1(noRepTrialIdx, block.events.currentBlockValues, eIdx)';
    end

    %Get trial start/end times, stim start times, closed loop start times, feedback times, etc.
    stimPeriodStart = e.stimPeriodOnOffTimes(e.stimPeriodOnOffValues == 1)';
    stimPeriodStart = stimPeriodStart(eIdx);
    feedbackValues = e.feedbackValues(eIdx)';
    timeOuts = feedbackValues==0;

    %%
    %Calculate an approximate time to the first wheel movement. This is different from the "timeToFeedback" in that it is based on wheel movement, rather
    %than the time when the threshold was reached. WheelMove is an interpolation of the wheel movement (to get it's nearest position at every ms).

    %Define a sample rate (sR--used the same as timeline) and resample wheelValues at that rate using 'pchip' and 'extrap' to get rawWheel. Get the
    %indices for stimOnset and feedback based on event times and sR.
    % PIP SHOULD WRITE A FUNCTION TO DO THIS...

    %Get the response the mouse made on each trial based on the correct response and then taking the opposite for incorrect trials. NOTE: this will not
    %work for a task with more than two response options.
    responseRecorded = double(correctResponse).*~timeOuts;
    responseRecorded(feedbackValues<0) = -1*(responseRecorded(feedbackValues<0));
    responseRecorded = ((responseRecorded>0)+1).*(responseRecorded~=0);
    correctResponse = ((correctResponse>0)+1).*(correctResponse~=0);

    % good sanity check to keep if mean(responseCalc(~isnan(responseCalc)) == responseRecorded(~isnan(responseCalc))) < 0.50 && sum(~isnan(responseCalc)) >= 50
    %     warning('Why are most of the movements not in the same direction as the response?!?');
    %     keyboard;
    % end

    %Create a "logical" for each trial type (blank, auditory, visual, coherent, and incoherent trials)
    is_noStimTrial = isnan(visInitialAzimuth) & isnan(audInitialAzimuth);
    is_blankTrial = (visContrast==0 | visInitialAzimuth==0) & (audAmplitude>0) & (audInitialAzimuth==0);
    is_auditoryTrial = (visContrast==0 | visInitialAzimuth==0) & (audAmplitude>0 & audInitialAzimuth~=0);
    is_visualTrial = (audAmplitude==0 | audInitialAzimuth==0) & (visContrast>0 & visInitialAzimuth~=0);
    is_coherentTrial = sign(visInitialAzimuth.*audInitialAzimuth)>0 & audAmplitude>0 & visContrast>0;
    is_conflictTrial = sign(visInitialAzimuth.*audInitialAzimuth)<0 & audAmplitude>0 & visContrast>0;

    %% Info from timeline!! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    timelineTime = timeline.rawDAQTimestamps;             %Timestamps in the timeline file
    sR = 1/diff(timeline.rawDAQTimestamps(1:2));          %Timeline sample rate

    trialStTimes = preproc.align.event2Timeline(block.events.newTrialTimes, ...
        alignmentBlock.originTimes,alignmentBlock.timelineTimes);
    trialEnTimes = preproc.align.event2Timeline(block.events.endTrialTimes, ...
        alignmentBlock.originTimes,alignmentBlock.timelineTimes);
    trialStEnTimes = [trialStTimes(eIdx)' trialEnTimes(eIdx)'];

    stimStartBlock = preproc.align.event2Timeline(block.events.stimPeriodOnOffTimes, ...
        alignmentBlock.originTimes,alignmentBlock.timelineTimes);
    stimStartBlock = stimStartBlock(1:2:end);
    stimStartBlock = stimStartBlock(eIdx)';

    %% Reward times using standard code
    tExt.rewardTimes = timeproc.getChanEventTime(timeline, 'rewardEcho')';

    %% Extract audio clicks (these are pretty reliable, so can extract every click)
    %Detrend timeline trace, threshold using kmeans, detect onsets and offsets of sound, estimate duration from this.
    audTrace = timeproc.extractChan(timeline,'audioOut');
    audTrace = [0;diff(detrend(audTrace))];
    [~, thresh] = kmeans(audTrace(1:5:end),5);
    timelineClickOn = timelineTime(strfind((audTrace>max(thresh)*0.2)', [0 1]));
    timelineClickOff = timelineTime(strfind((audTrace<min(thresh)*0.2)', [0 1]));
    detectedDuration = round(mean(timelineClickOff-timelineClickOn)*1000);

    %Sanity check: same number of onsets and offsets, check that detected duration matches the duration parameter (assumed constant here)
    if length(timelineClickOn)~=length(timelineClickOff)
        error('There should always be an equal number on/off signals for clicks');
    end
    if abs(detectedDuration-(unique(clickDuration)*1000))>3
        error('Diff in detected and requested click durations');
    end

    %Create vector that is sorted by time: [onset time, offset time, 1, 0] and
    % find large gaps between successive onsets (stimulus period onsets)
    trialGapThresh = 1./max(clickRate)*5;
    audPeriodOnTimeline = timelineClickOn(diff([0,timelineClickOn])>trialGapThresh)';
    audPeriodOffTimeline = timelineClickOff(diff([timelineClickOff, 10e10])>trialGapThresh)';
    audPeriodOnOffTimeline = [audPeriodOnTimeline, audPeriodOffTimeline];
    audPeriodOnOffTimeline(diff(audPeriodOnOffTimeline,[],2)<(1/2000),:) = [];

    %Sanity check (should be match between stim starts from block and from timeline)
    compareTest = @(x,y) (getNearestPoint(x(:)',y(:)')-(1:length(x(:))))';

    nonAudTrials = audAmplitude(eIdx) == 0;
    stimStartRef = stimStartBlock(~nonAudTrials);

    if any(compareTest(stimStartRef, audPeriodOnOffTimeline))
        fprintf('****WARNING: problem matching auditory stimulus start and end times \n');
        fprintf('****Will try removing points that do not match stimulus starts \n');

        [~, nearestPoint] = getNearestPoint(audPeriodOnOffTimeline(:,1), stimStartRef);
        audPeriodOnOffTimeline(nearestPoint>0.75,:) = [];
    end
    if any(compareTest(stimStartRef, audPeriodOnOffTimeline))
        fprintf('****WARNING: Could not fix start-end times\n');
        fprintf('****Will perform incomplete identification based on trial structure\n');

        audBoundsByTrial = indexByTrial(trialStEnTimes(~nonAudTrials,:), sort(audPeriodOnOffTimeline(:)));
        audBoundsByTrial(cellfun(@length, audBoundsByTrial)~=2) = [];
        audPeriodOnOffTimeline = cell2mat(cellfun(@(x) x', audBoundsByTrial, 'uni', 0));
    else
        audPeriodOnOffTimeline = audPeriodOnOffTimeline(1:length(stimStartRef),:);
    end
    tExt.audStimOnOff = [timelineClickOn' timelineClickOff'];
    tExt.audStimPeriodOnOff = audPeriodOnOffTimeline;

    %% Will loop through photodiode types in case one is broken
    try
        %% Extract visual onsets (unreliable after initial flip)
        %Detrend timeline trace, threshold using kmeans, detect onsets and offsets of sound, estimate duration from this.
        [photoDiodeFlipTimes, photoName] = timeproc.extractBestPhotodiode(timeline, block);
        fprintf('****Using %s channel for photodiode...\n', photoName);
        trialGapThresh = 1./max(clickRate)*5;
        visPeriodOnTimeline = photoDiodeFlipTimes(diff([0,photoDiodeFlipTimes])>trialGapThresh)';
        visPeriodOffTimeline = photoDiodeFlipTimes(diff([photoDiodeFlipTimes, 10e10])>trialGapThresh)';
        visPeriodOnOffTimeline = [visPeriodOnTimeline, visPeriodOffTimeline];
        visPeriodOnOffTimeline(diff(visPeriodOnOffTimeline,[],2)<(1/2000),:) = [];

        %Sanity check (should be match between stim starts from block and from timeline)
        compareTest = @(x,y) (getNearestPoint(x(:)',y(:)')-(1:length(x(:))))';

        nonVisTrials = visContrast(eIdx)==0;
        stimStartRef = stimStartBlock(~nonVisTrials);
        if any(compareTest(stimStartRef, visPeriodOnOffTimeline(:,1)))
%             fprintf('****Removing photodiode times that do not match stimulus starts \n');

            [~, nearestPoint] = getNearestPoint(visPeriodOnOffTimeline(:,1), stimStartRef);
            visPeriodOnOffTimeline(nearestPoint>0.75,:) = [];
        end

        if any(compareTest(stimStartRef, visPeriodOnOffTimeline(:,1)))
            fprintf('****WARNING: Could not fix start-end times\n');
            fprintf('****Will perform incomplete identification based on trial structure\n');

            visBoundsByTrial = indexByTrial(trialStEnTimes(~nonVisTrials,:), sort(visPeriodOnOffTimeline(:)));
            visBoundsByTrial(cellfun(@length, visBoundsByTrial)~=2) = [];
            visPeriodOnOffTimeline = cell2mat(cellfun(@(x) x', visBoundsByTrial, 'uni', 0));
        else
            visPeriodOnOffTimeline = visPeriodOnOffTimeline(1:length(stimStartRef),:);
        end
        tExt.visStimPeriodOnOff = visPeriodOnOffTimeline;

        % Could add this in for passive
        if ~isempty(visPeriodOnOffTimeline)
            photoFlipsByTrial = indexByTrial(visPeriodOnOffTimeline, photoDiodeFlipTimes(:));
            photoFlipsByTrial = indexByTrial(trialStEnTimes(~nonVisTrials,:), cell2mat(photoFlipsByTrial));
        else
            % audio only
            photoFlipsByTrial = {};
        end
        if isfield(block.events,'selected_paramsetValues')
            responseWindow = block.events.selected_paramsetValues.responseWindow;
        else
            responseWindow = [block.paramsValues.responseWindow];
            responseWindow = responseWindow(1);
        end
        if isinf(responseWindow); responseWindow = 0; end
        expectedFlashTrainLength = clickRate*responseWindow*2*(stimStartRef*0+1);
        misMatchFlashtrain = expectedFlashTrainLength-cellfun(@length,photoFlipsByTrial);

        repeatNums = e.repeatNumValues(eIdx)';
        stimMoves = repeatNums*0;

        %To deal with old mice where closed loop never changed
        if ~isfield(block.events, 'wheelMovementOnValues')
            block.events.wheelMovementOnValues = block.events.newTrialValues;
        end
        stimMoves(repeatNums==1) = block.events.wheelMovementOnValues(1:sum(repeatNums==1))';
        stimMoves = arrayfun(@(x) stimMoves(find(repeatNums(1:x)==1, 1, 'last')), (1:length(eIdx))');
        stim_closedLoop = stimMoves;
        stimMoves = stimMoves(~nonVisTrials);

        isTimeOut = responseRecorded(~nonVisTrials)==0;
        photoFlipsByTrial((~isTimeOut & stimMoves) | (isTimeOut & misMatchFlashtrain~=0)) = [];
        photoFlipsByTrial(cellfun(@length, photoFlipsByTrial) < 2) = [];
        photoFlipsByTrial = cellfun(@(x) x(1:(floor(length(x)/2)*2)), photoFlipsByTrial, 'uni', 0);

        visStimOnOffTimes = sort(cell2mat(photoFlipsByTrial));
        tExt.visStimOnOff = [visStimOnOffTimes(1:2:end) visStimOnOffTimes(2:2:end)];
        if (isempty(tExt.visStimOnOff)); tExt.visStimOnOff = [0 0]; end


        % response
        responseMadeIdx = responseRecorded ~= 0;
        timelineVisOnset = indexByTrial(trialStEnTimes, tExt.visStimPeriodOnOff(:,1), tExt.visStimPeriodOnOff(:,1));
        timelineVisOnset(cellfun(@isempty, timelineVisOnset)) = deal({nan});
        timelineAudOnset = indexByTrial(trialStEnTimes, tExt.audStimPeriodOnOff(:,1), tExt.audStimPeriodOnOff(:,1));
        timelineAudOnset(cellfun(@isempty, timelineAudOnset)) = deal({nan});
        timelineStimOnset = min(cell2mat([timelineVisOnset timelineAudOnset]), [],2, 'omitnan');

        missedOnset = isnan(timelineStimOnset) & ~(audAmplitude==0 & visContrast == 0);
        stimFoundIdx = responseMadeIdx & ~(audAmplitude==0 & visContrast == 0);
        stimOnsetIdx = round(timelineStimOnset(stimFoundIdx)*sR);


        %%% some hack to fix the NaN at the first trial which was caused by powers in the universe     
        % Caused by session GB002 on 19 Jan 2024 -- FT
        stimFoundIdx_true = find(stimFoundIdx); 
        stimFoundIdx_true(find(isnan(stimOnsetIdx)))=[];
        stimFoundIdx = zeros(1,numel(stimFoundIdx_true)); stimFoundIdx(stimFoundIdx_true)=1;
        stimFoundIdx = logical(stimFoundIdx);
        stimOnsetIdx = stimOnsetIdx(~isnan(stimOnsetIdx)); 
        
        %%%%%%%%%%%%%%%%
        stimEndIdx = min([stimOnsetIdx+1.5*sR trialStEnTimes(stimFoundIdx,2)*sR],[],2);
        stimEndIdx = stimEndIdx-stimOnsetIdx;
        if any(missedOnset)
            if sum(missedOnset) >0.25*length(missedOnset)
                error('Cannot find expected stimulus onset over 25% of stimulus onsets are missing???');
            else
                warning('Cannot find expected stimulus onset for some trials. Will process identified ones');
            end
        end
        if isempty(stimOnsetIdx)
            warning('Looks like the mouse did not make a single choice?!');
        end

    catch me
        msgText = getReport(me);
        error(msgText)
    end


    %% LASER RELATED STUFF that gets whether there was a laser and some timings related to the laser


    % supposed laser Onset times from block. This includes the fake trials,
    % and basically should be equal to the end of the quiescent period. 
    if isfield(block.events, 'laserPeriodStartTimes')
    blockLaserStartTimes =  preproc.align.event2Timeline(block.events.laserPeriodStartTimes(eIdx), ...
        alignmentBlock.originTimes,alignmentBlock.timelineTimes)';
    else
        blockLaserStartTimes = NaN(numel(is_blankTrial),1); 
    end

    if isfield(e, 'is_laserOnValues') && any(e.is_laserOnValues>0)
        disp('opto data...')
        dat = csv.loadData(block.expInfo, 'dataType', {{'opto'}});

        if any("dataoptoLog" == string(dat.Properties.VariableNames))
            opto = dat.dataoptoLog{1,1}; 
            optoLogExists = true; 
        else
            warning('optoLog does not exist ... opto data will only rely on block.')
            optoLogExists = false;
        end
        is_laser_On_block = e.is_laserOnValues(eIdx); 
        laserPos = NaN(1,numel(is_laser_On_block));

        if isfield(e,'laser_power1Values') && optoLogExists % the Controller way of extracting the data          
         % sometimes there is some issue and we miss issuing a waveform
            is_laser_On_optoLog = opto.is_laserOn; 

    	    if sum(is_laser_On_block)~=sum(is_laser_On_optoLog)
                is_laser_On = zeros(1,numel(is_laser_On_block)); 
                for i=1:numel(is_laser_On_block)
                    trialNum_idx = opto.trialNum==i; 
                    if sum(trialNum_idx)==1
                    is_laser_On(i) = is_laser_On_optoLog(opto.trialNum==i);
                    end 
                end
                is_laser_On = logical(is_laser_On);                
            else
                is_laser_On = is_laser_On_block; 
            end 
    
            % also saving out power and other variables 
            power_laser1 = e.laser_power1Values(eIdx) .* double(is_laser_On); %
            power_laser2 = e.laser_power2Values(eIdx) .* double(is_laser_On); % 

            is_laser_On = (power_laser1+power_laser2)>0; % that is laser power 0 is issued in a bunch of cases
            
            % location of laser 
            laserPosID = e.laserPosValues(eIdx) .* double(is_laser_On);
            if strcmp(opto.laser1_hemisphere(1),'L'); hemisphere1 = -1; elseif strcmp(opto.laser1_hemisphere(1),'R'); hemisphere1 = 1; end 
            if strcmp(opto.laser2_hemisphere(1),'L'); hemisphere2 = -1; elseif strcmp(opto.laser2_hemisphere(1),'R'); hemisphere2 = 1; end 
            
            laserPos(laserPosID==1) = hemisphere1;
            laserPos(laserPosID==2) = hemisphere2;
            laserPos(laserPosID==3) = 0;   % for bilateral, pretty random... %  % will be also 0 but will 
            
            % laser powers: laser1 will be left,laser 2 will be right from
            % now on, so swap, laserpower if hemishere1 = 1  

            if (hemisphere1==1) && (hemisphere2==-1) 
                old_power1 = power_laser1; 
                old_power2 = power_laser2; 
                power_laser1 = old_power2;
                power_laser2 = old_power1; 
            elseif (hemisphere1==1) && (hemisphere2==1)
                % this is when both lasers were admined to be plugged in to
                % R but only really one was plugged in 
                old_power1 = power_laser1; 
                old_power2 = power_laser2; 
                power_laser1 = old_power2*0;
                power_laser2 = old_power1; 
            elseif (hemisphere1==-1) && (hemisphere2==-1)
                % this is when both lasers were admined to be plugged in to
                % L but only really one was plugged in 
                old_power1 = power_laser1; 
                old_power2 = power_laser2; 
                power_laser1 = old_power2;
                power_laser2 = old_power1*0; 
            end 

            all_laser1_times  = timeproc.getChanEventTime(timeline,'laserOut1');
            all_laser2_times  = timeproc.getChanEventTime(timeline,'laserOut2');        
            all_laser_times = [all_laser1_times;all_laser2_times];
            all_laser_times = sortrows(all_laser_times); 
    
            % as I actually send out the waves together atm there is no reason
            % to detect them separaytely
            all_laser_times(logical([0;(diff(all_laser_times(:,1))<.1)]),:) = [];     
            
             % and sometimes we issue the laserOn at the last trial... that was
             % terminated before it finished ... 
             if (size(all_laser_times,1)-sum(is_laser_On))==1
                 all_laser_times(end,:) = []; 
             end

        elseif ~isfield(e,'laser_power1Values') && optoLogExists % when we did not have power saved in block. i.e. 2022 Dec experiments
            is_laser_On = is_laser_On_block;
            all_laser_times  = timeproc.getChanEventTime(timeline,'laserOut');
            if strcmp(opto.Hemisphere(1),'L'); hemisphere1 = -1; elseif strcmp(opto.Hemisphere(1),'R'); hemisphere1 = 1; end 
            laserPos(is_laser_On) = hemisphere1; 
            power_laser1 = zeros(1,numel(is_laser_On_block));
            power_laser2 = zeros(1,numel(is_laser_On_block));
            if hemisphere1==-1
                power_laser1(is_laser_On) = str2double(opto.LaserPower_mW);
            else
                power_laser2(is_laser_On) = str2double(opto.LaserPower_mW);
            end


        elseif isfield(e,'laser_power1Values') && ~optoLogExists  % when no optoLog is saved but we have powers in the block
            is_laser_On = is_laser_On_block;
            all_laser_times = NaN(sum(is_laser_On),4); 
             % also saving out power and other variables 
            power_laser1 = e.laser_power1Values(eIdx) .* double(is_laser_On); %
            power_laser2 = e.laser_power2Values(eIdx) .* double(is_laser_On); % 
            is_laser_On = (power_laser1+power_laser2)>0; % that is laser power 0 is issued in a bunch of cases            
            % location of laser 
            laserPos = NaN(numel(eIdx),1)';

        end 

        
        laser_times_per_trial = indexByTrial(trialStEnTimes,all_laser_times(:,1)); 
        % the longer ITI is at the end of the trial so most of the time of
        % there is an extra flip, it will happen in that ITI

        % sometimes the laser does not turn on for some reason even if
        % there is a pulse: I think it is queing the previous..? But that
        % should be indicated 

        is_laser_On_trial_indexed  = (~cellfun(@isempty,laser_times_per_trial));
        % check whether any is missed 
        missed_waveforms = sum(is_laser_On'-is_laser_On_trial_indexed);         
        if missed_waveforms>0 && missed_waveforms<5
            fprintf('****WARNING: seems like a few laser Trials did not output a waveform... (if more than 5, experiment will error) \n');
            is_laser_On = is_laser_On_trial_indexed';
        end 


        kept_times = cellfun(@(x) x(1), laser_times_per_trial(is_laser_On));
        [~,idx,~] = intersect(all_laser_times(:,1),kept_times);
        laser_times_trial_indexed = NaN(numel(is_laser_On),4);
        laser_times_trial_indexed(is_laser_On,:)= all_laser_times(idx,:);  

    else
        is_laser_On = NaN(numel(eIdx),1)';
        laser_times_trial_indexed = NaN(numel(is_laser_On),4);
        laserPos = NaN(numel(eIdx),1)';
        power_laser1 = zeros(numel(eIdx),1)'; 
        power_laser2 = zeros(numel(eIdx),1)';
    end
    laser_times_trial_indexed = single(laser_times_trial_indexed);


    %% WHEEL RELATED STUFF 
    wheelDeg = extractWheelDeg(timeline); % in timeline time
    wheelVel = diff([0; wheelDeg])*sR;

    %sumWin = 51; % this is the variable that corresponds to 50ms in the methods
    sumWin = 51;
    if isfield(block.events,'selected_paramsetValues')
        whlDecThr = round(60./block.events.selected_paramsetValues.wheelGain); % usually 20 deg at final stage 
    else
        wg = [block.paramsValues.wheelGain];
        whlDecThr = round(60./wg(1));
    end

    velThresh  = sR*(whlDecThr*0.01)/sumWin;

    % "scan" the velocity traces, with a 50 ms "forward" window (for
    % sumWin=51). In other words, summing the velocity in the upcoming 50
    % ms after each point. Every time wheel moves in the "opposite"
    % direction from the "scan" there is a big penalty and it can't cross
    % threshold. So basically, looking for "continuous" movements in one
    % direction as a "choice"
    posVelScan = conv(wheelVel.*double(wheelVel>0) - double(wheelVel<0)*1e6, [ones(1,sumWin) zeros(1,sumWin-1)]./sumWin, 'same').*(wheelVel~=0);
    negVelScan = conv(wheelVel.*double(wheelVel<0) + double(wheelVel>0)*1e6, [ones(1,sumWin) zeros(1,sumWin-1)]./sumWin, 'same').*(wheelVel~=0);
    
    % detect any periods of movement (summed over 20ms) in any direction
    movingScan = smooth((posVelScan'>=velThresh) + (-1*negVelScan'>=velThresh),21);
    
    falseIdx = (movingScan(stimOnsetIdx(~isnan(stimOnsetIdx)))~=0); %don't want trials when mouse is moving at stim onset
    
    % 

    % I am gonna actually save out the trials that were like this
    movedAtStim = zeros(1,numel(is_blankTrial)); 
    movedAtStim(stimFoundIdx) = falseIdx; 

    %find times when wheel crosses the decision threshold between stim on
    %and stim off times
    choiceCrsIdx = arrayfun(@(x,y) max([nan find(abs(wheelDeg(x:(x+y))-wheelDeg(x))>whlDecThr,1)+x]), stimOnsetIdx, round(stimEndIdx));
    choiceCrsIdx(falseIdx) = nan;
    % Only trials where a choice is detected in this way are "good"
    gdIdx = ~isnan(choiceCrsIdx);

    % Get the timeline time and direction for the "good" choices
    choiceThreshTime = choiceCrsIdx/sR;
    choiceThreshDirection = choiceThreshTime*nan;
    choiceThreshDirection(gdIdx) = sign(wheelDeg(choiceCrsIdx(gdIdx)) - wheelDeg(choiceCrsIdx(gdIdx)-25));
    choiceThreshDirection(gdIdx) = (((choiceThreshDirection(gdIdx)==-1)+1).*(abs(choiceThreshDirection(gdIdx))))';

    % Look for times when the velocity crosses the threshold AFTER a period
    % (equal to sumWin) without any threshold crossings
    tstWin = [zeros(1, sumWin-1), 1];
    velThreshPoints = [(strfind((posVelScan'>=velThresh), tstWin)+sumWin-2) -1*(strfind((-1*negVelScan'>=velThresh), tstWin)+sumWin-2)]';

    % Use these crossings as the movement onsets
    [~, srtIdx] = sort(abs(velThreshPoints));
    moveOnsetIdx = abs(velThreshPoints(srtIdx));
    moveOnsetSign = sign(velThreshPoints(srtIdx))';
    moveOnsetDir = (((moveOnsetSign==-1)+1).*(abs(moveOnsetSign)))';

    % Indexing onsets into trials
    onsetTimDirByTrial = indexByTrial(trialStEnTimes, moveOnsetIdx/sR, [moveOnsetIdx/sR moveOnsetDir]);
    onsetTimDirByTrial(cellfun(@isempty, onsetTimDirByTrial)) = deal({[nan nan]});

    % Indexing onsets into trials where there was a stim (i.e. a "choice")
    onsetTimDirByChoiceTrial = onsetTimDirByTrial(stimFoundIdx);
    onsetTimDirByChoiceTrial(cellfun(@isempty, onsetTimDirByTrial)) = deal({[nan nan]});

    %"firstMoveTimes" are the first onsets occuring after stimOnset. "largeMoveTimes" are the first onsets occuring after stimOnsetIdx that match the
    %sign of the threshold crossing defined earlier. Eliminate any that are longer than 1.5s, as these would be timeouts. Also, remove onsets when the
    %mouse was aready moving at the time of the stimulus onset (impossible to get an accurate movement onset time in this case)
    firstMoveTimeDir = cell2mat(cellfun(@(x,y) x(find(x(:,1)>y,1),:), onsetTimDirByChoiceTrial, num2cell(stimOnsetIdx/sR), 'uni', 0)); 
    choiceInitTimeDir = cellfun(@(x,y) x(find(x(:,1)<y,1,'last'),:), onsetTimDirByChoiceTrial, num2cell(choiceThreshTime(:,1)), 'uni', 0);
    choiceInitTimeDir(cellfun(@isempty, choiceInitTimeDir)) = {[nan nan]};
    choiceInitTimeDir = cell2mat(choiceInitTimeDir);

    %SANITY CHECK
    blockTstValues = responseRecorded(stimFoundIdx);
    if ~isempty(choiceInitTimeDir)
        tstIdx = ~isnan(choiceInitTimeDir(:,2));
        if mean(choiceInitTimeDir(tstIdx,2) == blockTstValues(tstIdx)) < 0.75
            error('Why are most of the movements not in the same direction as the response?!?')
        end
    end

    if isempty(stimOnsetIdx)
        tExt.firstMoveTimeDir = [nan, nan];
        tExt.choiceInitTimeDir = [nan, nan];
        tExt.choiceThreshTimeDir = [nan, nan];
    else
        tExt.firstMoveTimeDir = firstMoveTimeDir;
        tExt.choiceInitTimeDir = choiceInitTimeDir;
        tExt.choiceThreshTimeDir = [choiceThreshTime, choiceThreshDirection];
    end
    tExt.allMovOnsetsTimDir = cell2mat(onsetTimDirByTrial);

    changePoints = strfind(diff([0,wheelDeg'])==0, [1 0]);
    trialStEnIdx = (trialStEnTimes*sR);
    points2Keep = sort([1 changePoints changePoints+1 length(wheelDeg) ceil(trialStEnIdx(:,1))'+1, floor(trialStEnIdx(:,2))'-1]);
    points2Keep(points2Keep > length(wheelDeg)) = [];
    tExt.wheelTraceTimeValue = [timelineTime(points2Keep)' wheelDeg(points2Keep)];

   
    % wheel movment related stuff in relation to the laser 
    if ~all(isnan(is_laser_On),'all') 
        % caclulate when there was a movement already at the time of laser
        % onset 
        tl_laserOnset = laser_times_trial_indexed(:,1);
        laserOnsetIdx_tl  = round(tl_laserOnset*sR); % because of the collorary dischange of the laser, at times in these early recordings the LED came on when it wasn't supposed to

        moveAtlaser_tl = zeros(1,numel(is_blankTrial));
        moveAtlaser_tl(is_laser_On) = (movingScan(laserOnsetIdx_tl(~isnan(laserOnsetIdx_tl)))~=0); %don't want trials when mouse is moving at stim onset  
        % first movement after laser based on timeline, excluing those that
        % moved at the time of laser
        tExt.firstMovePostLaserTimeDir_tl = cell2mat(cellfun(@(x,y) x(find(x(:,1)>y,1),:), onsetTimDirByTrial(is_laser_On & ~moveAtlaser_tl), num2cell(tl_laserOnset(is_laser_On & ~moveAtlaser_tl)), 'uni', 0));
    else
        moveAtlaser_tl = zeros(1,numel(is_blankTrial));
        tExt.firstMovePostLaserTimeDir_tl = [nan, nan];
    end 


   % similar calculations from block (to estimate the end of the quiescent
   % period)
    if isfield(block.events, 'laserPeriodStartTimes') & ~all(isnan(is_laser_On),'all') 
        % I had a set of data 
        laserOnsetIdx_block  = round((blockLaserStartTimes)*sR);
        moveAtlaser_block = (movingScan(laserOnsetIdx_block(~isnan(laserOnsetIdx_block)))~=0); %don't want trials when mouse is moving at stim onset  
        tExt.firstMovePostLaserTimeDir_block = cell2mat(cellfun(@(x,y) x(find(x(:,1)>y,1),:), onsetTimDirByTrial(~moveAtlaser_block), num2cell(blockLaserStartTimes(~moveAtlaser_block)), 'uni', 0));
         
        % deterimine choice movement after laser onset or the equivalent end of quiescent period (where we include trials
        % as "choices" the animal was alreay moving at stimulus onset.
        %find times when wheel crosses the decision threshold between stim on
        %and stim off times        
        
        % take timeline times for when laser on, block times whel laser off
        % (I did't send a pulse but it is okish)
        laserOnset_tlOnBlockOff = laserOnsetIdx_block; 
        laserOnset_tlOnBlockOff(is_laser_On) = laserOnsetIdx_tl(is_laser_On); 
        % keep only those when a response was actually made
        laserOnset_tlOnBlockOff = laserOnset_tlOnBlockOff(responseMadeIdx); 

        % for the end period take stimEndIdx (= end of response period, or whenever the stimulus went off,i.e. when response was made earlier)
        % except when ther are laser only trials/no stimulus. In that case
        % take the offset of the laser. which is 1.89*sR
        % There are also the nogo trials which indeed we don't calculate
        % towards the choices. This is taken care of when calculating stimFoundIndex        
        tl_laserOffset = laser_times_trial_indexed(:,4);
        EndPeriodAfterLaserIdx = min([round(timelineStimOnset(responseMadeIdx)*sR)+1.5*sR trialStEnTimes(responseMadeIdx,2)*sR tl_laserOffset(responseMadeIdx)*sR],[],2);
        
        % get the threshold crossings, similar to above when it was
        % calculated in relation to the stimulus. 
        choiceCrsIdx_l = arrayfun(@(x,y) max([nan find(abs(wheelDeg(x:y)-wheelDeg(x))>whlDecThr,1)+x]), laserOnset_tlOnBlockOff, round(EndPeriodAfterLaserIdx));
        choiceCrsIdx_l(logical(moveAtlaser_block(responseMadeIdx))) = nan;
        % Only trials where a choice is detected in this way are "good"
        gdIdx_l = ~isnan(choiceCrsIdx_l);
    
        % Get the timeline time and direction for the "good" choices
        choiceThreshTime_l = choiceCrsIdx_l/sR;
        choiceThreshDirection_l = choiceThreshTime_l*nan;
        choiceThreshDirection_l(gdIdx_l) = sign(wheelDeg(choiceCrsIdx_l(gdIdx_l)) - wheelDeg(choiceCrsIdx_l(gdIdx_l)-25));
        choiceThreshDirection_l(gdIdx_l) = (((choiceThreshDirection_l(gdIdx_l)==-1)+1).*(abs(choiceThreshDirection_l(gdIdx_l))))';
        tExt.choiceMovePostLaserTimeDir = [choiceThreshTime_l, choiceThreshDirection_l];    
    else
        moveAtlaser_block = zeros(numel(is_blankTrial),1); 
        tExt.firstMovePostLaserTimeDir_block = [nan, nan];
        tExt.choiceMovePostLaserTimeDir = [nan,nan]; 

    end 
    %%

    rawFields = fields(tExt);
    for i = 1:length(rawFields)
        currField = rawFields{i};
        currData = tExt.(currField);
        tExt.(currField) = indexByTrial(trialStEnTimes, currData(:,1), currData);
        emptyIdx = cellfun(@isempty, tExt.(currField));

        if any(strcmp(currField, {'allMovOnsetsTimDir'; 'audStimOnOff'; 'visStimOnOff'; 'rewardTimes';'wheelTraceTimeValue'}))
            if contains(currField, {'OnOff', 'TimeValue', 'TimDir'}, 'IgnoreCase',1)
                nColumns = 2;
            else
                nColumns = 1;
            end
            tExt.(currField)(emptyIdx) = {nan*ones(1,nColumns)};
            tExt.(currField) = cellfun(@single,tExt.(currField), 'uni', 0);
        end
        if any(strcmp(currField, {'audStimPeriodOnOff'; 'visStimPeriodOnOff'; 'laserTTLPeriodOnOff';'firstMovePostLaserTimeDir_tl';'firstMovePostLaserTimeDir_block'; 'firstMoveTimeDir'; 'choiceInitTimeDir'; 'choiceThreshTimeDir';'choiceMovePostLaserTimeDir'}))
            nColumns = max(cellfun(@(x) size(x,2), tExt.(currField)));
            if nColumns == 0; nColumns = size(currData,2); end
            tExt.(currField)(emptyIdx) = deal({nan*ones(1, nColumns)});
            tExt.(currField) = single(cell2mat(tExt.(currField)));
        end
    end
    tExt.rewardTimes(cellfun(@length, tExt.rewardTimes)>1) = {nan};
    tExt.rewardTimes(responseRecorded~=1) = {nan};
    tExt.rewardTimes = cellfun(@double, tExt.rewardTimes);

    %% Populate n with all fields;
    ev.is_blankTrial = is_blankTrial;
    ev.is_visualTrial = is_visualTrial;
    ev.is_auditoryTrial = is_auditoryTrial;
    ev.is_coherentTrial = is_coherentTrial;
    ev.is_conflictTrial = is_conflictTrial;
    ev.is_validTrial = vIdx(:) & ~is_noStimTrial;
    ev.is_noStimTrial = is_noStimTrial; 

    if exist('currentBlockValues', 'var')
        ev.block_currentBlock = currentBlockValues;
    end

    ev.block_trialOn = single(trialStEnTimes(:,1));
    ev.block_trialOff = single(trialStEnTimes(:,2));
    ev.block_stimOn = single(stimStartBlock);
    ev.block_laserStartTimes = single(blockLaserStartTimes); 
    ev.block_firstMovePostLaserOn = tExt.firstMovePostLaserTimeDir_block(:,1);
    ev.block_firstMovePostLaserDir = tExt.firstMovePostLaserTimeDir_block(:,2);
    ev.block_isMovedAtLaser = logical(moveAtlaser_block); 

    ev.timeline_rewardOn = single(tExt.rewardTimes);
    ev.timeline_audOn = cellfun(@(x) x(:,1), tExt.audStimOnOff, 'uni', 0);
    ev.timeline_audOff = cellfun(@(x) x(:,2), tExt.audStimOnOff, 'uni', 0);
    ev.timeline_visOn = cellfun(@(x) x(:,1), tExt.visStimOnOff, 'uni', 0);
    ev.timeline_visOff = cellfun(@(x) x(:,2), tExt.visStimOnOff, 'uni', 0);

    ev.timeline_audPeriodOn = tExt.audStimPeriodOnOff(:,1);
    ev.timeline_audPeriodOff = tExt.audStimPeriodOnOff(:,2);
    ev.timeline_visPeriodOn = tExt.visStimPeriodOnOff(:,1);
    ev.timeline_visPeriodOff = tExt.visStimPeriodOnOff(:,2);
    ev.timeline_firstMoveOn = tExt.firstMoveTimeDir(:,1);
    ev.timeline_firstMoveDir = tExt.firstMoveTimeDir(:,2);
    ev.timeline_choiceMoveOn = tExt.choiceInitTimeDir(:,1);
    ev.timeline_choiceMoveDir = tExt.choiceInitTimeDir(:,2);
    ev.timeline_choiceThreshOn = tExt.choiceThreshTimeDir(:,1);
    ev.timeline_allMoveOn = cellfun(@(x) x(:,1), tExt.allMovOnsetsTimDir, 'uni', 0);
    ev.timeline_allMoveDir  = cellfun(@(x) x(:,2), tExt.allMovOnsetsTimDir, 'uni', 0);
    ev.timeline_wheelTime  = cellfun(@(x) x(:,1), tExt.wheelTraceTimeValue, 'uni', 0);
    ev.timeline_wheelValue  = cellfun(@(x) x(:,2), tExt.wheelTraceTimeValue, 'uni', 0);
    ev.timeline_isMovedAtStim = logical(movedAtStim'); 

    ev.is_laserTrial = is_laser_On';
    ev.timeline_laserOn_rampStart = laser_times_trial_indexed(:,1);
    ev.timeline_laserOn_rampEnd = laser_times_trial_indexed(:,2);
    ev.timeline_laserOff_rampStart = laser_times_trial_indexed(:,3);
    ev.timeline_laserOff_rampEnd = laser_times_trial_indexed(:,4);
    ev.stim_laserPosition = laserPos'; 
    ev.stim_laser1_power = power_laser1';
    ev.stim_laser2_power = power_laser2';
    ev.timeline_firstMovePostLaserOn = tExt.firstMovePostLaserTimeDir_tl(:,1);
    ev.timeline_firstMovePostLaserDir = tExt.firstMovePostLaserTimeDir_tl(:,2);
    ev.timeline_isMovedAtLaser = logical(moveAtlaser_tl'); 
    ev.timeline_choiceThreshPostLaserOn  = tExt.choiceMovePostLaserTimeDir(:,1);
    ev.timeline_choiceMovePostLaserDir  = tExt.choiceMovePostLaserTimeDir(:,2);

    ev.stim_correctResponse = single(correctResponse);
    ev.stim_repeatNum = single(repeatNums);
    ev.stim_audAmplitude = single(audAmplitude);
    ev.stim_audAzimuth = single(audInitialAzimuth);
    ev.stim_visContrast = single(visContrast);
    ev.stim_visAzimuth = single(visInitialAzimuth);
    ev.stim_closedLoop = single(stim_closedLoop>0);

    ev.response_direction = single(responseRecorded);
    ev.response_feedback = single(feedbackValues);

end
