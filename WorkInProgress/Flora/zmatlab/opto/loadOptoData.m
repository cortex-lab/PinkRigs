function [opto] = loadOptoData(varargin) 

p = inputParser;
addOptional(p,'balanceTrials',1);
addOptional(p,'sepMice',1);
addOptional(p,'sepHemispheres', 1);
addOptional(p,'sepPowers', 1);
addOptional(p,'sepChoices',0)
addOptional(p,'reExtract', 0);
addOptional(p,'addFakeTrial', 0);
addOptional(p,'sepDiffPowers', 0); % trying to separate the bilateral recordings when the applied power is imbalanced


parse(p, varargin{:});

params = p.Results; 

if params.reExtract
    extractionParams = optoParams; 
    extractOptoData(extractionParams);
end

savepath = 'C:\Users\Flora\Documents\Processed data\Audiovisual\opto';
evPQTPath = [savepath '\_opto_trials.table.full_set.pqt'];
if exist(evPQTPath, 'file')
    events = table2struct(parquetread(evPQTPath),"ToScalar",1);
    mouseIDs = table2struct(parquetread([savepath '\_opto_trials.mouseIDs_largeData.full_set.pqt']),"ToScalar",1);
end


events.diff_power = events.stim_laser1_power-events.stim_laser2_power; 

% get equal number of trials arcoss a bunch of conditions
if params.balanceTrials
    events.stim_laserPosition(isnan(events.stim_laserPosition)) = -1000; 
    paramSet = unique([events.subjectID,events.stim_laserPosition, events.laser_power], 'rows');
    [~,conditionID] = ismember([events.subjectID,events.stim_laserPosition, events.laser_power],paramSet,'rows');
    events = filterStructRows(events, makeFreqUniform(conditionID)>0);
    % reconvert the pos..
    events.stim_laserPosition(events.stim_laserPosition==-1000) = nan; 
end

if ~params.sepMice
    events.subjectID_ = events.subjectID; 
    events.subjectID = ones(numel(events.subjectID),1);
end

if ~params.sepHemispheres
    events.stim_laserPosition(~isnan(events.stim_laserPosition)) = 1;
end 

if ~params.sepPowers
    events.laser_power(~isnan(events.stim_laserPosition)) = 1; 
end 

if ~params.sepDiffPowers
    events.diff_power(~isnan(events.stim_laserPosition)) = 0 ; 
end 

subjects = events.subjectID; 
hemispheres = events.stim_laserPosition;
powers = events.laser_power;
diffpowers = events.diff_power; 

paramSet = unique([subjects(~isnan(hemispheres)), ...
    hemispheres(~isnan(hemispheres)) powers(~isnan(hemispheres)) diffpowers(~isnan(hemispheres))], 'rows');


for i=1:size(paramSet,1)
   event_subset =filterStructRows(events,events.subjectID==paramSet(i,1) & ...
       ((events.stim_laserPosition==paramSet(i,2))) & ...
       ((events.laser_power == paramSet(i,3))) & ...
       ((events.diff_power == paramSet(i,4))) );
   
   subset_sessions = unique(event_subset.sessionID); % so that controls are from the same sessions
   sessIDs{i}=subset_sessions;
   event_controls = filterStructRows(events,...
       (ismember(events.sessionID,subset_sessions) & ...
       isnan(events.stim_laserPosition))); 

   opto.data{i,1} = concatenateEvents({event_subset,event_controls}); 
   opto.subject{i,1}=mouseIDs.name(paramSet(i,1));
   opto.hemisphere{i,1}=paramSet(i,2); 
   opto.power{i,1}=paramSet(i,3);
   opto.diff_power{i,1} = paramSet(i,4);
end 

disp(sprintf('*** COMPLETED. Extracted %.0f datasets from %.0f subjects. ***',numel(opto.subject),numel(unique([opto.subject{:}]))))


end 
