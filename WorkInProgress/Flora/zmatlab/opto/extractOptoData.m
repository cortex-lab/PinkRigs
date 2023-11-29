 % get Opto data;;

function extractOptoData(varargin)  
% wrapper around getTrainingData to specifically filter for certain opto
% conditions. (as n is the hemisphere);
varargin = ['reverse_opto', {0}, varargin];
varargin = ['combDates', {0}, varargin]; % whether to combine dates for each mouse
varargin = ['combMice', {1}, varargin]; % whether to combine mice 
varargin = ['selHemispheres', {[-1 1]}, varargin]; % whether to merge inhibiton of 2 hemispheres or not -- only an option if reverse_opto is True 
varargin = ['selPowers', {0}, varargin]; % whether to merge unhibiton of 2 hemispheres or not -- only an option if reverse_opto is True 
varargin = ['balanceTrials', {1}, varargin];
varargin = ['minN', {600}, varargin];
varargin = ['includeNoGo', {0}, varargin]; % 1 include nogo % >1=only include noGo that is not followed by other nogos, with this window size


params = csv.inputValidation(varargin{:});
extracted = plts.behaviour.getTrainingData(varargin{:});

unique_subjects = unique(extracted.subject);
subject_indices=1:numel(unique_subjects);

for i=1:numel(extracted.subject)
    % calculate things related to power 
    nTrials = numel(extracted.data{i, 1}.is_blankTrial);
    extracted.data{i, 1}.laser_power = extracted.data{i, 1}.stim_laser1_power + extracted.data{i, 1}.stim_laser2_power;
    laserPositions = extracted.data{i, 1}.stim_laserPosition; % this might need to be corrected for which laser is how much power exactly
    extracted.data{i, 1}.laser_power_signed = extracted.data{i, 1}.stim_laserPosition.*extracted.data{i, 1}.laser_power ;
    % add the session a session ID
    extracted.data{i, 1}.sessionID = ones(nTrials,1)*i;
    % get mouse ID
    extracted.data{i, 1}.subjectID  = ones(nTrials,1)*subject_indices(strcmp(unique_subjects,extracted.subject{i}));

    % calculate whether the nogo is in a block or not
    
    if params.includeNoGo{1}>0
        window_size = params.includeNoGo{1};     
        isnoGo_block = conv((extracted.data{i, 1}.response_direction==0),ones(window_size,1));
        extracted.data{i, 1}.isnoGo_block = (isnoGo_block(1:(end-(window_size-1)))>=window_size);
    end 
    % assert whether that is actully a non-opto session
    laserPowers = extracted.data{i, 1}.laser_power; laserPositions = extracted.data{i, 1}.stim_laserPosition;
    usedPowers = unique(laserPowers);
    usedPositions = unique(laserPositions(~isnan(laserPositions)));

    is_no_laser_session(i) = (sum(usedPowers)==0);
    is_power_test_session(i) = numel(usedPositions)~=1; % some sessions play high fraction visual trials

    % things related to what side was stimulated;     
    if params.reverse_opto{1} && (sum(usedPositions)<0)
        % flip if the opto was on the left side, apparently I need to flip
        % all the params! 
       extracted.data{i, 1}.stim_audAzimuth = extracted.data{i, 1}.stim_audAzimuth * -1 ;
       extracted.data{i, 1}.stim_visAzimuth = extracted.data{i, 1}.stim_visAzimuth * -1 ;
       extracted.data{i, 1}.stim_visDiff = extracted.data{i, 1}.stim_visDiff * -1 ;
       extracted.data{i, 1}.stim_audDiff = extracted.data{i, 1}.stim_audDiff * -1 ;
       extracted.data{i, 1}.timeline_choiceMoveDir = ((extracted.data{i, 1}.timeline_choiceMoveDir-1.5)*-1)+1.5;
       swapped = ((extracted.data{i, 1}.response_direction-1.5)*-1)+1.5; % maybe it is response direction we are talking
       swapped(swapped==3) = 0; 
       extracted.data{i, 1}.response_direction = swapped; 
       
%        if strcmp(params.selHemispheres{i},'comb')
%         usedPositions = sort(abs(usedPositions)); % flip the posotion used as well
%        end
    end 
    extracted.usedPositions{i,1} = usedPositions;
end 
% throw away no laser sessions that ought to not be in opto analysis 

extracted.validSubjects = num2cell(extracted.validSubjects);
extracted  = filterDataStruct(extracted,(~is_no_laser_session' & ~is_power_test_session'));

% concatenate data into one giant structure
optoExtracted = concatenateEvents(extracted.data); 
% names = fieldnames(extracted.data{1});
% for k=1:numel(names)
%     a = {horzcat(extracted.data{:}).(names{k})};
%     optoExtracted.(names{k}) = vertcat(a{:});   
% end



% power selection
lowPower = cell2mat(arrayfun(@(x) [x, min(optoExtracted.laser_power(optoExtracted.subjectID==x))], unique(optoExtracted.subjectID), 'uni', 0));
highPower = cell2mat(arrayfun(@(x) [x, max(optoExtracted.laser_power(optoExtracted.subjectID==x))], unique(optoExtracted.subjectID), 'uni', 0));

% select for minimum nimber for each trial 
minN = params.minN{1};
optoExtracted.stim_laserPosition(isnan(optoExtracted.stim_laserPosition)) = -1000; 
powerConditions = unique([optoExtracted.subjectID,optoExtracted.stim_laserPosition,optoExtracted.laser_power], 'rows');
nTrials = arrayfun(@(x,y,z) sum(ismember([optoExtracted.subjectID,optoExtracted.stim_laserPosition, optoExtracted.laser_power],[x,y,z],'rows')),powerConditions(:,1),powerConditions(:,2),powerConditions(:,3));
[~,conditionID] = ismember([optoExtracted.subjectID,optoExtracted.stim_laserPosition, optoExtracted.laser_power],powerConditions,'rows');
optoExtracted = filterStructRows(optoExtracted, ismember(conditionID,find(nTrials>minN)));
optoExtracted.stim_laserPosition(optoExtracted.stim_laserPosition==-1000) = nan; 


if ~ischar(params.selHemispheres{1})
    optoExtracted = filterStructRows(optoExtracted, isnan(optoExtracted.stim_laserPosition) | ...
        ismember(optoExtracted.stim_laserPosition, params.selHemispheres{1}));
elseif strcmpischar(params.selHemispheres{1}) 
    disp('the current situation ls...')
end

if ischar(params.selPowers{1}) && strcmp(params.selPowers{1}, 'high')
    optoExtracted = filterStructRows(optoExtracted, isnan(optoExtracted.stim_laserPosition) | ...
        ismember([optoExtracted.subjectID, optoExtracted.laser_power], highPower, 'rows'));
elseif ischar(params.selPowers{1}) && strcmp(params.selPowers{1}, 'low')
    optoExtracted = filterStructRows(optoExtracted, isnan(optoExtracted.stim_laserPosition) | ...
        ismember([optoExtracted.subjectID, optoExtracted.laser_power], lowPower, 'rows'));
else
    optoExtracted = filterStructRows(optoExtracted, ismember(optoExtracted.laser_power, [0,params.selPowers{1}]));
end

% thow away control data from sessions when there is no power keptoptoExtracted = filterStructRows(optoExtracted, ismember(optoExtracted.sessionID, optoExtracted.sessionID(optoExtracted.laser_power~=0)));

% also throw away invalid trials that don't go into the fitting

if params.includeNoGo{1}==1
    optoExtracted = filterStructRows(optoExtracted, (optoExtracted.is_validTrial & ...
        abs(optoExtracted.stim_audAzimuth)~=30));
elseif params.includeNoGo{1}>1
    optoExtracted =filterStructRows(optoExtracted, (optoExtracted.is_validTrial & ...
        ~optoExtracted.isnoGo_block  & abs(optoExtracted.stim_audAzimuth)~=30)); 
else 
   optoExtracted = filterStructRows(optoExtracted, (optoExtracted.is_validTrial & ...
        optoExtracted.response_direction & abs(optoExtracted.stim_audAzimuth)~=30));
end 

% save the extracted dataset as a parquet
savepath = 'C:\Users\Flora\Documents\Processed data\Audiovisual\opto';
stub = 'full_set';
saveONEFormat(optoExtracted,savepath,'_opto_trials','table','pqt',stub)   ;
% save some metadata about the extraction
mySubjects.name=unique_subjects; mySubjects.IDs = subject_indices';
saveONEFormat(mySubjects,savepath,'_opto_trials','mouseIDs','pqt',stub);
save([savepath '\sessionParams.mat'],'params')

end