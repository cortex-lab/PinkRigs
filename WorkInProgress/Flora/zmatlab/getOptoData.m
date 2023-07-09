% get Opto data

function optoExtracted  = getOptoData(varargin)  
% wrapper around getTrainingData to specifically filter for certain opto
% conditions. (as n is the hemisphere);
varargin = ['sepPlots', {1}, varargin];
varargin = ['expDef', {'t'}, varargin];
% parameters specific for this opto script
varargin = ['power', {10}, varargin];
varargin = ['location', {1}, varargin];
varargin = ['reverse_opto', {0}, varargin];
varargin = ['combDates', {0}, varargin]; % whether to combine dates for each mouse
varargin = ['combMice', {1}, varargin]; % whether to combine mice 
varargin = ['selHemispheres', {[-1 1]}, varargin]; % whether to merge inhibiton of 2 hemispheres or not -- only an option if reverse_opto is True 
varargin = ['selPowers', {0}, varargin]; % whether to merge unhibiton of 2 hemispheres or not -- only an option if reverse_opto is True 


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


    % assert whether that is actully a non-opto session
    laserPowers = extracted.data{i, 1}.laser_power; laserPositions = extracted.data{i, 1}.stim_laserPosition;
    usedPowers = unique(laserPowers);
    usedPositions = unique(laserPositions(~isnan(laserPositions)));

    is_no_laser_session(i) = (sum(usedPowers)==0);
    is_power_test_session(i) = numel(usedPositions)~=1; % some sessions play high fraction visual trials

    % things related to what side was stimulated;     
    if params.reverse_opto{i} && (sum(usedPositions)<0)
        % flip if the opto was on the left side, apparently I need to flip
        % all the params! 
       extracted.data{i, 1}.stim_audAzimuth = extracted.data{i, 1}.stim_audAzimuth * -1 ;
       extracted.data{i, 1}.stim_visAzimuth = extracted.data{i, 1}.stim_visAzimuth * -1 ;
       extracted.data{i, 1}.stim_visDiff = extracted.data{i, 1}.stim_visDiff * -1 ;
       extracted.data{i, 1}.stim_audDiff = extracted.data{i, 1}.stim_audDiff * -1 ;
       extracted.data{i, 1}.timeline_choiceMoveDir = ((extracted.data{i, 1}.timeline_choiceMoveDir-1.5)*-1)+1.5;
       swapped = ((extracted.data{i, 1}.response_direction-1.5)*-1)+1.5;
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
fn = fieldnames(extracted);
for k=1:numel(fn)
    d = extracted.(fn{k});
    extracted.(fn{k}) = {d{~is_no_laser_session' & ~is_power_test_session'}}';         
end
% concatenate data into one giant structure
names = fieldnames(extracted.data{1});
for k=1:numel(names)
    a = {horzcat(extracted.data{:}).(names{k})};
    b.(names{k}) = vertcat(a{:});   
end

if ~ischar(params.selPowers{1})
    b = filterStructRows(b, isnan(b.stim_laserPosition) | ...
        ismember(b.stim_laserPosition, params.selHemispheres{1}));
end
% power selection
lowPower = cell2mat(arrayfun(@(x) [x, min(b.laser_power(b.subjectID==x))], unique(b.subjectID), 'uni', 0));
highPower = cell2mat(arrayfun(@(x) [x, max(b.laser_power(b.subjectID==x))], unique(b.subjectID), 'uni', 0));
minN = 100;
powerConditions = unique([b.subjectID, b.laser_power], 'rows');
ismember([b.subjectID, b.laser_power],paramSet,'rows');

if ischar(params.selPowers{1}) && strcmp(params.selPowers{1}, 'high')
    b = filterStructRows(b, isnan(b.stim_laserPosition) | ...
        ismember([b.subjectID, b.laser_power], highPower, 'rows'));
elseif ischar(params.selPowers{1}) && strcmp(params.selPowers{1}, 'low')
    b = filterStructRows(b, isnan(b.stim_laserPosition) | ...
        ismember([b.subjectID, b.laser_power], lowPower, 'rows'));
else
    b = filterStructRows(b, ismember(b.laser_power, params.selPowers{1}));
end
b = filterStructRows(b, ismember(b.sessionID, b.sessionID(b.laser_power~=0)));

% get equal number of trials arcoss a bunch of conditions
paramSet = unique([b.subjectID, b.laser_power], 'rows');
[~,conditionID] = ismember([b.subjectID, b.laser_power],paramSet,'rows');
b = filterStructRows(b, makeFreqUniform(conditionID)>0);


% then rearange

% nTrials = numel(b.is_blankTrial);
% posValues = b.stim_laserPosition;
% if strcmp(params.selHemispheres{1},'comb')
%     keepIdx = ones(nTrials,1); 
% elseif strcmp(params.selHemispheres{1},'uni')
%     keepIdx = find(posValues~=0);
% elseif strcmp(params.selHemispheres{1},'bi')
%     keepIdx = find(posValues==0);
% elseif strcmp(params.selHemispheres{1},'uni1')
%     % select one random hemisphere per mouse
%     mouseIDs = unique(b.subjectID);
%     keepIdx = find(abs(posValues)==1);
%     [hemisets,~,selIdx] = unique([b.subjectID(keepIdx),posValues(keepIdx)],'rows');
%     whichtoKeep=[];
%     for m=1:numel(mouseIDs)
%         sIdx = find(hemisets(:,1)==mouseIDs(m));
%         whichtoKeep = [whichtoKeep;keepIdx((selIdx==sIdx(randperm(numel(sIdx),1))))]; 
%     end
%     keepIdx = whichtoKeep; 
% else
%     % when input for hemisphere is numeric
%     hemiID = str2num(params.selHemispheres{1});
%     keepIdx = find(posValues==hemiID);
% end
% conditionID = b.subjectID.*((double(b.laser_power~=0)*2)-1);
% conditionID(conditionID>0) = conditionID.*((double(b.laser_power~=0)*2)-1);
% b = filterStructRows(b, makeFreqUniform(conditionID));



% keepIdx = intersect(keepIdx,keepIdx_power);
% select sessions that we want to keep based on that criteria
% selected_sessions = unique(b.sessionID(keepIdx));
% keepIdx = (ismember(b.sessionID,selected_sessions));
% newBlock = filterStructRows(b, keepIdx); 

% throw away powers we don't actually want to keep from those
% keepIdx = ismember(newBlock.laser_power,[params.selPowers{1},0]);
% newBlock = filterStructRows(newBlock, keepIdx); 

% re_contstruct structures for each condition we are testing
% mysubjects = unique(newBlock.subjectID);
% 
% %  combine the mice 
% %  subselect certain no of trials
% %  get the highest power
% for s=1:numel(mysubjects)
%     keepIdx = b.subjectID==mysubjects(s);
%     optoExtracted.data{s,1} = filterStructRows(b, keepIdx); 
%     optoExtracted.subject{s,1} = unique_subjects{subject_indices(mysubjects(s))};
% end
% 
% disp('ffs')
% % work though all the conditions that one might calll
% % loop though each block and add information for every trial that includes 
% % mouse name % total_power
% 
% 
% 
% unique_subjects = unique(extracted.subject);
% subject_indices=1:numel(unique_subjects);
% for i=1:numel(extracted.subject)
%     optoParams(i,1) = subject_indices(strcmp(unique_subjects,extracted.subject{i}));
%     optoParams(i,2) = extracted.usedPositions{i,1};
%     % still needs some parameter about power -- otherwise we wll 
% 
% end  
% 
% % get unique optoParamsSets and sort extracted based on those params i.e.
% % position on session and mice
% [optoParamSets,~,uniMode] = unique(optoParams,'rows');
% 
% nSets = size(optoParamSets,1); 
% ct=1; 
% for i=1:nSets
%     mySubject = unique_subjects{optoParamSets(i,1)};
%     % all the data events
%     % nExp,validSubjects,etc.
%     modeIdx = uniMode==i ; 
%     dataEvents = extracted.data(modeIdx);
%     % merge fields
%     names = fieldnames(dataEvents{1});
%     for k=1:numel(names)
%         a = {horzcat(dataEvents{:}).(names{k})};
%         currBlock.(names{k}) = vertcat(a{:});   
%     end
%     % seprate each power 
%     currBlock.laserPower = currBlock.stim_laser1_power + currBlock.stim_laser2_power; 
%     powers = unique(currBlock.laserPower); 
%     n_powers = numel(powers); 
%     for p=1:n_powers
%         keepIdx = currBlock.response_direction & currBlock.is_validTrial & abs(currBlock.stim_audAzimuth)~=30 & (currBlock.laserPower==powers(p));
%         filteredCurrBlock = filterStructRows(currBlock, keepIdx);  
%         intermediateDat.subject{ct,1} = mySubject; 
%         intermediateDat.nExp{ct,1} = sum(uniMode==i); 
%         intermediateDat.data{ct,1} = filteredCurrBlock; 
%         intermediateDat.subjectIdx{ct,1} = optoParamSets(i,1); 
%         intermediateDat.hemisphere{ct,1} = optoParamSets(i,2); 
%         intermediateDat.power{ct,1} = powers(p); 
%         intermediateDat.nTrials{ct,1} = numel(filteredCurrBlock.is_blankTrial);
%         ct=ct+1;
% 
% 
%     end 
% end
% 
% % now merge/throw away data that is not requested
% %
% nTrials = [intermediateDat.nTrials{:}];
% combParams = [[intermediateDat.subjectIdx{:}];[intermediateDat.hemisphere{:}];[intermediateDat.power{:}]]';
% 
%  
% % identify sets to keep based on inputs
% if strcmp(params.selHemispheres{1},'comb')
%     combParams(:,2) = 1; 
% elseif strcmp(params.selHemispheres{1},'uni')
%     keepIdx = find(combParams(:,2)~=0);
% elseif strcmp(params.selHemispheres{1},'bi')
%     keepIdx = find(combParams(:,2)==0);
% elseif strcmp(params.selHemispheres{1},'uni1')
%     % select one random hemisphere per mouse
%     mouseIDs = unique(combParams(:,1));
%     keepIdx = find(combParams(:,2)~=0);
%     [hemisets,~,selIdx] = unique(combParams(keepIdx,1:2),'rows');
%     whichtoKeep=[];
%     for m=1:numel(mouseIDs)
%         sIdx = find(hemisets(:,1)==m);
%         whichtoKeep = [whichtoKeep;keepIdx((selIdx==sIdx(randperm(numel(sIdx),1))))]; 
%     end
%     keepIdx = whichtoKeep; 
% else
%     % when input for hemisphere is numeric
%     hemiID = str2num(params.selHemispheres{1});
%     keepIdx = find(combParams(:,2)==hemiID);
% end
% 
% throwIdx = 1:numel(nTrials); throwIdx(keepIdx) = [];
% combParams(throwIdx,:)  = -1;
% 
% if strcmp(params.selPowers{1},'comb')
%     disp('tbi')
% else
%     disp('tbi')
% end 
% 
% if params.combMice{1}
%     combParams(:,1) = 1; 
% end
% % possible parameters to employ 
% [combParamSets,~,uniMode] = unique(combParams,'rows');
% 
% % 
% ct = 1; 
% for c=1:size(combParamSets,1)
%     modeIdx = uniMode==c; 
%     dataEvents = intermediateDat.data(modeIdx);
%     minTrials = min(nTrials(modeIdx)); 
%     
%     names = fieldnames(dataEvents{1});
%     for k=1:numel(names)
%         a = {horzcat(dataEvents{:}).(names{k})}; 
%         for s=1:numel(a)
%             a{s}= a{s}(1:minTrials); 
%         end 
%         currBlock.(names{k}) = vertcat(a{:});   
%     end
% 
%     if combParamSets(c,3)~=-1 
%         % determining combset power as -1 is flagged as throw away
%         % conditions
% %         intermediateDat.subject{ct,1} = mySubject; 
% %         intermediateDat.nExp{ct,1} = sum(uniMode==i); 
% %         intermediateDat.data{ct,1} = filteredCurrBlock; 
%         optoExtracted.subjectIdx{ct,1} = combParamSets(c,1); 
%         optoExtracted.hemisphere{ct,1} = combParamSets(c,2); 
%         optoExtracted.power{ct,1} = combParamSets(c,3); 
%         optoExtracted.data{ct,1} = currBlock; 
%         ct = ct+1; 
%     end 
% end 
% 
% disp('lalala')








end