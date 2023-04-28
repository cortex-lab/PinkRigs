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
varargin = ['combHemispheres', {0}, varargin]; % whether to merge unhibiton of 2 hemispheres or not -- only an option if reverse_opto is True 


params = csv.inputValidation(varargin{:});
extracted = plts.behaviour.getTrainingData(varargin{:});

% extract some more information that aplies the opto data
for i=1:numel(extracted.subject)
    % calculate things related to power 
    laserPowers = extracted.data{i, 1}.stim_laser1_power + extracted.data{i, 1}.stim_laser2_power;
    usedPowers = unique(laserPowers);
    extracted.userPowers{i,1} = usedPowers;  
    is_no_laser_session(i) = (sum(usedPowers)==0);
    is_power_test_session(i) = numel(usedPowers)>2;

    % things related to what side was stimulated; 
    usedPositions = unique(extracted.data{i, 1}.stim_laserPosition); 
    
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
       
       if params.combHemispheres{i}
        usedPositions = sort(abs(usedPositions)); % flip the posotion used as well
       end
    end 
    extracted.usedPositions{i,1} = usedPositions;
end 
% throw away no laser sessions that ought to not be in opto analysis %
extracted.validSubjects = num2cell(extracted.validSubjects);
fn = fieldnames(extracted);
for k=1:numel(fn)
    d = extracted.(fn{k});
    extracted.(fn{k}) = {d{~is_no_laser_session' & ~is_power_test_session'}}';         
end
% by now we have thrown away sessions with 0 or >1 powers 
% and in practice we are not using more than one position 
% merge dates of the same powers and position set if requested 
unique_subjects = unique(extracted.subject);
subject_indices=1:numel(unique_subjects);
for i=1:numel(extracted.subject)
    optoParams(i,1) = extracted.userPowers{i,1}(find(extracted.userPowers{i,1}~=0));
    optoParams(i,2) = extracted.usedPositions{i,1}(find(extracted.usedPositions{i,1}~=0));
    if params.combMice{1}==1
        optoParams(i,3) = 1;
    else 
        optoParams(i,3) = subject_indices(strcmp(unique_subjects,extracted.subject{i}));
    end
end  
% get unique optoParamsSets and sort extracted based on those params
[optoParamSets,~,uniMode] = unique(optoParams,'rows');

for i=1:size(optoParamSets,1)
    optoExtracted.subject{i,1} = unique_subjects{optoParamSets(i,3)};
    % all the data events
    % nExp,validSubjects,etc.
    modeIdx = uniMode==i ; 
    optoExtracted.nExp{i,1} = sum(uniMode==i); 
    dataEvents = extracted.data(modeIdx);
    names = fieldnames(dataEvents{1});
    for k=1:numel(names)
        a = {horzcat(dataEvents{:}).(names{k})};
        optoExtracted.data{i,1}.(names{k}) = vertcat(a{:});
    end
    optoExtracted.optoParams{i,1} = optoParamSets(i,:); 
end

% separate control data and opto data 




end