function expList = loadData(varargin)
%% Load ev and/or spk data from particular mice and/or dates
% NOTE: This function uses csv.inputValidate to parse inputs

% Add default values for extra inputs:
% dataType (default='events'): string or cell of strings to indicate which
% data types to load. 
% REMEMBER: use double cells for each input if you want more than one dataType.
% For example: (subject='AV015', dataType = {{'probe1'; 'events'}})
%   'blk' or 'block to load raw block files (output = dataBlock)
%   'tim' or 'timeline' to load timeline files (output = dataTimeline)
%   'ev' or 'events' to load trial events (output = dataEvents)
%   'eventsFull' to load all (including large) trial events (output = dataEvents)
%   'probe' load spike information (can specify probe number) (output = dataSpikes)
%   'all' to load all data

% object (default='all'): string or cell of strings to indicate which
% objects to load for each dataType. At the moment, this is only relevant 
% for "probe" dataTypes but will likely be relevant for others later. 
% Examples below:
%   'spikes' to load only spike data
%   'templates' to load only template data

% attribute (default='all'): string or cell of strings to indicate which
% attributes to load for each object. At the moment, this is only relevant 
% for "probe" dataTypes but will likely be relevant for others later. 
%   'spikes' to load only spike data
%   'templates' to load only template data

% NOTE: loadtag continuous. i.e. 'timblk' loads timeline and block
varargin = ['dataType', {'events'}, varargin];
varargin = ['object', {'all'}, varargin];
varargin = ['attribute', {'all'}, varargin];
varargin = ['verbose', {1}, varargin];
varargin = [varargin, 'invariantParams', {{'dataType'; 'object'; 'attribute'}}];
params = csv.inputValidation(varargin{:});
verbose = params.verbose{1};

%% This section deals with the requested inputs to make sure they are valid
dataTypes = unnestCell(params.dataType{1});
if any(contains(dataTypes, 'all'))&& length(dataTypes)~=1
    error('If requesting "all" dataTypes, then length of dataTypes should be "1"')
end

objects = cellfun(@(x) unnestCell(x), unnestCell(params.object{1},0), 'uni', 0);
attributes = cellfun(@(x) unnestCell(x), unnestCell(params.attribute{1},0), 'uni', 0);
if length(objects) == 1
    objects = repmat(objects, length(dataTypes),1);
elseif  length(objects) ~= length(dataTypes)
    error('Length of objects must be equal to "1" or length of dataTypes');
end
objects = cellfun(@(x) strjoin(x, ','), objects, 'uni', 0);


if length(attributes) == 1
    attributes = repmat(attributes, length(objects),1);
elseif  length(attributes) ~= length(objects)
    error('Length of attributes must be equal to "1" or length of objects');
end
attributes = cellfun(@(x) strjoin(x, ','), attributes, 'uni', 0);
params = rmfield(params, {'dataType'; 'object'; 'attribute';'verbose'});

%% 
expList = csv.queryExp(params);

% Add new fields for loaded data to the expList
newFields = {'dataBlock'; 'dataEvents'; 'dataSpikes'; 'dataTimeline'};
for i = 1:length(newFields)
    if any(strcmp(expList.Properties.VariableNames, newFields{i})); continue; end
    expList.(newFields{i}) = cell(size(expList,1),1);
end

% End function if there are no experiments matching search criteria
if isempty(expList)
    warning('No sessions matched params so will not load any data'); 
    return
end

% Indicate which data will be loaded
if verbose
cellfun(@(x,y,z) fprintf('***Will load "%s" with objects=(%s) and attributes=(%s)\n', x, y, z), ...
        dataTypes, objects, attributes);
end

% Loop over each line of the expList and load the requested data
for i=1:height(expList)
    % Clear any existing data and get current exp details
    clear dataBlock dataEvents dataSpikes dataTimeline

    currExp = expList(i,:);
    ONEPath = [currExp.expFolder{1} '\ONE_preproc\'];
    ONENames = dir([currExp.expFolder{1} '\ONE_preproc\']);
    ONENames = ONENames(cellfun(@(x) ~strcmp(x(1),'.'),{ONENames.name}'));
    ONENames = {ONENames.name}';
    expPathStub = strcat(currExp.expDate, {'_'}, currExp.expNum, {'_'}, currExp.subject);
    
    %% Load dataSpikes if requested
    if any(contains(dataTypes, {'probe', 'all'}, 'IgnoreCase',1))
        dataIdx = contains(ONENames, 'probe');
        for j = find(dataIdx)'
            if isempty(j); continue; end

            %If requested ONEFolder "probe1", skip other probes
            if ~contains({ONENames{j}, 'all'}, dataTypes)
                continue
            end
            %If requested object "probe" load all objects in probe folder
            spikeStatus = cell2mat(cellfun(@str2num, split(currExp.extractSpikes{1}, ','), 'uni', 0));
            if all(isnan(spikeStatus)) || ~(spikeStatus(str2double(ONENames{j}(end))+1) == 1)
                continue;
            end

            objPath = fullfile(ONEPath, ONENames{j});
            idx = contains(dataTypes, {'probe', 'all'});
            expList.dataSpikes{i}.(ONENames{j}) = ...
                loadAttributes(objects(idx), attributes(idx), objPath);
        end
    end
    
    %% Load dataEvents if requested
    evCheck =  {'ev'; 'events'; 'eventsFull'; 'all';};
    if any(contains(dataTypes, evCheck, 'IgnoreCase',1))
        evPQTPath = cell2mat([ONEPath 'events\_av_trials.table.' expPathStub '.pqt' ]);
        if exist(evPQTPath, 'file')
            expList.dataEvents{i} = table2struct(parquetread(evPQTPath),"ToScalar",1);
        end
        evPQTPath = strrep(evPQTPath, '.table', '.table_largeData');
        if any(contains(dataTypes, 'eventsFull', 'IgnoreCase',1)) && exist(evPQTPath, 'file')
            largeEvents = table2struct(parquetread(evPQTPath),"ToScalar",1);
            expList.dataEvents{i} = catStructs(expList.dataEvents{i},largeEvents);
        end
    end

    %% Load dataBlock if requested
    if any(contains(dataTypes, {'blk', 'block'}))
        blockPath = cell2mat([currExp.expFolder '\' expPathStub '_block.mat']);
        if exist(blockPath, 'file')
            blk = load(blockPath, 'block');
            if exist('blk', 'var')
                expList.dataBlock{i} = blk.block;
            end
        end
    end

    %% Load timeline data if requested
    if any(contains(dataTypes, {'tim'; 'timeline'}))
        timelinePath = cell2mat([currExp.expFolder '\' expPathStub '_timeline.mat']);
        if exist(timelinePath, 'file')
            tim = load(timelinePath, 'Timeline');
            if exist('tim', 'var')
                expList.dataTimeline{i} = tim.Timeline;
            end
        end
    end
end


for i = 1:length(newFields)
    emptyCells = cellfun(@isempty, expList.(newFields{i}));
    if all(emptyCells)
        expList = removevars(expList, newFields{i});
    else
        expList.(newFields{i})(emptyCells) = {nan};
    end
end
end


function outData = loadAttributes(objects, attributes, objPath)
if ~iscell(objects); objects = {objects}; end
if ~iscell(attributes); attributes = {attributes}; end
objects = strsplit(objects{1}, ',');
attributes = strsplit(attributes{1}, ',');

allFiles = dir(objPath);
allFiles = allFiles(cellfun(@(x) ~strcmp(x(1),'.'),{allFiles.name}'));
allFiles = {allFiles.name}';


splitNames = split(allFiles, '.');
[matchedObj, matchedAttr] = deal(ones(size(splitNames,1), 1));
if ~contains(objects, 'all')
    matchedObj = contains(splitNames(:,1), objects);
end

if ~contains(attributes{:}, 'all')
    matchedAttr = contains(splitNames(:,2), attributes{:});
end

loadPaths = fullfile(objPath, allFiles(matchedObj & matchedAttr));
loadObj = splitNames(matchedObj & matchedAttr,1);
loadAttr = splitNames(matchedObj & matchedAttr,2);
for i = 1:size(loadPaths,1)
    loadAttr{i} = strrep(loadAttr{i}, '_av_', '');
    outData.(loadObj{i}).(loadAttr{i}) = readNPY(loadPaths{i});
end
end


