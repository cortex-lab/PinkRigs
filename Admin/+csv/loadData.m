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
params = csv.inputValidation(varargin{:});
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

    %% Check that the number of objects and attributes match ONEFolders
    currDataType = currExp.dataType{1};
    currObj = currExp.object{1};
    currAttr = currExp.attribute{1};
    if ~iscell(currDataType); currDataType = {currDataType}; end
    if ~iscell(currObj); currObj = {currObj}; end
    if ~iscell(currAttr); currAttr = {currAttr}; end


    if size(currObj, 1) == 1 && strcmp(currObj, 'all')
        currObj = repmat(currObj, size(currDataType, 1), 1);
    elseif size(currObj, 1) ~= size(currDataType, 1)
        error('If object is not "all" it must be provided for each ONEFolder');
    end

    if size(currAttr, 1) == 1 && strcmp(currAttr, 'all')
        currAttr = repmat(currAttr, size(currDataType, 1), 1);
    elseif size(currAttr, 1) ~= size(currDataType, 1)
        error('If attribute is not "all" it must be provided for each ONEFolder');
    end
    
    %% Load dataSpikes if requested
    if any(contains(currDataType, {'probe', 'all'}, 'IgnoreCase',1))
        dataIdx = contains(ONENames, 'probe');
        for j = find(dataIdx)'
            if isempty(j); continue; end

            %If requested ONEFolder "probe1", skip other probes
            if ~contains({ONENames{j}, 'all'}, currDataType)
                continue
            end
            %If requested object "probe" load all objects in probe folder
            objPath = fullfile(ONEPath, ONENames{j});
            loadObj = currObj(contains(currDataType, {'probe', 'all'}));
            loadAttr = currAttr(contains(currDataType, {'probe', 'all'}));
            expList.dataSpikes{i}.(ONENames{j}) = loadAttributes(loadObj, loadAttr, objPath);
        end
    end
    
    %% Load dataEvents if requested
    evCheck =  {'ev'; 'events'; 'eventsFull'; 'all';};
    if any(contains(currDataType, evCheck, 'IgnoreCase',1))
        evPQTPath = cell2mat([ONEPath 'events\_av_trials.table.' expPathStub '.pqt' ]);
        if exist(evPQTPath, 'file')
            expList.dataEvents{i} = table2struct(parquetread(evPQTPath),"ToScalar",1);
        end
        evPQTPath = strrep(evPQTPath, '.table', '.table_largeData');
        if any(contains(currDataType, 'eventsFull', 'IgnoreCase',1)) && exist(evPQTPath, 'file')
            largeEvents = table2struct(parquetread(evPQTPath),"ToScalar",1);
            expList.dataEvents{i} = catStructs(expList.dataEvents{i},largeEvents);
        end
    end

    %% Load dataBlock if requested
    if any(contains(currDataType, {'blk', 'block'}))
        blockPath = cell2mat([currExp.expFolder '\' expPathStub '_block.mat']);
        if exist(blockPath, 'file')
            blk = load(blockPath, 'block');
            if exist('blk', 'var')
                expList.dataBlock{i} = blk.block;
            end
        end
    end

    %% Load timeline data if requested
    if any(contains(currDataType, {'tim'; 'timeline'}))
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

expList = removevars(expList, {'object'; 'dataType'; 'attribute'});
end


function outData = loadAttributes(objects, attributes, objPath)
allFiles = dir(objPath);
allFiles = allFiles(cellfun(@(x) ~strcmp(x(1),'.'),{allFiles.name}'));
allFiles = {allFiles.name}';


splitNames = split(allFiles, '.');
[matchedObj, matchedAttr] = deal(ones(size(splitNames,1), 1));
if ~contains(objects, 'all')
    matchedObj = contains(splitNames(:,1), objects);
end

if ~contains(attributes, 'all')
    matchedAttr = contains(splitNames(:,2), attributes);
end

loadPaths = fullfile(objPath, allFiles(matchedObj & matchedAttr));
loadObj = splitNames(matchedObj & matchedAttr,1);
loadAttr = splitNames(matchedObj & matchedAttr,2);
for i = 1:size(loadPaths,1)
    loadAttr{i} = strrep(loadAttr{i}, '_av_', '');
    outData.(loadObj{i}).(loadAttr{i}) = readNPY(loadPaths{i});
end
end


