function expList = loadData(varargin)
%% Load ev and/or spk data from particular mice and/or dates
% 
% NOTE: This function uses csv.inputValidate to parse inputs. Paramters are 
% name-value pairs, including those specific to this function
%
% NOTE: That the same dataTypes, objects, and attributes will be loaded for
% all mice. You will have to separate mice before calling the function if
% you want these to be different for each mouse. ALSO, the length of
% objects should equal the length of dataTypes OR "1" and the same is true
% for attributes. If you want to suppress the written confirmation of
% loading, then set "verbose" to 0.
%
% Parameters: 
% ---------------
% Classic PinkRigs inputs (optional)
%
% dataType (default='events'): str/cell of strings 
%   indicates which data types to load.   
%   'blk' or 'block': raw block (output = dataBlock)
%   'tim' or 'timeline': raw timeline (output = dataTimeline)
%   'cam' or 'cameras': camera data (output = dataCam) 
%   'mic' or 'microphone': raw microphone data (output = dataMic) 
%   'opto' or 'optoLog': raw optoData
%   'ev' or 'events':  trial events (output = dataEvents)
%   'eventsFull':  all (including large) trial events (output = dataEvents)
%   'probe': load spike information (can specify probe number) (output = dataSpikes)
%   'all': loads 'blk', 'tim', 'ev' 
%   
% object (default='all'): str/cell of strings 
%   objects to load for each dataType. At the moment, this is only relevant 
%   for "probe" dataTypes but will likely be relevant for others later. 
%   Examples below:
%   'spikes' to load only spike data
%   'templates' to load only template data
%
% attribute (default='all'): str/cell of strings 
%   attributes to load for each object. At the moment, this is only relevant 
%   for "probe" dataTypes but will likely be relevant for others later. 
%   'spikes' to load only spike data
%   'templates' to load only template data
%
% Returns: 
% ---------------
% expList: table 
%   New fields with loaded data in structures will be included. 
%   {'dataBlock'; 'dataEvents'; 'dataSpikes'; 'dataTimeline';'dataCam';'dataMic';'dataOpto'}
%    These fields will not be included if they are not requested.
%
% Examples: 
% ---------------
% csv.loadData('subject', 'AV008', 'dataType', {{'tim'; 'blk'}});
% csv.loadData(queryExpTable, 'dataType', {{'tim'; 'blk'}});

varargin = ['dataType', {'events'}, varargin];
varargin = ['object', {'all'}, varargin];
varargin = ['attribute', {'all'}, varargin];
varargin = ['verbose', {1}, varargin];
varargin = [varargin, 'invariantParams', {{'dataType'; 'object'; 'attribute'}}];
params = csv.inputValidation(varargin{:});
verbose = params.verbose{1};

%% This section deals with the requested inputs to make sure they are valid
dataTypes = unnestCell(params.dataType{1});
dataTypes = dataTypes(:);
if any(contains(dataTypes, 'all'))&& length(dataTypes)~=1
    error('If requesting "all" dataTypes, then length of dataTypes should be "1"')
end

objects = cellfun(@(x) unnestCell(x), unnestCell(params.object{1},0), 'uni', 0);
objects = objects(:);
attributes = cellfun(@(x) unnestCell(x), unnestCell(params.attribute{1},0), 'uni', 0);
attributes = attributes(:);
if length(objects) == 1
    objects = repmat(objects, length(dataTypes),1);
elseif length(objects) ~= 1 && length(dataTypes) == 1
    objects = {unnestCell(objects)};
elseif length(objects) ~= length(dataTypes)
    error('length(objects) must equal length(datatypes) if neither of them is "1"');
end
objects = cellfun(@(x) strjoin(x, ','), objects(:), 'uni', 0);


if length(attributes) == 1
    attributes = repmat(attributes, length(objects),1);
elseif length(attributes) ~= 1 && length(objects) == 1
    attributes = {unnestCell(attributes)};
elseif  length(attributes) ~= length(objects)
    error('length(objects) must equal length(attributes) if neither of them is "1"');
end
attributes = cellfun(@(x) strjoin(x, ','), attributes(:), 'uni', 0);
params = rmfield(params, {'dataType'; 'object'; 'attribute';'verbose'});

%% 
expList = csv.queryExp(params);

% Add new fields for loaded data to the expList
newFields = {'dataBlock'; 'dataEvents'; 'dataSpikes'; 'dataTimeline'; 'dataCam'; 'dataMic';'dataOpto'};
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
    clear dataBlock dataEvents dataSpikes dataTimeline dataMic dataOpto

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
            if ~contains({ONENames{j},'probes', 'all'}, dataTypes)
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
            for j = fields(largeEvents)'
                expList.dataEvents{i}.(j{1}) = largeEvents.(j{1});
            end
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

    %% load optoLog if requested 
    if any(contains(dataTypes, {'opto'; 'optoLog'}))
        optoPath = cell2mat([currExp.expFolder '\' expPathStub '_optoLog.mat']);
        optoPath_old = cell2mat([currExp.expFolder '\' expPathStub '_optoMetaData.csv']);
        if exist(optoPath, 'file')
            opto = load(optoPath);

            if exist('opto', 'var')
                expList.dataoptoLog{i} = opto;
            end

        elseif exist(optoPath_old, 'file')
            opto = table2struct(csv.readTable(optoPath_old));
            if exist('opto', 'var')
                expList.dataoptoLog{i} = opto;
            end
        end         
    end

    %% load cam data if requested
    if any(contains(dataTypes, {'cam', 'all'}, 'IgnoreCase',1))
        dataIdx = contains(ONENames, 'Cam');
        for j = find(dataIdx)'
            if isempty(j); continue; end
            %If requested ONEFolder for a specific camera, skip other probes
            if ~contains({ONENames{j}, 'cameras','all'}, dataTypes)
                continue
            end
            %If requested object "cam" load all objects in camera folder
            camStatus = cell2mat(cellfun(@str2num, split(currExp.(sprintf('fMap%s%s',upper(ONENames{j}(1)),ONENames{j}(2:end))){1}, ','), 'uni', 0));
            if isnan(camStatus) || ~(camStatus == 1)
                continue;
            end

            objPath = fullfile(ONEPath, ONENames{j});
            idx = contains(dataTypes, {'cam', 'all'}, 'IgnoreCase',1);
            expList.dataCam{i}.(ONENames{j}) = ...
                loadAttributes(objects(idx), attributes(idx), objPath);
        end
    end

    %% load mic data if requested
    if any(contains(dataTypes, {'mic'; 'microphone'}))
        micPath = cell2mat([currExp.expFolder '\' expPathStub '_mic.mat']);
        if exist(micPath, 'file')
            mic = load(micPath);
            if exist('mic', 'var')
                expList.dataMic{i} = mic;
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
for j = 1:length(objects)
    object = strsplit(objects{j}, ',');
    attribute = strsplit(attributes{j}, ',');

    allFiles = dir(objPath);
    allFiles = allFiles(cellfun(@(x) ~strcmp(x(1),'.'),{allFiles.name}'));
    allFiles = {allFiles.name}';


    splitNames = split(allFiles, '.');
    [matchedObj, matchedAttr] = deal(ones(size(splitNames,1), 1));
    if ~contains(object, 'all')
        matchedObj = contains(splitNames(:,1), object);
    end

    if ~contains(attribute, 'all')
        matchedAttr = contains(splitNames(:,2), attribute);
    end

    loadPaths = fullfile(objPath, allFiles(matchedObj & matchedAttr));
    loadObj = splitNames(matchedObj & matchedAttr,1);
    loadAttr = splitNames(matchedObj & matchedAttr,2);
    loadExt = splitNames(matchedObj & matchedAttr,4);
    for i = 1:size(loadPaths,1)
        loadObj{i} = strrep(loadObj{i}, '_av_', '');
        loadObj{i} = strrep(loadObj{i}, '_bc_', 'bc_');
        loadAttr{i} = strrep(loadAttr{i}, '_av_', '');
        loadAttr{i} = strrep(loadAttr{i}, '_bc_', 'bc_');
        if contains(loadExt{i},{'npy'})
            outData.(loadObj{i}).(loadAttr{i}) = readNPY(loadPaths{i});
        elseif contains(loadExt{i},{'pqt','parquet'})
            outData.(loadObj{i}).(loadAttr{i}) = table2struct(parquetread(loadPaths{i}),"ToScalar",1);
        end
    end
end
end


