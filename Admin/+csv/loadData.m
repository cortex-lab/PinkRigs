function expList = loadData(varargin)
%% Load ev and/or spk data from particular mice and/or dates
% NOTE: This function uses csv.inputValidate to parse inputs

% Add default values for extra inputs:
% loadTag (default='ev'): string--indicates data types to laod
%   'blk' or 'block to load ev files (output = blockData)
%   'ev' to load ev files (output = evData)
%   'spk' to load ev files (output = spkData)
%   'tim' or 'timeline' to load ev files (output = timelineData)
%   'all' to load all data
% NOTE: loadtag continuous. i.e. 'timblk' loads timeline and block
varargin = ['loadTag', {'ev'}, varargin];
params = csv.inputValidation(varargin{:});
expList = csv.queryExp(params);

% Add new fields for loaded data to the expList
newFields = {'blockData'; 'evData'; 'spkData'; 'timelineData'};
for i = 1:length(newFields)
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
    clear ev spk blk tim;
    currExp = expList(i,:);
    currONEStub = [currExp.expFolder '\ONE_preproc\'];

    currLoadTag = currExp.loadTag{1};
    if strcmp(currLoadTag, 'all')
        currLoadTag = 'evspkblktim'; 
    end
    expPathStub = strcat(currExp.expDate, {'_'}, currExp.expNum, {'_'}, currExp.subject);
    
    %Load ev/spk data if requested
    if contains(currLoadTag, {'spk'})
        preProcPath = cell2mat([currExp.expFolder '\' expPathStub '_preprocData.mat']);
        if ~exist(preProcPath, 'file'); continue; end
        if contains(currLoadTag, {'spk'})
            spk = load(preProcPath, 'spk');
            if exist('spk', 'var')
                expList.spkData{i} = spk.spk;
            end
        end
    end

    if contains(currLoadTag, {'ev'})
        evPQTPath = cell2mat([currONEStub '\events\_av_trials.table.' expPathStub '.pqt' ]);
        if ~exist(evPQTPath, 'file'); continue; end
        expList.evData{i} = table2struct(parquetread(evPQTPath),"ToScalar",1);
    end


    %Load block data if requested
    if contains(currLoadTag, {'blk', 'block'})
        preProcPath = cell2mat([currExp.expFolder '\' expPathStub '_block.mat']);
        if ~exist(preProcPath, 'file'); continue; end
        blk = load(preProcPath, 'block');
        if exist('blk', 'var')
            expList.blockData{i} = blk.block;
        end
    end

    %Load timeline data if requested
    if contains(currLoadTag, {'tim'; 'timeline'})
        preProcPath = cell2mat([currExp.expFolder '\' expPathStub '_timeline.mat']);
        if ~exist(preProcPath, 'file'); continue; end
        tim = load(preProcPath, 'Timeline');
        if exist('tim', 'var')
            expList.timelineData{i} = tim.Timeline;
        end
    end
end

expList = removevars(expList,{'block'; 'timeline';...
    'frontCam'; 'sideCam'; 'eyeCam'; 'micDat'; 'ephysFolderExists'; ...
    'alignBlkFrontSideEyeMicEphys'; 'faceMapFrontSideEye'; 'issortedKS2'; ...
    'preProcSpkEV'; 'issortedPyKS'; 'expFolder'});

for i = 1:length(newFields)
    emptyCells = cellfun(@isempty, expList.(newFields{i}));
    if all(emptyCells)
        expList = removevars(expList, newFields{i});
    else
        expList.(newFields{i})(emptyCells) = {nan};
    end
end
end
