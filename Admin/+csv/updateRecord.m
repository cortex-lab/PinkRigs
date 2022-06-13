function csvData = updateRecord(varargin) %subject, expDate, expNum, saveData)
if isempty(varargin); csvData = []; return; end
varargin = ['subject', {'active'}, varargin];
varargin = ['expDate', 1, varargin];
varargin = ['expNum', {'all'}, varargin];
varargin = ['saveData', {1}, varargin];
varargin = ['queryExp', {1}, varargin];
params = csv.inputValidation(varargin{:});
serverLocations = getServersList;

if params.queryExp{1}
    expList = csv.queryExp(params);
    csvData = arrayfun(@(x) csv.updateRecord(expList(x,1:3), 'queryExp', 0, ...
        'saveData', params.saveData{1}), 1:height(expList), 'uni', 0);
    csvData = vertcat(csvData{:});
    return;
end

%Check if servers are all accessible before updating anything
if ~all(cellfun(@(x) exist(x, 'dir'), serverLocations))
    error('No server access so cannot update');
end
csvData = [];

subject = params.subject{1}; 
expDate = params.expDate{1}; 
expNum = params.expNum{1}; 

expPath = getExpPath(subject, expDate, expNum);
nameStub = [expDate '_' expNum '_' subject];
blockPath = [fullfile(expPath, nameStub) '_Block.mat'];

csvPathMouse = csv.getLocation(subject);

nDat.expDate = {expDate};
nDat.expNum = {expNum};
nDat.expDef = {};
nDat.expDuration = {};
nDat.rigName = {};
nDat.block = {};
nDat.timeline = {};
nDat.frontCam = {};
nDat.sideCam = {};
nDat.eyeCam = {};
nDat.micDat = {};
nDat.ephysFolderExists = {};
nDat.alignBlkFrontSideEyeMicEphys = {};
nDat.faceMapFrontSideEye = {};
nDat.issorted = {};
nDat.preProcSpkEV = {};
nDat.expFolder = {};

if ~exist(csvPathMouse, 'file') && params.saveData{1}
    csvDataMouse = struct2table(nDat, 'AsArray', 1);
    writetable(csvDataMouse,csvPathMouse,'Delimiter',',');
end

if ~exist(blockPath, 'file')
    fprintf('No block file for %s %s %s. Skipping... \n', subject, expDate, expNum);
    pause(0.01);
    if params.saveData{1}; csv.removeDataRow(subject, expDate, expNum); end
    return
end

blk = load(blockPath); blk = blk.block;
if ~contains(blk.rigName, 'zelda'); return; end
if blk.duration/60<2
    fprintf('Block < 2 mins for %s %s %s. Skipping... \n', subject, expDate, expNum);
    if params.saveData{1}; csv.removeDataRow(subject, expDate, expNum); end
    pause(0.01);
    return;
elseif blk.duration/60<5 && contains(blk.expDef, {'training'; 'multiSpaceWorld'})
    fprintf('Training block < 5 mins for %s %s %s. Skipping... \n', subject, expDate, expNum);
    if params.saveData{1}; csv.removeDataRow(subject, expDate, expNum); end
    pause(0.01);
    return;
end

[~, nDat.expDef] = fileparts(blk.expDef);
nDat.expDuration = blk.duration;
nDat.rigName = blk.rigName;
camNames = {'front';'side';'eye'};
%%
folderContents = dir(fileparts(blockPath));
nDat.block = max([folderContents(contains({folderContents.name}','Block.mat')).bytes 0]);
nDat.timeline = max([folderContents(contains({folderContents.name}','Timeline.mat')).bytes 0]);
nDat.sideCam = max([folderContents(contains({folderContents.name}','sideCam.mj2')).bytes 0]);
nDat.frontCam = max([folderContents(contains({folderContents.name}','frontCam.mj2')).bytes 0]);
nDat.eyeCam = max([folderContents(contains({folderContents.name}','eyeCam.mj2')).bytes 0]);
nDat.micDat = max([folderContents(contains({folderContents.name}','mic.mat')).bytes 0]);
nDat.ephysFolderExists = exist(fullfile(fileparts(expPath), 'ephys'), 'dir')>0;
nDat.expFolder = {fileparts(blockPath)};

nDat.alignBlkFrontSideEyeMicEphys = zeros(1,6);
nDat.issorted = '0';
alignFile = contains({folderContents.name}', [nameStub '_alignment.mat']);

if any(alignFile) 
    alignment = load([fullfile(folderContents(alignFile).folder,nameStub) '_alignment.mat']);
    
    fileExists = [nDat.block, nDat.frontCam, nDat.sideCam, nDat.eyeCam, nDat.micDat, nDat.ephysFolderExists]>0;
    tDat = nDat.alignBlkFrontSideEyeMicEphys;
    tDat(~fileExists) = NaN;

    if isfield(alignment, 'video')
        for i = 1:length(camNames)
            camIdx = contains({alignment.video.name}', camNames(i));
            if (isempty(camIdx) || ~any(camIdx)) && ~isnan(tDat(i + i))
                fprintf('FAILED: Conflict between cameras detected by align file and CSV?! for %s %s %s. ... \n', subject, expDate, expNum);
                return;
            elseif isempty(camIdx) || ~any(camIdx); tDat(i+1) = nan;
            elseif isnan(alignment.video(camIdx).frameTimes(1)); tDat(i+1) = nan;
            elseif strcmpi(alignment.video(camIdx).frameTimes, 'error'); tDat(i+1) = 2;
            elseif isnumeric(alignment.video(camIdx).frameTimes); tDat(i+1) = 1;
            end
        end
    else
        tDat(2:4) = 0;
    end

    tstDat = {'block', 1; 'mic', 5; 'ephys', 6};
    for i = 1:size(tstDat,1)
        if ~isfield(alignment, tstDat{i,1}); tDat(tstDat{i,2}) = 0;
        elseif isstruct(alignment.(tstDat{i,1})); tDat(tstDat{i,2}) = 1;
        elseif strcmpi(alignment.(tstDat{i,1}), 'error'); tDat(tstDat{i,2}) = 2;
        elseif isnan(alignment.(tstDat{i,1})); tDat(tstDat{i,2}) = NaN;
        end
    end

    nDat.alignBlkFrontSideEyeMicEphys = tDat;
    
    if nDat.alignBlkFrontSideEyeMicEphys(6) == 1
        issorted = cellfun(@(x) ~isempty(dir([x '\**\*rez.mat'])), {alignment.ephys.ephysPath});
        nDat.issorted = num2str(mean(issorted));
    elseif isnan(nDat.alignBlkFrontSideEyeMicEphys(6))
        nDat.issorted = nan;
    else
    end    
end
if isnan(nDat.alignBlkFrontSideEyeMicEphys(6)) && round(now-blk.endDateTime) < 7
    nDat.alignBlkFrontSideEyeMicEphys(6) = 0;
    nDat.issorted = 0;
end

faceMapDetect = double(cellfun(@(x) any(contains({folderContents.name}', [x 'Cam_proc.npy'])), camNames'));
faceMapDetect(isnan(nDat.alignBlkFrontSideEyeMicEphys(2:4))) = nan;

nDat.preProcSpkEV = zeros(1,2);
preProcFile = contains({folderContents.name}','preprocData.mat');
if any(preProcFile)
    preProcDat = load([fullfile(folderContents(alignFile).folder,nameStub) '_preprocData.mat']);
    if isfield(preProcDat, 'ev'); ev = preProcDat.ev; else, ev = 0; end
    if isfield(preProcDat, 'spk'); spk = preProcDat.spk; else, spk = 0; end
    
    if iscell(spk); spk = spk{1}; end
    tstDat = {spk, 1; ev, 2};
    for i = 1:size(tstDat,1)
        if isnumeric(tstDat{i,1}) && tstDat{i,1} == 0; nDat.preProcSpkEV(tstDat{i,2}) = 0;
        elseif isstruct(tstDat{i,1});  nDat.preProcSpkEV(tstDat{i,2}) = 1;
        elseif strcmpi(tstDat{i,1}, 'error'); nDat.preProcSpkEV(tstDat{i,2}) = 2;
        elseif isnan(tstDat{i,1}); nDat.preProcSpkEV(tstDat{i,2}) = NaN;
        elseif ~isempty(tstDat{i,1}); nDat.preProcSpkEV(tstDat{i,2}) = 1;
        end
    end

    if str2double(nDat.issorted) ~= 1 && nDat.preProcSpkEV(1) == 1
        nDat.preProcSpkEV(1) = 0;
    end
end

if isnan(nDat.alignBlkFrontSideEyeMicEphys(6)) && nDat.preProcSpkEV(1) == 0
    nDat.preProcSpkEV(1) = nan;
end
if isnan(nDat.alignBlkFrontSideEyeMicEphys(6)) && str2double(nDat.issorted) == 0
    nDat.issorted = num2str(nan);
end

upperFolderContents = dir(fileparts(folderContents(1).folder))';
if any(strcmpi('AllErrorsValidated.txt', [{folderContents.name}'; {upperFolderContents.name}']))
    fprintf('Errors have been validated for %s \n', folderContents(1).folder)
    nDat.preProcSpkEV(nDat.preProcSpkEV==2) = nan;
    nDat.alignBlkFrontSideEyeMicEphys(nDat.alignBlkFrontSideEyeMicEphys==2) = nan;
    faceMapDetect(faceMapDetect==2) = nan;
end
nDat.preProcSpkEV = regexprep(num2str(nDat.preProcSpkEV),'\s+',',');
nDat.alignBlkFrontSideEyeMicEphys = regexprep(num2str(nDat.alignBlkFrontSideEyeMicEphys),'\s+',',');
nDat.faceMapFrontSideEye = regexprep(num2str(faceMapDetect),'\s+',',');

csvData = struct2table(nDat, 'AsArray', 1);
if params.saveData{1}
    combinedData = csv.insertNewData(csvData, subject);
    csv.writeClean(combinedData, csvPathMouse, 0);
end
end
