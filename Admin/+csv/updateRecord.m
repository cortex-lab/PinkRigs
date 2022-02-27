function csvData = updateRecord(subject, expDate, expNum, saveData)
if ~exist('subject', 'var'); error('Must provide subject'); end
if ~exist('expDate', 'var'); error('Must provide expDate'); end
if ~exist('expNum', 'var'); error('Must provide expNum'); end
if ~exist('saveData', 'var'); saveData = 1; end

if iscell(expDate); expDate = expDate{1}; end
if ~ischar(expDate); error('Cannot parse expDate'); end

if iscell(expNum); expNum = expNum{1}; end
if isnumeric(expNum); expNum = double2str(expNum); end
if ~ischar(expNum); error('Cannot parse expNum: should be a string.'); end
csvData = [];

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
% nDat.complete = {};

if ~exist(csvPathMouse, 'file') && saveData
    csvDataMouse = struct2table(nDat, 'AsArray', 1);
    writetable(csvDataMouse,csvPathMouse,'Delimiter',',');
end

if ~exist(blockPath, 'file')
    fprintf('No block file for %s %s %s. Skipping... \n', subject, expDate, expNum);
    pause(0.01);
    return
end

blk = load(blockPath); blk = blk.block;
if ~contains(blk.rigName, 'zelda'); return; end
if blk.duration/60<5
    fprintf('Block < 5 mins for %s %s %s. Skipping... \n', subject, expDate, expNum);
    pause(0.01);
    return;
end

[~, nDat.expDef] = fileparts(blk.expDef);
nDat.expDuration = blk.duration;
nDat.rigName = blk.rigName;
camNames = {'front';'side';'eye'};
%%
fileContents = dir(fileparts(blockPath));
nDat.block = max([fileContents(contains({fileContents.name}','Block.mat')).bytes 0]);
nDat.timeline = max([fileContents(contains({fileContents.name}','Timeline.mat')).bytes 0]);
nDat.sideCam = max([fileContents(contains({fileContents.name}','sideCam.mj2')).bytes 0]);
nDat.frontCam = max([fileContents(contains({fileContents.name}','frontCam.mj2')).bytes 0]);
nDat.eyeCam = max([fileContents(contains({fileContents.name}','eyeCam.mj2')).bytes 0]);
nDat.micDat = max([fileContents(contains({fileContents.name}','mic.mat')).bytes 0]);
nDat.ephysFolderExists = exist(fullfile(fileparts(expPath), 'ephys'), 'dir')>0;
nDat.expFolder = {fileparts(blockPath)};

nDat.alignBlkFrontSideEyeMicEphys = zeros(1,6);
nDat.issorted = '0';
alignFile = contains({fileContents.name}','alignment.mat');

if any(alignFile)
    load(fullfile(fileContents(alignFile).folder, 'alignment.mat'), 'alignment');
    expectedFields = {'block', 'video', 'mic', 'ephys'};
    if ~all(contains(expectedFields, fields(alignment)))
        fprintf('WARNING: fields are incorrect in alignment.mat for %s %s %s. ... \n', subject, expDate, expNum);
    else 
        fileExists = [nDat.block, nDat.frontCam, nDat.sideCam, nDat.eyeCam, nDat.micDat, nDat.ephysFolderExists]>0;
        tDat = nDat.alignBlkFrontSideEyeMicEphys;
        tDat(~fileExists) = NaN;
        
        for i = 1:length(camNames)
            camIdx = contains({alignment.video.name}', camNames(i));
            if ~any(camIdx) && ~isnan(tDat(i + i))
                fprintf('FAILED: Conflict between cameras detected by align file and CSV?! for %s %s %s. ... \n', subject, expDate, expNum);
                return;
            end
            if isnan(alignment.video(camIdx).frameTimes(1)); tDat(i+1) = nan;
            elseif strcmpi(alignment.video(camIdx).frameTimes, 'error'); tDat(i+1) = 2;
            elseif isnumeric(alignment.video(camIdx).frameTimes); tDat(i+1) = 1;
            end
        end
        
        tstDat = {alignment.block, 1; alignment.mic, 5; alignment.ephys, 6};
        for i = 1:size(tstDat,1)
            if isstruct(tstDat{i,1}); tDat(tstDat{i,2}) = 1;
            elseif strcmpi(tstDat{i,1}, 'error'); tDat(tstDat{i,2}) = 2;
            elseif isnan(tstDat{i,1}); tDat(tstDat{i,2}) = NaN;
            end
        end
                
        nDat.alignBlkFrontSideEyeMicEphys = tDat;
    end
    
    if nDat.alignBlkFrontSideEyeMicEphys(6) == 1
        issorted = cellfun(@(x) ~isempty(dir([x '\**\*rez.mat'])), {alignment.ephys.ephysPath});
        nDat.issorted = num2str(mean(issorted));
    else
        nDat.issorted = num2str(nan);
    end    
end

faceMapDetect = cellfun(@(x) any(contains({fileContents.name}', [x 'Cam_proc.npy'])), camNames');
alignedCams = nDat.alignBlkFrontSideEyeMicEphys(2:4);
faceMapDetect(~faceMapDetect) = alignedCams(~faceMapDetect);

nDat.alignBlkFrontSideEyeMicEphys = regexprep(num2str(nDat.alignBlkFrontSideEyeMicEphys),'\s+',',');
nDat.faceMapFrontSideEye = regexprep(num2str(faceMapDetect),'\s+',',');

nDat.preProcSpkEV = zeros(1,2);
preProcFile = contains({fileContents.name}','preprocData.mat');
if any(preProcFile)
    load(fullfile(fileContents(alignFile).folder, 'preprocData.mat'), 'ev', 'spk');
    if ~exist('ev', 'var') || ~exist('spk', 'var')
        fprintf('WARNING: Data missing from preprocData.mat for %s %s %s. ... \n', subject, expDate, expNum);
    else
        if iscell(spk); spk = spk{1}; end
        tstDat = {spk, 1; ev, 2};
        for i = 1:size(tstDat,1)
            if isstruct(tstDat{i,1});  nDat.preProcSpkEV(tstDat{i,2}) = 1;
            elseif strcmpi(tstDat{i,1}, 'error'); nDat.preProcSpkEV(tstDat{i,2}) = 2;
            elseif isnan(tstDat{i,1}); nDat.preProcSpkEV(tstDat{i,2}) = NaN;
            elseif ~isempty(tstDat{i,1}); nDat.preProcSpkEV(tstDat{i,2}) = 1;
            end
        end
        
        if str2double(nDat.issorted) ~= 1 && nDat.preProcSpkEV(1) == 1
            nDat.preProcSpkEV(1) = 0;
        end
    end
end
nDat.preProcSpkEV = regexprep(num2str(nDat.preProcSpkEV),'\s+',',');

csvData = struct2table(nDat, 'AsArray', 1);
if saveData
    combinedData = csv.insertNewData(csvData, subject);
    csv.writeClean(combinedData, csvPathMouse, 0);
end
end
