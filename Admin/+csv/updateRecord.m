function csvData = updateRecord(varargin)
%% Function to check for any new recordings on the pink rigs and update csvs
% NOTE: This function uses csv.inputValidate to parse inputs

% In genral, the following meanings should be ascrived to the values in the
% csv files, and this code attempts to be consistent with that:
%   '1' indicates "all good" whether it's alingment, spk, etc.
%   '0' indicates "not attempted yet"
%   '2' indicates there was an error in processing this
%   'NaN' indicates "not done, and not expected to be done"

% For example, issortedKS2 would be "NaN" if there was no ephys recorded
% since you would not expect to extract any spikes in this case. Similarly
% if the faceCam file is missing, then alignment would be "NaN" for
% facecam, since there is no hope of alignning the file. Conversely, if the
% file did exist, but there was an error in aligning the frames, it should
% be a "2".

% Return an empty "csvData" if no inputs are given
if isempty(varargin); csvData = []; return; end

%% This section populates an strcture and checks for file existence
% Add default values and extra inputs:
% "saveData": logical--whether the updated csv entry should be saved
% "queryExp": logical--whether to run csv.queryExp and update returned rows
varargin = ['subject', {'active'}, varargin]; % For clarity--default anyway
varargin = ['expDate', 1, varargin];
varargin = ['expNum', {'all'}, varargin];
varargin = ['saveData', {1}, varargin];
varargin = ['queryExp', {1}, varargin];
params = csv.inputValidation(varargin{:});

% Take first value since these inputs cannot differ between mice
queryExp = params.queryExp{1};
saveData = params.saveData{1};

% NOTE: this is a recursive to update multiple EXISTING entries (not new)
% according to the standard csv.inputValidation input options. If queryExp
% is true, it will run csv.queryExp on "params", then csv.updateRecord on
% each row of the table that csv.queryExp returns. This isn't particularly
% good coding practice... but it works here
if queryExp
    expList = csv.queryExp(params);
    csvData = arrayfun(@(x) csv.updateRecord(expList(x,1:3), 'queryExp', 0, ...
        'saveData', saveData), 1:height(expList), 'uni', 0);
    csvData = vertcat(csvData{:});
    % Once complete, return to the original function
    return;
end

% Check if servers are all accessible before updating anything
serverLocations = getServersList;
if ~all(cellfun(@(x) exist(x, 'dir'), serverLocations))
    error('No server access so cannot update');
end
csvData = [];

% Since this *must* be a single experiment now, we can take the first cell
subject = params.subject{1};
expDate = params.expDate{1};
expNum = params.expNum{1};

% Get relevant paths and file names
expPath = getExpPath(subject, expDate, expNum);
nameStub = [expDate '_' expNum '_' subject];
blockPath = [fullfile(expPath, nameStub) '_Block.mat'];
csvPathMouse = csv.getLocation(subject);

% Populate the structure "nDat" with all the fields expected in the csv
nDat.expDate = {expDate}; % Date of experiment
nDat.expNum = {expNum}; % Number of experiment
nDat.expDef = {}; % exp definition used for experiment
nDat.expDuration = {}; % duration of experiment
nDat.rigName = {}; % name of rig (stim server) where experiment ran
nDat.block = {}; % size of block file
nDat.timeline = {}; % size of timeline file
nDat.frontCam = {}; % size of frontCam file
nDat.sideCam = {}; % size of sideCam file
nDat.eyeCam = {}; % size of eyeCam file
nDat.micDat = {}; % size of microphone data file
nDat.ephysFolderExists = {}; % logical--if ephys folder exists on date
nDat.alignBlkFrontSideEyeMicEphys = {}; % string defining alignment status
nDat.faceMapFrontSideEye = {}; %
nDat.issortedKS2 = {}; % logical--is there a Kilosort output yet
nDat.issortedPyKS = {}; % logical--is there a PyKilosort output yet
nDat.preProcSpkEV = {}; % string defining preprocessing status
nDat.expFolder = {}; % the experiment folder

% If a mouse csv doesn't exist, write empty data to a csv and create it
if ~exist(csvPathMouse, 'file') && saveData
    csvDataMouse = struct2table(nDat, 'AsArray', 1);
    writetable(csvDataMouse,csvPathMouse,'Delimiter',',');
end

% If the block file doesn't exist, remove any existing row and "return"
if ~exist(blockPath, 'file')
    fprintf('No block file for %s %s %s. Skipping... \n', subject, expDate, expNum);
    pause(0.01);
    if saveData; csv.removeDataRow(subject, expDate, expNum); end
    return
end

% Check whether the rigName contains "zelda" and "return" if it doesn't
% This assumes that all data we want to process has been recorded on the
% pink rigs, but this may not be true (e.g. see current exception for
% FT009). Need to decide how to change this in the future...??
blk = load(blockPath); blk = blk.block;
if ~contains(blk.rigName, 'zelda') && ~strcmp(subject, 'FT009')
    return;
end

% If the block duration is less than 2 mins, or less than 5 mins in the
% case of a "training" experiment, remove any existing row and "return"
if blk.duration/60<2
    fprintf('Block < 2 mins for %s %s %s. Skipping... \n', subject, expDate, expNum);
    if saveData; csv.removeDataRow(subject, expDate, expNum); end
    pause(0.01);
    return;
elseif blk.duration/60<5 && contains(blk.expDef, {'training'; 'multiSpaceWorld'})
    fprintf('Training block < 5 mins for %s %s %s. Skipping... \n', subject, expDate, expNum);
    if saveData; csv.removeDataRow(subject, expDate, expNum); end
    pause(0.01);
    return;
end

% Populated nDat with some basic infro from the block file
[~, nDat.expDef] = fileparts(blk.expDef);
nDat.expDuration = blk.duration;
nDat.rigName = blk.rigName;

% These are the cam names on the pink rigs. Currently HARDCODED
camNames = {'front';'side';'eye'};

% Check if a file called "IgnoreExperiment.txt" exists in the experiment
% folder (or one level higher). If so, remove any existing row and "return"
folderContents = dir(fileparts(blockPath));
upperFolderContents = dir(fileparts(folderContents(1).folder))';
if any(strcmpi('IgnoreExperiment.txt', [{folderContents.name}'; {upperFolderContents.name}']))
    fprintf('Ignoring experiment due to .txt file %s \n', folderContents(1).folder)
    if saveData; csv.removeDataRow(subject, expDate, expNum); end
    pause(0.01);
    return;
end

% Populate fields of nDat with the size of the corresponding file (will
% have a value of 0 if no file matching the name exists)
nDat.block = max([folderContents(contains({folderContents.name}','Block.mat')).bytes 0]);
nDat.timeline = max([folderContents(contains({folderContents.name}','Timeline.mat')).bytes 0]);
nDat.sideCam = max([folderContents(contains({folderContents.name}','sideCam.mj2')).bytes 0]);
nDat.frontCam = max([folderContents(contains({folderContents.name}','frontCam.mj2')).bytes 0]);
nDat.eyeCam = max([folderContents(contains({folderContents.name}','eyeCam.mj2')).bytes 0]);
nDat.micDat = max([folderContents(contains({folderContents.name}','mic.mat')).bytes 0]);

% Check if there is an ephys folder
nDat.ephysFolderExists = exist(fullfile(fileparts(expPath), 'ephys'), 'dir')>0;

% Record the experiment folder
nDat.expFolder = {fileparts(blockPath)};

%% This section deals with the "alignment" and "sorted" status
% alignBlkFrontSideEyeMicEphys is a string with 6 comma-separated
% entries to indicate the alignment status of the block, front/side/eye
% cameras, ephys, and microphone. Note that this begins as a vector of 6
% values and is converted to a string later

% Initialize alignBlkFrontSideEyeMicEphys and "issorted" with zeros
nDat.alignBlkFrontSideEyeMicEphys = zeros(1,6);
nDat.issortedKS2 = '0';
nDat.issortedPyKS = '0';

% Get the path of the alignement file. If so, check the alignement status
alignFile = contains({folderContents.name}', [nameStub '_alignment.mat']);
if any(alignFile)
    % Load the alignment file
    alignment = load([fullfile(folderContents(alignFile).folder,nameStub) '_alignment.mat']);

    % Check if files exist. If not, then corresponding alignment is NaN.
    % Here "tDat" is just a temp variable to save writing alignBlkFrontSideEyeMicEphys
    fileExists = [nDat.block, nDat.frontCam, nDat.sideCam, nDat.eyeCam, nDat.micDat, nDat.ephysFolderExists]>0;
    tDat = nDat.alignBlkFrontSideEyeMicEphys;
    tDat(~fileExists) = NaN;

    % This loop checks the alignement status for each video
    if isfield(alignment, 'video') && ~isempty(fields(alignment.video))
        for i = 1:length(camNames) % Loop over the camera names
            % Find the idx corresponding to the current camera
            camIdx = contains({alignment.video.name}', camNames(i));

            % Sanity check: return if cam file exists but name is wrong
            if (isempty(camIdx) || ~any(camIdx)) && ~isnan(tDat(i + i))
                fprintf('FAILED: Conflict between cameras detected by align file and CSV?! for %s %s %s. ... \n', subject, expDate, expNum);
                return;
            end

            if isempty(camIdx) || ~any(camIdx) || isnan(alignment.video(camIdx).frameTimes(1))
                % Issue a "NaN" if cam entry is missing or a NaN
                tDat(i+1) = nan;
            elseif strcmpi(alignment.video(camIdx).frameTimes, 'error')
                % Issue a "2" if there was an error in processing
                tDat(i+1) = 2;
            elseif isnumeric(alignment.video(camIdx).frameTimes)
                % Issue a "1" if frameTimes are numerci (i.e. alignment worked)
                tDat(i+1) = 1;
            end
        end
    elseif isfield(alignment, 'video') && isempty(fields(alignment.video))
        % If the video structure is empty for some reason, issue errors
        tDat(2:4) = 2;
    else
        % If "video" field is missing then issue "0" to all video fields
        tDat(2:4) = 0;
    end

    % This loop checks the alignment status for "block", "ephys", and "mic"
    tstDat = {'block', 1; 'mic', 5; 'ephys', 6};
    for i = 1:size(tstDat,1)
        if ~isfield(alignment, tstDat{i,1})
            % Issue a "0" if field is missing from "alignment"
            tDat(tstDat{i,2}) = 0;
        elseif isstruct(alignment.(tstDat{i,1}))
            % Issue a "1" if a strcture is detected
            tDat(tstDat{i,2}) = 1;
        elseif strcmpi(alignment.(tstDat{i,1}), 'error')
            % Issue a "2" if the field is labeled with "error"
            tDat(tstDat{i,2}) = 2;
        elseif isnan(alignment.(tstDat{i,1}))
            % Issue a "NaN" if the field is labeled with "NaN"
            tDat(tstDat{i,2}) = NaN;
        end
    end

    % This loop checks the issorted fields if ephys alignment is good ("1")
    if tDat(6) == 1
        % If ephys alignment is "good" check if sorting files exist. If
        % they do, then give a "1" to issortedKS2 or issortedPyKS. Note
        % that an "issorted" value of 0.5 indicates that 1 file is sorted,
        % and one isn't, in a two-probe recording
        ephysPaths = {alignment.ephys.ephysPath};
        ephysPaths(cellfun(@(x) any(isnan(x)), ephysPaths)) = [];
        issortedKS2 = cellfun(@(x) ~isempty(dir([x '\**\*rez.mat'])), ephysPaths);
        nDat.issortedKS2 = num2str(mean(issortedKS2));
        issortedPyKS = cellfun(@(x) ~isempty(dir([x '\**\output\spike_times.npy'])), ephysPaths);
        nDat.issortedPyKS = num2str(mean(issortedPyKS));
    elseif isnan(tDat(6))
        % Issue a "NaN" if ephys alignment isn't "1"
        nDat.issortedKS2 = nan;
        nDat.issortedPyKS = nan;
    end

    % Assign tDat to alignBlkFrontSideEyeMicEphys field
    nDat.alignBlkFrontSideEyeMicEphys = tDat;
end

% If ephys alignment is a NaN, but the recording was made in the past 7
% days, then change the values of ephys alignment and sorting to "0"
% because the ephys data may not have transferred yet
if isnan(nDat.alignBlkFrontSideEyeMicEphys(6)) && round(now-blk.endDateTime) < 7
    nDat.alignBlkFrontSideEyeMicEphys(6) = 0;
    nDat.issortedKS2 = 0;
    nDat.issortedPyKS = 0;
end

% Determine if the facemap files exist for each camera. If not, issue a "0"
% unless the camera alignment is a "NaN" in which case issue a "NaN"
faceMapDetect = double(cellfun(@(x) any(contains({folderContents.name}', [x 'Cam_proc.npy'])), camNames'));
faceMapDetect(isnan(nDat.alignBlkFrontSideEyeMicEphys(2:4))) = nan;

%% This section deals with the "preProcSpkEV" status
% preProcSpkEV is a string with 2 comma-separated entries to indicate the 
% preproc statuse for "ev" (which are the events extracted from the block
% and timelines) and "spk" which is the spike output from the sorting

% Get the path of the preproc file. If it exists, check status
preProcFile = contains({folderContents.name}','preprocData.mat');

% Initialize preProcSpkEV with zeros
nDat.preProcSpkEV = zeros(1,2);
if any(preProcFile)
    % Load the preproc file as "preProcDat"
    preProcDat = load([fullfile(folderContents(alignFile).folder,nameStub) '_preprocData.mat']);

    % Check whether the "ev" and "spk" fields exist. If they do, then
    % assign them to variables "ev" and "spk". If they don't, set both
    % variables equal to zero
    if isfield(preProcDat, 'ev'); ev = preProcDat.ev; else, ev = 0; end
    if isfield(preProcDat, 'spk'); spk = preProcDat.spk; else, spk = 0; end

    % If "spk" is a cell, then just use the first entry to check whether
    % spks were properly processed. @Celian, this needs to change now?!
    if iscell(spk); spk = spk{1}; end

    % Loop over "spk" and "ev" and issue the correct status
    tstDat = {spk, 1; ev, 2};
    for i = 1:size(tstDat,1)
        if isnumeric(tstDat{i,1}) && tstDat{i,1} == 0
            % Issue a "0" if the value is "0" (i.e. didn't exist above)
            nDat.preProcSpkEV(tstDat{i,2}) = 0;
        elseif isstruct(tstDat{i,1})
            % Issue "1" if it's a struct
            nDat.preProcSpkEV(tstDat{i,2}) = 1;
        elseif strcmpi(tstDat{i,1}, 'error')
            % Issue a "2" if the field is labeled with "error"
            nDat.preProcSpkEV(tstDat{i,2}) = 2;
        elseif isnan(tstDat{i,1}) 
            % Issue a "NaN" if the field is labeled with "NaN"
            nDat.preProcSpkEV(tstDat{i,2}) = NaN;
        elseif ~isempty(tstDat{i,1})
            % Otherwise, if it isn't empty, then issue a "1"
            nDat.preProcSpkEV(tstDat{i,2}) = 1;
        end
    end
    
    % If "issorted" is not "1" (i.e. sorting is incomplete", the make the 
    % preProcSpkEV(1) = "0" even if it was previously something else. This
    % can happen if, for example, one probe has been sorted and the other
    % hasn't
    if str2double(nDat.issortedKS2) ~= 1 && nDat.preProcSpkEV(1) == 1
        nDat.preProcSpkEV(1) = 0;
    end
%%%%%%% NOTE: Below would require PyKS sorting, which we are leaving atm
%     if str2double(nDat.issortedPyKS) ~= 1 && nDat.preProcSpkEV(1) == 1
%         nDat.preProcSpkEV(1) = 0;
%     end
end

%% This section is a final cleanup and dealing with some edge cases

% If ephys alignement is "NaN" then issue "NaN" to preProcSpkEV(1)
if isnan(nDat.alignBlkFrontSideEyeMicEphys(6)) && nDat.preProcSpkEV(1) == 0
    nDat.preProcSpkEV(1) = nan;
end
% If ephys alignement is "NaN" then issue "NaN" and sorting is still "0"
% (i.e. in a "waiting" state) then issue "NaN" to issorted field. 
% NOTE that this only applies to KS2 atm, since PyKS is new.
if isnan(nDat.alignBlkFrontSideEyeMicEphys(6)) && str2double(nDat.issortedKS2) == 0
    nDat.issortedKS2 = num2str(nan);
end

% If a file called "AllErrorsValidated.txt" is detected in the exp folder,
% or the parent folder, then replace all cases of "2" with a "NaN" as the
% presence of the files indicates that those errors have been checked and
% could not be resolved.
if any(strcmpi('AllErrorsValidated.txt', [{folderContents.name}'; {upperFolderContents.name}']))
    fprintf('Errors have been validated for %s \n', folderContents(1).folder)
    nDat.preProcSpkEV(nDat.preProcSpkEV==2) = nan;
    nDat.alignBlkFrontSideEyeMicEphys(nDat.alignBlkFrontSideEyeMicEphys==2) = nan;
    faceMapDetect(faceMapDetect==2) = nan;
end

% Change "alignBlkFrontSideEyeMicEphys", "preProcSpkEV", and
% "faceMapFrontSideEye" from vectors to comma-separated strings
nDat.preProcSpkEV = regexprep(num2str(nDat.preProcSpkEV),'\s+',',');
nDat.alignBlkFrontSideEyeMicEphys = regexprep(num2str(nDat.alignBlkFrontSideEyeMicEphys),'\s+',',');
nDat.faceMapFrontSideEye = regexprep(num2str(faceMapDetect),'\s+',',');

% If "saveData" then insert the new data into the existing csv
csvData = struct2table(nDat, 'AsArray', 1);
if saveData
    combinedData = csv.insertNewData(csvData, subject);
    csv.writeClean(combinedData, csvPathMouse, 0);
end
end
