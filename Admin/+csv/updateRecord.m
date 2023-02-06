function csvData = updateRecord(varargin)
%% Update the csv files for specified experiments
% 
% NOTE: This function uses csv.inputValidate to parse inputs. Paramters are 
% name-value pairs, including those specific to this function
%
% Parameters:
% ------------
% Classic PinkRigs inputs (optional)
%
% In genral, the following meanings should be ascribed to the values in the
% csv files, and this code attempts to be consistent with that:
%   '1' indicates "all good" whether it's alignment, spk, etc.
%   '0' indicates "not attempted yet"
%   '2' indicates there was an error in processing this
%   'NaN' indicates "not done, and not expected to be done"
%
% For example, issortedKS2 would be "NaN" if there was no ephys recorded
% since you would not expect to extract any spikes in this case. Similarly
% if the faceCam file is missing, then alignment would be "NaN" for
% facecam, since there is no hope of alignning the file. Conversely, if the
% file did exist, but there was an error in aligning the frames, it should
% be a "2".
%
% Examples: 
% ---------------
% csv.updateRecord('subject', 'AV008', 'expDate', 'last20');
% csv.updateRecord('subject', 'all', 'expDate', 0, 'expDef', 'passive');

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
    csvData = arrayfun(@(x) csv.updateRecord(expList(x,:), 'queryExp', 0, ...
        'saveData', saveData), 1:height(expList), 'uni', 0);
    csvData = vertcat(csvData{:});
    % Once complete, return to the original function
    return;
end

% Check if servers are all accessible before updating anything
serverLocations = getServersList;
if ~all(cellfun(@(x) exist(x, 'dir'), serverLocations))
    error('No server access so cannot update (%d)', ...
        serverLocations{~cellfun(@(x) exist(x, 'dir'), serverLocations)>1});
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
nDat.existBlock = {}; % exist of block file
nDat.existTimeline = {}; % exist of timeline file
nDat.existFrontCam = {}; % exist of frontCam file
nDat.existSideCam = {}; % exist of sideCam file
nDat.existEyeCam = {}; % exist of eyeCam file
nDat.existMic = {}; % exist of microphone data file
nDat.existEphys = {}; % exist ephysFolder
nDat.alignBlock = {}; % alignment status for block file
nDat.alignFrontCam = {}; % alignment status for front file
nDat.alignSideCam = {}; % alignment status for side cam
nDat.alignEyeCam = {}; % alignment status for eye cam
nDat.alignMic = {}; % alignment status for microphone
nDat.alignEphys = {}; % alignment status for ephys
nDat.fMapFrontCam = {}; % facemap status for front cam
nDat.fMapSideCam = {}; % facemap status for side cam
nDat.fMapEyeCam = {}; % facemap status for eye cam
nDat.issortedPyKS = {}; % logical--is there a PyKilosort output yet
nDat.extractSpikes = {}; % extraction status for spikes
nDat.extractEvents = {}; % extraction status for events
nDat.expFolder = {}; % the experiment folder
nDat.ephysPathProbe0 = {}; % the experiment folder
nDat.ephysPathProbe1 = {}; % the experiment folder

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
if ~contains(blk.rigName, 'zelda') && ~contains(subject, {'FT008';'FT009';'FT010';'FT011';'FT027';'AV031'})
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
nDat.expDuration = num2str(round(blk.duration));
nDat.rigName = blk.rigName;

% Get experiment folder contents and ONE folder contents
expFoldContents = dir([fileparts(blockPath) '\**\*.*']);
expFoldContents = expFoldContents(cellfun(@(x) ~strcmp(x(1),'.'),{expFoldContents.name}'));
ONEContents = dir([fileparts(blockPath) '\ONE_preproc\**\*.*']);
ONEContents = ONEContents(cellfun(@(x) ~strcmp(x(1),'.'),{ONEContents.name}'));

% Check if a file called "IgnoreExperiment.txt" exists in the experiment
% folder (or one level higher). If so, remove any existing row and "return"
dateFoldContents = dir([fileparts(expFoldContents(1).folder)])';
dateFoldContents = dateFoldContents(cellfun(@(x) ~strcmp(x(1),'.'),{dateFoldContents.name}'));
if any(strcmpi('IgnoreExperiment.txt', [{expFoldContents.name}'; {dateFoldContents.name}']))
    fprintf('Ignoring experiment due to .txt file %s \n', expFoldContents(1).folder)
    if saveData; csv.removeDataRow(subject, expDate, expNum); end
    pause(0.01);
    return;
end

% Populate fields of nDat with the exist of the corresponding file (will
% have a value of 0 if no file matching the name exists)
nDat.existBlock = num2str(any(contains({expFoldContents.name}','Block.mat')));
nDat.existTimeline = num2str(any(contains({expFoldContents.name}','Timeline.mat')));
nDat.existSideCam = num2str(any(contains({expFoldContents.name}','sideCam.mj2')));
nDat.existFrontCam = num2str(any(contains({expFoldContents.name}','frontCam.mj2')));
nDat.existEyeCam = num2str(any(contains({expFoldContents.name}','eyeCam.mj2')));
nDat.existMic = num2str(any(contains({expFoldContents.name}','mic.mat')));

% Check if there is an ephys folder
nDat.existEphys = num2str(exist(fullfile(fileparts(expPath), 'ephys'), 'dir')>0);

% Record the experiment folder
nDat.expFolder = {fileparts(blockPath)};

%% This section deals with the "alignment" and "facemap" status
% Get the path of the alignment file. If so, check the alignment status
alignFile = contains({expFoldContents.name}', [nameStub '_alignment.mat']);

probeInfo = csv.checkProbeUse(subject, 'all', 0, params.mainCSV{1});

% check for acute recordings .... 
if strcmpi(probeInfo.probeType{1}, 'Acute')
    alignment = load([fullfile(expFoldContents(alignFile).folder,nameStub) '_alignment.mat']);
    if isfield(alignment,'ephys') 
        potentialProbes = max([2, length(alignment.ephys)]);
    else
        potentialProbes = 0;
    end
elseif strcmpi(probeInfo.implantDate, 'none') || datenum(expDate)<datenum(probeInfo.implantDate{1})
    potentialProbes = 0;
else
    potentialProbes = length(probeInfo.serialNumbers{1});
end


% Initialize "issorted" with zeros
nDat.issortedPyKS = zeros(1, potentialProbes);


% Populate alignBlock, alignEphys, and alignMic
if any(alignFile)
    % Load the alignment file
    alignment = load([fullfile(expFoldContents(alignFile).folder,nameStub) '_alignment.mat']);
    tstName = {'Block'};
    for i = 1:size(tstName,1)
        if nDat.(['exist' tstName{i}]) == 0 || ~strcmpi(nDat.existTimeline, '1')
            % Issue a "NaN" if correspoding file or timeline doesn't exist
            nDat.(['align' tstName{i}]) = 'NaN';
        elseif ~isfield(alignment, lower(tstName{i}))
            % Issue a "0" if field is missing from "alignment"
            nDat.(['align' tstName{i}]) = '0';
        elseif isstruct(alignment.(lower(tstName{i})))
            % Issue a "1" if a strcture is detected
            nDat.(['align' tstName{i}]) = '1';
        elseif isnan(alignment.(lower(tstName{i})))
            % Issue a "nan" if value is nan (e.g. spontaneous block)
            nDat.(['align' tstName{i}]) = 'NaN';
        elseif strcmpi(alignment.(lower(tstName{i})), 'error')
            % Issue a "2" if the field is labeled with "error"
            nDat.(['align' tstName{i}]) = '2';
        end
    end
    if round(now-blk.endDateTime)<7 && ~strcmpi(nDat.existTimeline, '1')
        nDat.alignBlock = '0';
    end

    % EPHYS alignment
    if potentialProbes == 0
        % Issue a "NaN" if no implantations in main CSV
        nDat.alignEphys = nan;
    elseif strcmpi(nDat.existEphys, '0') && round(now-blk.endDateTime)<7 && nDat.existTimeline
        % Issue a "0" if no ephys, but less than 7 days since recording
        nDat.alignEphys = zeros(1, potentialProbes);
    elseif strcmpi(nDat.existEphys, '0') || ~strcmpi(nDat.existTimeline, '1')
        % Issue a "NaN" if correspoding file or timeline doesn't exist
        nDat.alignEphys = nan*ones(1, potentialProbes);
    elseif ~isfield(alignment, 'ephys')
        % Issue a "0" if field is missing from "alignment"
        nDat.alignEphys = zeros(1, potentialProbes);
    elseif isstruct(alignment.ephys)
        % Issue a "1" if a strcture is detected
        if size(alignment.ephys,2) ~= potentialProbes
            fprintf('***WARNING: mismatch between recorded and expected probe number\n')
            nDat.alignEphys = 2*ones(1, potentialProbes);
        else
            nDat.alignEphys = double(arrayfun(@(x) ~any(strcmp(x.ephysPath,'error')), alignment.ephys));
            nDat.alignEphys(nDat.alignEphys == 0) = 2;
        end
    elseif strcmpi(alignment.ephys, 'error')
        % Issue a "2" if the field is labeled with "error"
        nDat.alignEphys = 2*ones(1, potentialProbes);
    end
else
    nDat.alignBlock = '0';
    if potentialProbes == 0
        nDat.alignEphys = nan;
    else
        nDat.alignEphys = zeros(1, potentialProbes);
    end
end

nDat.ephysPathProbe0 = 'NaN';
nDat.ephysPathProbe1 = 'NaN';
if nDat.alignEphys(1) == 1
    nDat.ephysPathProbe0 = alignment.ephys(1).ephysPath;    
end
if numel(nDat.alignEphys) == 2 && nDat.alignEphys(2) == 1
    nDat.ephysPathProbe1 = alignment.ephys(2).ephysPath;    
end

%Populate alignCamera entries
for vidName = {'FrontCam'; 'SideCam'; 'EyeCam'}'
    if ~strcmpi(nDat.(['exist' vidName{1}]), '1') && round(now-blk.endDateTime)<7 && nDat.existTimeline
        % Issue a "0" if no video, but less than 7 days since recording
        nDat.(['align' vidName{1}]) = '0';
    elseif ~strcmpi(nDat.(['exist' vidName{1}]), '1') || ~strcmpi(nDat.existTimeline, '1')
        % Issue a "NaN" if corresponding file or timeline doesn't exist
        nDat.(['align' vidName{1}]) = 'NaN';
    elseif any(cellfun(@(x) ~isempty(regexpi(x, ['camera.times.*' vidName{1} '.npy'], 'once')), {ONEContents.name}'))
        % Issue a "1" if an ONE file is detected
        nDat.(['align' vidName{1}]) = '1';
    elseif any(contains({ONEContents.name}', ['Error_' vidName{1} '.json'], 'ignorecase', 1))
        errIdx = contains({ONEContents.name}', ['Error_' vidName{1} '.json'], 'ignorecase', 1);
        errFile = (fullfile(ONEContents(errIdx).folder, ONEContents(errIdx).name));
        fid = fopen(errFile);
        errText = jsondecode(char(fread(fid, inf)'));
        fclose(fid);
        if contains(errText, 'initialize internal resources')
            % Issue a "NaN" if video is corrupt
            nDat.(['align' vidName{1}]) = 'NaN';
        else
            % Issue a "2" if error is something else
            nDat.(['align' vidName{1}]) = '2';
        end
    else
        % Issue a "0" if nothing detected but camera exists LOOK AT THIS
        nDat.(['align' vidName{1}]) = '0';
    end

    % Check whether facemap processing exists
    if ~strcmpi(nDat.(['exist' vidName{1}]), '1') && round(now-blk.endDateTime)<7 && nDat.existTimeline
        % Issue a "0" if no video, but less than 7 days since recording
        nDat.(['fMap' vidName{1}]) = '0';
    elseif ~strcmpi(nDat.(['exist' vidName{1}]), '1') || ~strcmpi(nDat.existTimeline, '1')
        % Issue a "NaN" if corresponding file or timeline doesn't exist
        nDat.(['fMap' vidName{1}]) = 'NaN';
    elseif any(cellfun(@(x) ~isempty(regexpi(x, ['camera.ROIAverageFrame.*' vidName{1} '.npy'], 'once')), {ONEContents.name}'))
        % Issue a "1" if an ONE file is detected
        nDat.(['fMap' vidName{1}]) = '1';
    elseif strcmpi(nDat.(['align' vidName{1}]), 'nan')
        nDat.(['fMap' vidName{1}]) = 'NaN';
    else
        nDat.(['fMap' vidName{1}]) = '0';
    end
end

% Populate alignMic entry
if ~strcmpi(nDat.existMic, '1') && round(now-blk.endDateTime)<7 && nDat.existTimeline
    % Issue a "0" if no video, but less than 7 days since recording
    nDat.alignMic = '0';
elseif ~strcmpi(nDat.existMic, '1') || ~strcmpi(nDat.existTimeline, '1')
    % Issue a "NaN" if corresponding file or timeline doesn't exist
    nDat.alignMic = NaN;
elseif any(contains({ONEContents.name}', '_av_mic.times.npy', 'ignorecase', 1))
    % Issue a "1" if an ONE file is detected
    nDat.alignMic = '1';
elseif any(contains({ONEContents.name}', 'Error.json', 'ignorecase', 1))
    nDat.alignMic = '2';
else
    % LOOK AT THIS
    nDat.alignMic = '0';
end


%% This section deals with "sorted" status
% This loop checks the issortedPyKS fields if ephys alignment is good ("1")
nDat.issortedPyKS = zeros(1, potentialProbes);
for pIdx = find(nDat.alignEphys == 1)
    % If ephys alignment is "good" check if sorting files exist. If
    % they do, then give a "1" to issortedPyKS or issortedPyKS.
    ephysPath = alignment.ephys(pIdx).ephysPath;
    if ~isempty(dir([ephysPath '\**\output\spike_times.npy']))
        % Issue a "1" if "results" file for pyKS exists
        nDat.issortedPyKS(pIdx) = 1;
    elseif ~isempty(dir([ephysPath '\**\pyKS_error.json']))
        % Issue a "2" if error file is in folder
        nDat.issortedPyKS(pIdx) = 2;
    else
        % Issue a "0" if no error, but sorting doesn't exist yet
        nDat.issortedPyKS(pIdx) = 0;
    end
end
% Assign "nan" or "0" if ephys alignment isn't "1" accordingly
nDat.issortedPyKS(isnan(nDat.alignEphys)) = nan;
nDat.issortedPyKS(nDat.alignEphys == 0) = 0;
nDat.issortedPyKS(nDat.alignEphys == 2) = 0;

%% This section deals with the event and spike extraction status

% Assign status for events extraction
if strcmpi(nDat.alignBlock, '1')
    if any(cellfun(@(x) ~isempty(regexp(x, '_av_trials.*.pqt')), {ONEContents.name}')) %#ok<RGXP1> 
        % Issue a "1" if .pqt output is in in folder
        nDat.extractEvents = '1';
    elseif any(cellfun(@(x) ~isempty(regexp(x, '_av_trials.*.npy')), {ONEContents.name}')) %#ok<RGXP1> 
        % Issue a "1" if .pqt output is in in folder
        nDat.extractEvents = '1';
    elseif any(contains({ONEContents.name}', 'GetEvError.json', 'ignorecase', 1))
        % Issue a "2" if error file is in folder
        nDat.extractEvents = '2';
    else
        % Issue a "0" if neither error or .pqt exist yet.
        nDat.extractEvents = '0';
    end
elseif strcmpi(nDat.alignBlock, 'nan')
    % Issue a "nan" if block alignment is a nan
    nDat.extractEvents = 'NaN';
elseif strcmpi(nDat.alignBlock, '0')
    % Issue a "0" if block alignment isn't complete yet
    nDat.extractEvents = '0';
else
    % Issue a "0" if non of the above are true
    nDat.extractEvents = '0';
end

% Assign status for spike extraction.
nDat.extractSpikes = zeros(1, potentialProbes);
for pIdx = find(nDat.issortedPyKS == 1)
    % If ephys alignment is "good" check if sorting files exist. If
    % they do, then give a "1" to issortedKS2 or issortedPyKS.
    probeStr = ['probe' num2str(pIdx-1)];
    fullNames = cellfun(@(x,y) fullfile(x,y), {ONEContents.folder}', {ONEContents.name}', 'uni', 0);
    if any(cellfun(@(x) ~isempty(regexp(x, [probeStr '.*.npy'])), fullNames)) %#ok<RGXP1> 
        % Issue a "1" if "results" file for KS2 exists
        nDat.extractSpikes(pIdx) = 1;
    elseif any(cellfun(@(x) ~isempty(regexp(x, [probeStr '.*GetSpkError.json'])), fullNames)) %#ok<RGXP1> 
        % Issue a "2" if error file is in folder
        nDat.extractSpikes(pIdx) = 2;
    else
        % Issue a "0" if no error, but extraction doesn't exist yet
        nDat.extractSpikes(pIdx) = 0;
    end
end
% Assign "nan" or "0" if ephys alignment isn't "1" accordingly
nDat.extractSpikes(isnan(nDat.issortedPyKS)) = nan;

%% This section is a final cleanup and dealing with some edge cases

% If a file called "AllErrorsValidated.txt" is detected in the exp folder,
% or the parent folder, then replace all cases of "2" with a "NaN" as the
% presence of the files indicates that those errors have been checked and
% could not be resolved.
if any(strcmpi('AllErrorsValidated.txt', [{expFoldContents.name}'; {dateFoldContents.name}']))
    fprintf('Errors have been validated for %s \n', expFoldContents(1).folder)
    %%% NEED TO DO THIS %%%%%
end

% % Change probe-related fields from vectors to comma-separated strings
nDat.alignEphys = regexprep(num2str(nDat.alignEphys),'\s+',',');
nDat.issortedPyKS = regexprep(num2str(nDat.issortedPyKS),'\s+',',');
nDat.extractSpikes = regexprep(num2str(nDat.extractSpikes),'\s+',',');

% If "saveData" then insert the new data into the existing csv
csvData = struct2table(nDat, 'AsArray', 1);
if saveData
    combinedData = csv.insertNewData(csvData, subject);
    csv.writeClean(combinedData, csvPathMouse, 0);
end
end
