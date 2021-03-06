function [data, dataDates]  = getDataFromDates(subject, requestedDates, expNum, expDef)
%% Function to load proessed files from dates. Works with files from the convertExpFiles funciton.

%INPUTS(default values)
%subject(required)----------The subject for which the dates are requested
%requestedDates('last')-----A string representing the dates requested. Can be...
%                'yyyy-mm-dd'--------------A specific date
%                'all'---------------------All data
%                'lastx'-------------------The last x days of data (especially useful during training)
%                'firstx'------------------The first x days of date
%                'yest'--------------------The x-1 day, where x is the most recent day
%                'yyyy-mm-dd:yyyy-mm-dd'---Dates in this range (including the boundaries)

%expDef('multiSpaceWorld')--Specify the expDef to be loaded (otherwise blk files will no concatenate properly)

%OUTPUTS
%data-----------------------A struct array of blk files, with additional raw, timeline, and ephys data if requested

%% Check inputs are cells, assign defaults, load the expList (as defined by prc.scanForNewFiles)
if ~exist('subject', 'var'); error('Must specify subject'); end
if ~exist('requestedDates', 'var') || isempty(requestedDates); requestedDates = {'last'}; end
if ~exist('expDef', 'var'); expDef = 'any'; end
if ~exist('expNum', 'var'); expNum = 'any'; end
if ~iscell(requestedDates); requestedDates = {requestedDates}; end
if ~iscell(subject); subject = {subject}; end
if iscell(requestedDates{1}); requestedDates = requestedDates{1}; end
%

expList = csv.readTable(csv.getLocation(subject{1}));
%Get list of available experiments for selected subject, update the paths, convert dates to datenums
availableExps = expList;
if ~strcmpi(expDef, 'any')
    availableExps = availableExps(strcmp(expList.expDef, expDef),:);
    expList = expList(strcmp(expList.expDef, expDef),:);
end
if ~strcmpi(expNum, 'any')
    availableExps = availableExps(strcmp(expList.expNum, expNum),:);
    expList = expList(strcmp(expList.expDef, expDef),:);
end
if isempty(availableExps); warning(['No processed files matching criteria for' subject{1}]); return; end
availableDateNums = datenum(availableExps.expDate);

%Depending on the "requestedDates" input, filter the available datnums
selectedDateNums = cell(size(requestedDates,1),1);
for i = 1:size(requestedDates,1)
    currDat = requestedDates{i};
    if strcmpi(currDat(1:3), 'las')
        if numel(currDat)==4; currDat = [currDat '1']; end %#ok<*AGROW>
        lastDate = str2double(currDat(5:end));
        selectedDateNums{i} = availableDateNums(end-min([lastDate length(availableDateNums)])+1:end);
    elseif strcmpi(currDat(1:3), 'fir')
        if numel(currDat)==5; currDat = [currDat '1']; end
        lastDate = str2double(currDat(6:end));
        selectedDateNums{i} = availableDateNums(1:min([length(availableDateNums), lastDate]));
    elseif strcmpi(currDat(1:3), 'yes');  selectedDateNums{i} = availableDateNums(end-1);
    elseif strcmpi(currDat(1:3), 'all');  selectedDateNums{i} = availableDateNums;
    elseif contains(lower(currDat), ':')
        dateNums = datenum(strsplit(currDat, ':')', 'yyyy-mm-dd');
        selectedDateNums{i} = availableDateNums(availableDateNums>=dateNums(1) & availableDateNums<=dateNums(2));
    else, selectedDateNums = datenum(requestedDates, 'yyyy-mm-dd');
    end
end
if iscell(selectedDateNums); selectedDateNums = unique(cell2mat(selectedDateNums)); end
selectedFiles = availableExps(ismember(availableDateNums, selectedDateNums),:);

% Load selected files
data = cell(size(selectedFiles,1),1);
dataDates = cell(size(selectedFiles,1),1);

for i = 1:size(selectedFiles,1)
    tDat = selectedFiles(i,:);
    blkName = [datestr(tDat.expDate, 'yyyy-mm-dd') '_' tDat.expNum{1} '_' subject{1} '_Block.mat'];
    requestedBlock = load([tDat.expFolder{1} '\' blkName]);
    data{i,1} = requestedBlock.block;
    dataDates{i,1} = datestr(tDat.expDate, 'yyyy-mm-dd');
end
end