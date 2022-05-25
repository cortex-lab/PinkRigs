function extractedExperiments = queryExp(varargin)
%%% This function will fetch all possible experiments to check for
%%% computing alignment, preprocessing etc.
params = csv.inputValidation(varargin{:});

%Assign defaults for remaining params not contained in "inputValidation"
params = csv.addDefaultParam(params, 'timeline2Check', {0});
params = csv.addDefaultParam(params, 'align2Check', {'*,*,*,*,*,*'});
params = csv.addDefaultParam(params, 'preproc2Check', {'*,*'});
params = csv.addDefaultParam(params, 'issorted', {[0 1]});

% Loop through csv to look for experiments that weren't
% aligned, or all if recompute isn't none.
extractedExperiments = table();
for mm = 1:numel(params.subject)
    % Loop through subjects
    expListMouse = csv.readTable(csv.getLocation(params.subject{mm}));

    % Get list of exp for this mouse
    expListMouse.subject = repmat(params.subject(mm), size(expListMouse,1),1);
    expListMouse = [expListMouse(:,end) expListMouse(:,1:end-1)];

    % Remove the expDefs that don't match
    if ~strcmp(params.expDef{mm},'all')
        expListMouse = expListMouse(contains(expListMouse.expDef, params.expDef{mm}),:);
    end
    if isempty(expListMouse); continue; end

    % Remove expNums that don't match
    if ~strcmp(params.expNum{mm},'all')
        expListMouse = expListMouse(contains(expListMouse.expNum, params.expNum{mm}),:);
    end
    if isempty(expListMouse); continue; end

    % Get exp with timeline only
    if params.timeline2Check{mm}
        expListMouse = expListMouse(str2double(expListMouse.timeline)>0,:);
    end
    if isempty(expListMouse); continue; end

    % Get exp with specific alignment status
    alignCodeChecked = csv.checkStatusCode(expListMouse.alignBlkFrontSideEyeMicEphys,params.align2Check{mm});
    expListMouse = expListMouse(alignCodeChecked,:);
    if isempty(expListMouse); continue; end

    % Get exp with specific sorting status
    if params.issorted{mm}
        expListMouse = expListMouse(ismember(expListMouse.issorted, params.issorted{mm}),:);
    end
    if isempty(expListMouse); continue; end

    % Get exp with specific preprocessing state
    preproc2Check = csv.checkStatusCode(expListMouse.preProcSpkEV,params.preproc2Check{mm});
    expListMouse = expListMouse(all(preproc2Check,2),:);
    if isempty(expListMouse); continue; end

    %Convert date inputs to actual dates based on the CSV data
    currDate = params.expDate{mm};
    if ~iscell(currDate); currDate = {currDate}; end
    selectedDates = arrayfun(@(x) extractDates(x, expListMouse.expDate), currDate, 'uni', 0);
    expListMouse = expListMouse(sum(cell2mat(selectedDates),2)>0,:);
    if isempty(expListMouse); continue; end
    
    extractedExperiments = [extractedExperiments; expListMouse];
end
end

function extractedDateIdx = extractDates(currDate, dateList)
todayDate = datenum(date);
dateNumsCSV = datenum(dateList);
sortedDatesCSV = unique(dateNumsCSV);
datePat = '\d\d\d\d-\d\d-\d\d';

for i = 1:length(currDate)
    if isnumeric(currDate{i}) && (currDate{i} < todayDate-2000 || isinf(currDate{i}))
        extractedDateIdx = todayDate - datenum(currDate{i}) <= dateNumsCSV;
    elseif isnumeric(currDate{i})
        extractedDateIdx = ismember(dateNumsCSV, currDate{i});
    elseif strcmpi(currDate{i}(1:3), 'las')
        if numel(currDate{i})==4; currDate{i} = [currDate{i} '1']; end %#ok<*AGROW>
        lastDate = str2double(currDate{i}(5:end));
        extractedDateIdx = ismember(dateNumsCSV, sortedDatesCSV(end-min([lastDate length(sortedDatesCSV)])+1:end));
    elseif strcmpi(currDate{i}(1:3), 'fir')
        if numel(currDate{i})==5; currDate{i} = [currDate{i} '1']; end %#ok<*AGROW>
        lastDate = str2double(currDate{i}(5:end));
        extractedDateIdx = ismember(dateNumsCSV, sortedDatesCSV(1:min([length(sortedDatesCSV), lastDate])));
    elseif contains(lower(currDate{i}), ':')
        currDateBounds = datenum(strsplit(currDate{i}, ':')', 'yyyy-mm-dd');
        extractedDateIdx = dateNumsCSV >= currDateBounds(1) & dateNumsCSV <= currDateBounds(2);
    elseif ~isempty(regexp(currDate{i}, datePat, 'once'))
        currDateNums = datenum(regexp(currDate{i}, datePat,'match'), 'yyyy-mm-dd');
        extractedDateIdx =  ismember(dateNumsCSV, currDateNums);
    elseif strcmpi(currDate{i}, 'all')
        extractedDateIdx = ones(length(dateList), 1);
    else, error('Did not understand your date input')
    end
end
end