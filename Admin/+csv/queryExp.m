function extractedExperiments = queryExp(varargin)
%%% This function will fetch all possible experiments to check for
%%% computing alignment, preprocessing etc.

% Assign defaults for params not contained in "inputValidation"
varargin = ['timeline2Check', {0}, varargin];
varargin = ['align2Check', {'*,*,*,*,*,*'}, varargin];
varargin = ['preproc2Check', {'*,*'}, varargin];
varargin = ['issortedCheck', -1, varargin];
params = csv.inputValidation(varargin{:});

% Loop through csv to look for experiments that weren't
% aligned, or all if recompute isn't none.
extractedExperiments = table();
for mm = 1:numel(params.subject)
    
    % Loop through subjects
    expListMouse = csv.readTable(csv.getLocation(params.subject{mm}));

    % Add "subject" to the csv table
    listFields = expListMouse.Properties.VariableNames;
    expListMouse.subject = repmat(params.subject(mm), size(expListMouse,1),1);
    expListMouse = expListMouse(:, ['subject', listFields]);

    % Add optional parameteres as new csvFields
    newfields = setdiff(fields(params), [expListMouse.Properties.VariableNames'; ...
        'timeline2Check'; 'align2Check'; 'preproc2Check'; 'issortedCheck']);
    for i = 1:length(newfields)
        expListMouse.(newfields{i}) = repmat(params.(newfields{i})(mm), height(expListMouse),1);
    end

    % Add implant info
    datNums = num2cell(datenum(expListMouse.expDate, 'yyyy-mm-dd'));
    if strcmp(params.implantDate(mm), 'none')
        expListMouse.daysSinceImplant = repmat({nan}, height(expListMouse), 1);
    else
        currImplant = datenum(params.implantDate{mm}, 'yyyy-mm-dd');
        expListMouse.daysSinceImplant = cellfun(@(x) (x-currImplant), datNums, 'uni', 0);
    end

    % Remove the expDefs that don't match
    if ~strcmp(params.expDef{mm},'all')
        expListMouse = expListMouse(contains(expListMouse.expDef, params.expDef{mm}),:);
    end
    if isempty(expListMouse); continue; end

    % Remove expNums that don't match
    if ~strcmp(params.expNum{mm},'all')
        expListMouse = expListMouse(strcmp(expListMouse.expNum, params.expNum{mm}),:);
    end
    if isempty(expListMouse); continue; end

    % Get exp with timeline only
    if params.timeline2Check{mm}==1
        expListMouse = expListMouse(str2double(expListMouse.timeline)>0,:);
    elseif params.timeline2Check{mm}==-1
        expListMouse = expListMouse(str2double(expListMouse.timeline)==0,:);
    end
    if isempty(expListMouse); continue; end

    % Get exp with specific alignment status
    alignCodeChecked = csv.checkStatusCode(expListMouse.alignBlkFrontSideEyeMicEphys,params.align2Check{mm});
    expListMouse = expListMouse(alignCodeChecked,:);
    if isempty(expListMouse); continue; end

    % Get exp with specific sorting status
    if params.issortedCheck{mm}~=-1
        expListMouse = expListMouse(ismember(expListMouse.issortedKS2, params.issortedKS2{mm}),:);
    end
    if isempty(expListMouse); continue; end

    % Get exp with specific preprocessing state
    preproc2Check = csv.checkStatusCode(expListMouse.preProcSpkEV,params.preproc2Check{mm});
    expListMouse = expListMouse(all(preproc2Check,2),:);
    if isempty(expListMouse); continue; end

    %Convert date inputs to actual dates based on the CSV data
    currDate = params.expDate{mm};
    if ~iscell(currDate); currDate = {currDate}; end
    selectedDates = arrayfun(@(x) extractDates(x, expListMouse), currDate, 'uni', 0);
    expListMouse = expListMouse(sum(cell2mat(selectedDates(:)'),2)>0,:);
    if isempty(expListMouse); continue; end

    extractedExperiments = [extractedExperiments; expListMouse];
end
end

%% This function interprets expDate input and extracts dates
function extractedDateIdx = extractDates(currDate, mouseData)
todayDate = datenum(date); %today's date
dateList = mouseData.expDate; %list of dates for current mouse csv
dateNumsCSV = datenum(dateList); %convert dates to datenums
daysSinceImplant = cell2mat(mouseData.daysSinceImplant); %get days since implant
sortedDatesCSV = unique(dateNumsCSV); %sort datenums in ascending order
datePat = '\d\d\d\d-\d\d-\d\d'; %define a date pattern of integers

for i = 1:length(currDate)
    if isnumeric(currDate{i}) && (currDate{i} < todayDate-2000 || isinf(currDate{i}))
        % If integer, n, that isn't a datenum, return exps in the last n days
        extractedDateIdx = todayDate - datenum(currDate{i}) <= dateNumsCSV;

    elseif isnumeric(currDate{i})
        % If datenum(s), return exps that match any datenum
        extractedDateIdx = ismember(dateNumsCSV, currDate{i});

    elseif strcmpi(currDate{i}(1:3), 'las')
        % If 'lastn', return the last n exps
        if numel(currDate{i})==4; currDate{i} = [currDate{i} '1']; end %#ok<*AGROW>
        lastDate = str2double(currDate{i}(5:end));
        extractedDateIdx = ismember(dateNumsCSV, sortedDatesCSV(end-min([lastDate length(sortedDatesCSV)])+1:end));
    
    elseif strcmpi(currDate{i}(1:3), 'fir')
        % If 'firstn' return the first n exps
        if numel(currDate{i})==5; currDate{i} = [currDate{i} '1']; end %#ok<*AGROW>
        lastDate = str2double(currDate{i}(5:end));
        extractedDateIdx = ismember(dateNumsCSV, sortedDatesCSV(1:min([length(sortedDatesCSV), lastDate])));
        
    elseif contains(lower(currDate{i}), ':')
        % If yyyy-mm-dd:yyyy-mm-dd, return all exps in date range (inclusive)
        currDateBounds = datenum(strsplit(currDate{i}, ':')', 'yyyy-mm-dd');
        extractedDateIdx = dateNumsCSV >= currDateBounds(1) & dateNumsCSV <= currDateBounds(2);
   
    elseif ~isempty(regexp(currDate{i}, datePat, 'once'))
        % If datestrings (yyyy-mm-dd) return exps matching any datestring
        currDateNums = datenum(regexp(currDate{i}, datePat,'match'), 'yyyy-mm-dd');
        extractedDateIdx =  ismember(dateNumsCSV, currDateNums);
        
    elseif strcmpi(currDate{i}, 'postimplant')
        % If 'postImplant', return all exps after implantation daty
        extractedDateIdx = daysSinceImplant>=0;

    elseif strcmpi(currDate{i}, 'all')
        % If 'all', return all exps
        extractedDateIdx = ones(length(dateList), 1);

    else, error('Did not understand your date input')
    end
end
end