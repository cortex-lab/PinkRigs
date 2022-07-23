function extractedExperiments = queryExp(varargin)
%%% This function will fetch all possible experiments to check for
%%% computing alignment, preprocessing etc.

% Assign defaults for params not contained in "inputValidation"
checkFields = {'checkTimeline', ...
    'checkAlignAny', ...
    'checkAlignEphys', ...
    'checkAlignBlock', ...
    'checkAlignCam', ...
    'checkAlignMic', ...
    'checkSorting', ...
    'checkSpikes', ...
    'checkEvents', ...
    };

defVals = cellfun(@(x) [x, {'ignore'}], checkFields, 'uni', 0);
varargin = [[defVals{:}], varargin];
params = csv.inputValidation(varargin{:});

% Loop through csv to look for experiments that weren't
% aligned, or all if recompute isn't none.
extractedExperiments = table();

alignExpFields = {'alignBlock', 'alignFrontCam', 'alignSideCam', 'alignEyeCam', 'alignMic', 'alignEphys'}';
stringExpFields = {'alignBlock', 'alignFrontCam', 'alignSideCam', 'alignEyeCam', 'alignMic', 'alignEphys'};
for mm = 1:numel(params.subject)
    
    % Loop through subjects
    mouseExps = csv.readTable(csv.getLocation(params.subject{mm}));

    % Add "subject" to the csv table
    listFields = mouseExps.Properties.VariableNames;
    mouseExps.subject = repmat(params.subject(mm), size(mouseExps,1),1);
    mouseExps = mouseExps(:, ['subject', listFields]);

    % Add optional parameteres as new csvFields
    newfields = setdiff(fields(params), [mouseExps.Properties.VariableNames'; checkFields']);
    for i = 1:length(newfields)
        mouseExps.(newfields{i}) = repmat(params.(newfields{i})(mm), height(mouseExps),1);
    end

    % Add implant info
    datNums = num2cell(datenum(mouseExps.expDate, 'yyyy-mm-dd'));
    if strcmp(params.implantDate(mm), 'none')
        mouseExps.daysSinceImplant = repmat({nan}, height(mouseExps), 1);
    else
        currImplant = datenum(params.implantDate{mm}, 'yyyy-mm-dd');
        mouseExps.daysSinceImplant = cellfun(@(x) (x-currImplant), datNums, 'uni', 0);
    end

    % Remove the expDefs that don't match
    if ~strcmp(params.expDef{mm},'all')
        mouseExps = mouseExps(contains(mouseExps.expDef, params.expDef{mm}),:);
    end
    if isempty(mouseExps); continue; end

    % Remove expNums that don't match
    if ~strcmp(params.expNum{mm},'all')
        mouseExps = mouseExps(strcmp(mouseExps.expNum, params.expNum{mm}),:);
    end
    if isempty(mouseExps); continue; end

    % Get exp with timeline only
    if ~strcmp(params.checkTimeline{mm}, 'ignore')
        mouseExps = mouseExps(mouseExps.timeline==params.checkTimeline{mm});
    end
    if isempty(mouseExps); continue; end

    % Get exp with specific alignment status
    alignCheckVals = inf(6, 1);
    if ~strcmp(params.checkAlignBlock{mm}, 'ignore')
        alignCheckVals(1) = params.checkAlignBlock{mm};
    end
    if ~strcmp(params.checkAlignCam{mm}, 'ignore')
        alignCheckVals(2:4) = params.checkAlignCam{mm};
    end
    if ~strcmp(params.checkAlignMic{mm}, 'ignore')
        alignCheckVals(5) = params.checkAlignMic{mm};
    end
    if ~strcmp(params.checkAlignEphys{mm}, 'ignore')
        alignCheckVals(6) = params.checkAlignEphys{mm};
    end

    for i = find(~isinf(alignCheckVals(i)))
        mouseExps = mouseExps(mouseExps.(alignExpFields(i))==alignCheckVals(i));
    end

    % Get exp with specific sorting status
    if ~strcmp(params.checkSorting{mm}, 'ignore')
%         sortingVals = cellfun(@(x, y), mouseExps.issortedKS2
%         mouseExps = mouseExps(any(mouseExps.issortedKS2), params.issortedKS2{mm}),:);
    end
    if isempty(mouseExps); continue; end

    % Get exp with specific preprocessing state
    preproc2Check = csv.checkStatusCode(mouseExps.preProcSpkEV,params.preproc2Check{mm});
    mouseExps = mouseExps(all(preproc2Check,2),:);
    if isempty(mouseExps); continue; end

    %Convert date inputs to actual dates based on the CSV data
    currDate = params.expDate{mm};
    if ~iscell(currDate); currDate = {currDate}; end
    selectedDates = arrayfun(@(x) extractDates(x, mouseExps), currDate, 'uni', 0);
    mouseExps = mouseExps(sum(cell2mat(selectedDates(:)'),2)>0,:);
    if isempty(mouseExps); continue; end

    extractedExperiments = [extractedExperiments; mouseExps];
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
        lastDate = str2double(currDate{i}(6:end));
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