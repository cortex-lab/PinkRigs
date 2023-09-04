function extractedExperiments = queryExp(varargin)
%% Fetch experiments according to predefined filters
% 
% NOTE: This function uses csv.inputValidate to parse inputs. Paramters are 
% name-value pairs, including those specific to this function
%
% Parameters: 
% ---------------
% Classic PinkRigs inputs (optional)
%
% checkTimeline (default='ignore'): string 
% checkAlignAny (default='ignore'): string 
% checkAlignEphys (default='ignore'): string 
% checkAlignBlock (default='ignore'): string 
% checkAlignCam (default='ignore'): string 
% checkAlignMic (default='ignore'): string 
% checkSorting (default='ignore'): string 
% checkSpikes (default='ignore'): string 
% checkEvents (default='ignore'): string 
%   In all the above cases, the relevant fields will be checked, and the
%   user can input a filter to match (e.g. 'checkEvents'='1' will return
%   experiments with successful event extraction) or a filter to exclude
%   by using the '~' symbol (e.g. 'checkEvents'='~nan' will return all
%   experiments without expected errors in event extraction)
%
% expDate (default='all')
%   There are many options for expDate to select dates, ranges etc. These
%   Are described in the "extractDates" function (within this function)
%
% Returns: 
% -----------
% extractedExperiments: table 
%   Table with all the csv information for the selected experiments
%
% Examples: 
% ------------
% csv.queryExp('subject', 'AV008', 'expDate', 'last10', 'expDef', 'training'});
% csv.queryExp('subject', 'AV008', 'expDate', '2022-11-11:2022-11-31' 'checkSpikes', '1'});


% Assign defaults for params not contained in "inputValidation"
checkFields = {'rig', ...
    'checkTimeline', ...
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

for mm = 1:numel(params.subject)
    
    % Loop through subjects
    if ~exist(csv.getLocation(params.subject{mm}), 'file'); continue; end
    mouseExps = csv.readTable(csv.getLocation(params.subject{mm}));
    csvHeaders = mouseExps.Properties.VariableNames';


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
    if isempty(mouseExps); continue; end
    datNums = num2cell(datenum(mouseExps.expDate, 'yyyy-mm-dd'));
    if strcmp(params.implantDate(mm), 'none')
        mouseExps.daysSinceImplant = repmat({nan}, height(mouseExps), 1);
    else
        currImplant = datenum(params.implantDate{mm}, 'yyyy-mm-dd');
        mouseExps.daysSinceImplant = cellfun(@(x) (x-currImplant), datNums, 'uni', 0);
    end

    % Remove the expDefs that don't match
    if ~strcmp(params.expDef{mm},'all')
        mouseExps = mouseExps(cellfun(@(x) any(strcmpi(params.expDef{mm},x)), mouseExps.expDef),:);
    end
    if isempty(mouseExps); continue; end

    % Remove expNums that don't match
    if ~strcmp(params.expNum{mm},'all')
        mouseExps = mouseExps(strcmp(mouseExps.expNum, params.expNum{mm}),:);
    end
    if isempty(mouseExps); continue; end

    % Remove rigs that don't match
    if ~strcmp(params.rig{mm}, 'ignore')
        if ~strcmp(params.rig{mm},'all')
            mouseExps = mouseExps(contains(mouseExps.rigName, params.rig{mm}),:);
        end
        if isempty(mouseExps); continue; end
    end

    % Get exp with matching timeline status
    if ~strcmp(params.checkTimeline{mm}, 'ignore')
        chkVal = num2str(params.checkTimeline{mm});
        if strcmpi(chkVal(1), '~')
            mouseExps = mouseExps(~contains(mouseExps.existTimeline, chkVal(2:end)),:);
        else
            mouseExps = mouseExps(contains(mouseExps.existTimeline, chkVal),:);
        end
    end
    if isempty(mouseExps); continue; end

    % Get exp with specific alignment status
    alignFields = csvHeaders(contains(csvHeaders, 'align'));
    chkVals = cell(6,1);
    chkVals{1} = num2str(params.checkAlignBlock{mm});
    chkVals(2:5) = {num2str(params.checkAlignCam{mm})};
    chkVals{6} = num2str(params.checkAlignMic{mm});
    chkVals{7} = num2str(params.checkAlignEphys{mm});
    keepIdx = zeros(height(mouseExps),1)>0;
    for i = find(~contains(chkVals, 'ignore'))'
        if isempty(i); continue; end
        if strcmpi(chkVals{i}(1), '~')
            keepIdx = keepIdx + ~contains(mouseExps.(alignFields{i}), chkVals{i}(2:end));
        else
            keepIdx = keepIdx + contains(mouseExps.(alignFields{i}), chkVals{i});
        end
        if i ~= 2 && i ~=3
            mouseExps = mouseExps(keepIdx>0,:);
            keepIdx = zeros(height(mouseExps),1)>0;
        end
    end
    if isempty(mouseExps); continue; end

    if ~strcmp(params.checkAlignAny{mm}, 'ignore')
        chkVal = num2str(params.checkAlignAny{mm});
        % Since alignMic is currently always 2, ignore if looking for 2's
        if strcmpi(chkVal, '2'); alignFields = alignFields([1:4,6]); end
        combAlign = cellfun(@(x) mouseExps.(x), alignFields, 'uni', 0);
        combAlign = [combAlign{:}];
        combAlign = arrayfun(@(x) cell2mat(combAlign(x,:)), 1:size(combAlign,1), 'uni', 0)';
        if strcmpi(chkVal(1), '~')
            mouseExps = mouseExps(~contains(combAlign, chkVal(2:end)),:);
        else
            mouseExps = mouseExps(contains(combAlign, chkVal),:);
        end
    end
    if isempty(mouseExps); continue; end

    % Get exp with specific sorting status
    sortFields = csvHeaders(contains(csvHeaders, 'issorted'));
    if ~strcmp(params.checkSorting{mm}, 'ignore')
        chkVal = num2str(params.checkSorting{mm});
        combSort = cellfun(@(x) mouseExps.(x), sortFields, 'uni', 0);
        combSort = [combSort{:}];
        combSort = arrayfun(@(x) cell2mat(combSort(x,:)), 1:size(combSort,1), 'uni', 0)';
        if strcmpi(chkVal(1), '~')
            mouseExps = mouseExps(~contains(combSort, chkVal(2:end), 'ignorecase', 1),:);
        else
            mouseExps = mouseExps(contains(combSort, chkVal),:);
        end
    end
    if isempty(mouseExps); continue; end

    % Get exp with any extractSpikess matching input state
    if ~strcmp(params.checkSpikes{mm}, 'ignore')
        chkVal = num2str(params.checkSpikes{mm});
        if strcmpi(chkVal(1), '~')
            mouseExps = mouseExps(~contains(mouseExps.extractSpikes, chkVal(2:end)),:);
        else
            mouseExps = mouseExps(contains(mouseExps.extractSpikes, chkVal),:);
        end
    end
    if isempty(mouseExps); continue; end

    % Get exp with any extractEventss matching input state
    if ~strcmp(params.checkEvents{mm}, 'ignore')
        chkVal = num2str(params.checkEvents{mm});
        if strcmpi(chkVal(1), '~')
            mouseExps = mouseExps(~contains(mouseExps.extractEvents, chkVal(2:end)),:);
        else
            mouseExps = mouseExps(contains(mouseExps.extractEvents, chkVal),:);
        end
    end
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