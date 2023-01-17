function outP = inputValidation(varargin)
%% Validates inputs formany functions (so they all use the same format)
%
% Parameters:
% ------------
% 
% NOTE: Inputs can take the form of a structure (e.g. params.subject = 'AV009')
% or name-value pairs (e.g. subject=AV009 or 'subject', 'AV009'). There
% are four parameters that will always be returned with defaults if they
% are not provided: "subject", "expDate", "expDef" and "expNum". 
%
% subject (default = 'active'): string (e.g. 'all' or 'AV009')
% ----The subject(s) to be included
% ----NOTE:This is the field that all other inputs are measured against.
% ----Every input of length "1" is repeated to match the length of "subject"
% ----For single inputs, these can typically be strings, a cell of strings,
% ----or integers. If you want to have a different input (e.g. date range)
% ----for each subject, you MUST input a cell array that is the same length
% ----as your "subject" cell array. Otherwise, you will receive an error.
% ----
%
% expDate (default = 'all'): string or integer (e.g. 'all' or '2022-11-12')
% ----The experiment date(s) to be included
% ----NOTE: for a all possible formats, see the "extractDates" function
% ----within csv.queryExp
% ----
%
% expNum (default = 'all'): string or integer (e.g. '1' or 1)
% ----The experiment number(s) to to be included
% ----
%
% expDef (default = 'all'): string or cell of strings (e.g. 's' or 'sparseNoise')
% ----The experiment definition(s) to to be included
% NOTE: there are some handy shortcuts (e.g. 't' for all training
% functions). Thse can be seen in the "string2expDef" function contained
% within this function
% ----
%
% ----NOTE: this function can deal with multiple inputs for each subject (for
% ----example, if you wanted multiple expDefs for each subject) BUT BE 
% ----CAREFUL with cell formatting. To illustrate this, consider these cases:
% ----(1) .subject = {'AV009'; 'AV007'} and .expDef({'t'; 's'})
% ----The expDef input is 't' for  'AV009' and 's' for 'AV007'
% ----(2) .subject = {'AV009'; 'AV007'} and .expDef({{'t'; 's'}})
% ----The expDef input is 't' AND 's' for  'AV009' AND 'AV007'
% ----(3) .subject = {'AV009'; 'AV007'} and .expDef({{'t'; 's'}; 't'})
% ----The expDef input is 't' AND 's' for  'AV009', BUT 't' for 'AV007'
%
% Returns: 
% ---------------
%
% outP: stuct of cell arrays
% ----For each structure field, there should be a cell for each subject

% Set up an inputPArser object and ensures that all inputs (not just those
% with defaults) are kept
p = inputParser;
p.KeepUnmatched = true;

% Default values for subject, expDate, expNum, and expDef
def_subject = {'active'};
def_expDate = {'all'};
def_expNum = {'all'};
def_expDef = {'all'};

% If the input is a table (e.g. the output of csv.queryExp) then convert
% the table to a structure where each column is a field
tblDat = varargin(cellfun(@istable, varargin));
if numel(tblDat)>0 && isempty(tblDat{1})
    % Case when an empty list of exp is given.
    def_subject = {'none'};
end
tblDatConverted = cellfun(@(x) table2struct(x, 'ToScalar', 1), tblDat, 'uni', 0);
varargin(cellfun(@istable, varargin)) = tblDatConverted;

% Set the default values for the 4 fields. "isStringOrCellOfStrings" is a
% validation function that checks the input is a string or cell of strings
addParameter(p, 'subject', def_subject, @isStringOrCellOfStrings)
addParameter(p, 'expDate', def_expDate)
addParameter(p, 'expNum', def_expNum)
addParameter(p, 'expDef', def_expDef)

% If varargin isn't empty, parse it along with the inputParser object ("p")
% This is will overwrite the defaults if inputs are suplied
if ~all(cellfun(@isempty, varargin)); parse(p, varargin{:});
else, parse(p)
end

% Extract and organize the results of the input parser such that the 4
% default fields always appear first. Add optional inputs (unmatched
% fields) to create a single strcuture with default and optional inputs.
outP = p.Results;
outP = orderfields(outP, {'subject'; 'expDate'; 'expNum'; 'expDef'});
unmatchedFields = fields(p.Unmatched);
for i = 1:length(unmatchedFields)
    if strcmpi(unmatchedFields{i}, 'invariantParams')
        invariantParams = unnestCell(p.Unmatched.(unmatchedFields{i}));
    else
    outP.(unmatchedFields{i}) = p.Unmatched.(unmatchedFields{i});
    end
end
if ~exist('invariantParams', 'var'); invariantParams = {'svloiubsverilvub'}; end

% This runs the "mkCell" function on all fields of "outP" to convert to
% cells if they aren't already cells.
outP = structfun(@mkCell, outP, 'uni', 0);

% Get location of the main csv and load it 
if ~isfield(outP, 'mainCSV') || isempty(outP.mainCSV)
    mainCSVLoc = csv.getLocation('main');
    mainCSV = csv.readTable(mainCSVLoc);
else
    mainCSV = outP.mainCSV{1};
end

% Validate subjects and interpret input optional input strings
if isempty(outP.subject)
    % No subject. Just return.
    return
end
if strcmp(outP.subject{1},'active')
    % All “active” mice in the main csv
    outP.subject = mainCSV.Subject(strcmp(mainCSV.IsActive,'1'));
elseif strcmp(outP.subject{1},'implanted')
    % All mice with an implant-date with a probe in the main csv
    implanted = cellfun(@(x) ~isempty(regexp(x,'\d\d\d\d-\d\d-\d\d', 'once')), mainCSV.P0_implantDate);
    implanted(strcmpi(mainCSV.P0_type, 'P3B')) = 0;
    outP.subject = mainCSV.Subject(implanted);
elseif strcmp(outP.subject{1},'all')
    % All mice in the main csv
    outP.subject = mainCSV.Subject;
elseif strcmp(outP.subject{1},'none')
    outP.subject = mainCSV.Subject([],:);
end

% Check that all "subjects" exist in the main csv. If not, error
if ~all(ismember(outP.subject, [mainCSV.Subject]))
    error('Unrecognized mouse names!')
end
P0_implants = mainCSV.P0_implantDate;
implantDates = cellfun(@(x) P0_implants{strcmp(mainCSV.Subject, x)}, outP.subject, 'uni', 0);
validImplants = cellfun(@(x) ~isempty(regexp(x,'\d\d\d\d-\d\d-\d\d', 'once')), implantDates);
implantDates(~validImplants) = deal({'none'});
outP.implantDate = implantDates;

% Check the length 
nSubjects = length(outP.subject);
outP.mainCSV = repmat({mainCSV}, nSubjects,1); % saves loading time later;
paramLengths = structfun(@length, outP);

% NOTE: all inputs must have length=length(subjects) or length=1. Otherwise
% throw an error. This is usually because an input intended for a single
% (or every) subject, e.g. {'t'; 's'} has been input as multiple cells
% instead of a single cell. In this example, {{'t';'s}} might be the
% correct input.
invariantParamsIdx = contains(fields(outP), invariantParams);
if ~all(paramLengths == nSubjects | paramLengths == 1 | invariantParamsIdx)
    error('All inputs should have length=1 or length=nSubjects')
end

% Make all fields the same length as "subject" field by replicating cases
% where a field has length=1 and subjects has length>1.
fieldNames = fields(outP);
for i = 1:length(fieldNames)
    if invariantParamsIdx(i)
        outP.(fieldNames{i}) = repmat({unnestCell(outP.(fieldNames{i}),0)},nSubjects,1);
        continue;
    elseif paramLengths(i) == nSubjects
        outP.(fieldNames{i}) = outP.(fieldNames{i})(:);
        continue; 
    end
    outP.(fieldNames{i}) = repmat(outP.(fieldNames{i}),nSubjects,1);
end

% Use the funtion "string2expDef" to convert optional input strings (e.g.
% 't' or 's') into their corresponding exp definitions
outP.expDef = cellfun(@string2expDef, outP.expDef, 'uni', 0);

% Use "convertCellNum2String" to make sure that expNum is a string or
% vertically concatenated cell of strings
outP.expNum = cellfun(@convertCellNum2String, outP.expNum, 'uni', 0);
outP.expNum = cellfun(@(x) vertcat(x{:}), outP.expNum, 'uni', 0);
end

%% Function to convert the input into a cell if it isn't already a cell
function cellOutput = mkCell(maybeCell)
cellOutput = maybeCell;
if ~iscell(maybeCell)
    cellOutput = {cellOutput};
end
end

%% Function to return "true" if the input is a string or cell of strings
function validInput  = isStringOrCellOfStrings(testInput)
if iscell(testInput)
    validInput = all(cellfun(@ischar, testInput));
else
    validInput = ischar(testInput);
end
end

%% Function that returns a set of exp definitions based on optional strings
function fullExpDef  = string2expDef(expDefInput)
%%% POSSIBLE INPUTS:
% 	't' or 'train' or 'training' or 'behaviour'—training expDefs
% 		{'multiSpaceWorld_checker_training'; 'multiSpaceWorld_checker'}
% 	'p' or  'passive' or 'pass'—passive AV stimuli expDefs
% 		{'AVPassive_ckeckerboard_postactive'; 'AVPassive_checkerboard_extended'}
% 	's' or 'spont' or 'spontaneous'—Spontaneous activity expDefs
% 		{'spontaneousActivity'}
% 	'i' or 'image' or 'nat' or 'naturalimages'—natural image presentation expDefs
% 		{'imageWorld_AllInOne'}
% 	'n' or 'sparse' or 'sparsenoise'—Sparse noise presentation expDefs
% 		{'sparseNoise'; 'AP_sparseNoise'}
% 	‘XXXX’—Anything that isn’t matched above will return the same string

% Make sure that the input is a cell
expDefInput = mkCell(expDefInput);
fullExpDef = cell(length(expDefInput),1);

% Match the input with possible pre-defined strings and convert to the
% appropriate exp definitions. Otherwise, return the original input.
for i = 1:length(expDefInput)
switch expDefInput{i}
    case 'all'
        fullExpDef{i} = {'all'};
    case {'t', 'train', 'training', 'behaviour'}
        fullExpDef{i} = {'multiSpaceWorld_checker_training'; 'multiSpaceWorld_checker'};
    case {'p', 'passive', 'pass'}
        fullExpDef{i} = {'AVPassive_ckeckerboard_postactive'; 'AVPassive_checkerboard_extended'};
    case {'s', 'spont', 'spontaneous'}
        fullExpDef{i} = {'spontaneousActivity'};
    case {'i', 'image', 'nat', 'naturalimages'}
        fullExpDef{i} = {'imageWorld_AllInOne','imageWorld_lefthemifield','imageWorld2'};
    case {'n', 'sparse', 'sparsenoise'}
        fullExpDef{i} = {'sparseNoise'; 'AP_sparseNoise'};
    otherwise 
        fullExpDef{i} = expDefInput(i);
end
end
fullExpDef = unnestCell(fullExpDef);
end

%% Function to convert a numeric input into a string or cell of strings
function  out = convertCellNum2String(in)
if iscell(in)
    out = cellfun(@convertCellNum2String, in, 'uni', 0);
elseif isnumeric(in)
    out = cellfun(@num2str, num2cell(in), 'uni',0);
elseif ischar(in)
    out = {in};
end
out = out(:);
end