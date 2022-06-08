function outP = inputValidation(varargin)
% Get active mouse list from main csv
%%% INPUTS
%   Inputs can take the form of a structure (e.g. params.subject = 'AV009')
%   or name-value pairs (e.g. subject=AV009 or 'subject', 'AV009'). There
%   are four parameters that will always be returned with defaults if they
%   are not provided: "subject", "expDate", "expDef" and "expNum". 

%   "subject" is the field that all other inputs are measured against such
%   that every input of length "1" will be repeated to match the length of
%   "subject"

%   For single inputs, these can typically be strings, a cell of strings,
%   or integers. If you want to have a different input (e.g. date range)
%   for each subject, you MUST input a cell array that is the same length
%   as your "subject" cell array. Otherwise, you will receive an error.

%   NOTE: this function can deal with multiple inputs for each subject (for
%   example, if you wanted multiple expDefs for each subject) BUT BE 
%   CAREFUL with cell formatting. To illustrate this, consider these cases:
%   (1) .subject = {'AV009'; 'AV007'} and .expDef({'t'; 's'})
%       The expDef input is 't' for  'AV009' and 's' for 'AV007'
%   (2) .subject = {'AV009'; 'AV007'} and .expDef({{'t'; 's'}})
%       The expDef input is 't' AND 's' for  'AV009' AND 'AV007'
%   (3) .subject = {'AV009'; 'AV007'} and .expDef({{'t'; 's'}; 't'})
%       The expDef input is 't' AND 's' for  'AV009', BUT 't' for 'AV007'

p = inputParser;
p.KeepUnmatched=true;

def_subject = {'active'};
def_expDate = {'all'};
def_expNum = {'all'};
def_expDef = {'all'};

tblDat = varargin(cellfun(@istable, varargin));
tblDatConverted = cellfun(@(x) table2struct(x, 'ToScalar', 1), tblDat, 'uni', 0);
varargin(cellfun(@istable, varargin)) = tblDatConverted;

addParameter(p, 'subject', def_subject, @isStringOrCellOfStrings)
addParameter(p, 'expDate', def_expDate)
addParameter(p, 'expNum', def_expNum)
addParameter(p, 'expDef', def_expDef)

if ~all(cellfun(@isempty, varargin)); parse(p, varargin{:});
else, parse(p)
end

outP = p.Results;
outP = orderfields(outP, {'subject'; 'expDate'; 'expNum'; 'expDef'});
unmatchedFields = fields(p.Unmatched);
for i = 1:length(unmatchedFields)
    outP.(unmatchedFields{i}) = p.Unmatched.(unmatchedFields{i});
end

%%%% FOR BACKWARDS COMPATIBILITY. SHOULD BE REMOVED IN TIME
fieldTest = {'mice2Check', 'expDef2Check', 'days2Check'};
fieldReplace = {'subject', 'expDef', 'expDate'};
for i = 1:length(fieldTest)
    if isfield(outP, fieldTest{i})
        outP.(fieldReplace{i}) = outP.(fieldTest{i});
    end
end
%%%%

outP = structfun(@mkCell, outP, 'uni', 0);

mainCSVLoc = csv.getLocation('main');
mouseList = csv.readTable(mainCSVLoc);

%Validate subjects and deal with "all" and "active" cases for subject selection
if strcmp(outP.subject{1},'active')
    outP.subject = mouseList.Subject(strcmp(mouseList.IsActive,'1'));
elseif strcmp(outP.subject{1},'all')
    outP.subject = mouseList.Subject;
end
if ~all(ismember(outP.subject, mouseList.Subject))
    error('Unrecognized mouse names!')
end

%Make all fields the same length as "subject"
nSubjects = length(outP.subject);
paramLengths = structfun(@length, outP);
if ~all(paramLengths == nSubjects | paramLengths == 1)
    error('All inputs should have length=1 or length=nSubjects')
end
fieldNames = fields(outP);
for i = 1:length(fieldNames)
    if paramLengths(i) == nSubjects
        outP.(fieldNames{i}) = outP.(fieldNames{i})(:);
        continue; 
    end
    outP.(fieldNames{i}) = repmat(outP.(fieldNames{i}),nSubjects,1);
end

outP.expDef = cellfun(@string2expDef, outP.expDef, 'uni', 0);
outP.expDef = cellfun(@(x) vertcat(x{:}), outP.expDef, 'uni', 0);

outP.expNum = cellfun(@convertCellNum2String, outP.expNum, 'uni', 0);
outP.expNum = cellfun(@(x) vertcat(x{:}), outP.expNum, 'uni', 0);
end

function cellOutput = mkCell(maybeCell)
cellOutput = maybeCell;
if ~iscell(maybeCell)
    cellOutput = {cellOutput};
end
end

function validInput  = isStringOrCellOfStrings(testInput)
if iscell(testInput)
    validInput = all(cellfun(@ischar, testInput));
else
    validInput = ischar(testInput);
end
end

function fullExpDef  = string2expDef(expDefInput)
expDefInput = mkCell(expDefInput);

fullExpDef = cell(length(expDefInput),1);
for i = 1:length(expDefInput)
switch expDefInput{i}
    case 'all'
        fullExpDef{i} = {'all'};
    case {'t', 'train', 'training', 'behaviour'}
        fullExpDef{i} = {'multiSpaceWorld_checker_training'; 'multiSpaceWorld_checker'};
    case {'p', 'passive', 'pass'}
        fullExpDef{i} = {'AVPassive_ckeckerboard_postactive'};
    case {'s', 'spont', 'spontaneous'}
        fullExpDef{i} = {'spontaneousActivity'};
    case {'i', 'image', 'nat', 'naturalimages'}
        fullExpDef{i} = {'imageWorld_AllInOne'};
    case {'n', 'sparse', 'sparsenoise'}
        fullExpDef{i} = {'sparseNoise'; 'AP_sparseNoise'};
    otherwise 
        fullExpDef{i} = expDefInput(i);
end
end
end

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