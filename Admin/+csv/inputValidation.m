%% Get parameters
if ~exist('params', 'var') || isempty(params); params = struct; end

%Validate subject input and check it is a cell
if ~exist('subject', 'var'); params.subject = 'active'; 
else, params.subject = subject;
end
if ~iscell(params.subject); params.subject = {params.subject}; end

%Validate expDate input and check it is a cell
if ~exist('expDate', 'var'); params.expDate = 'inf'; 
else, params.expDate = expDate;
end
if ~iscell(params.expDate); params.expDate = {params.expDate}; end

%Validate expDef input and check it is a cell
if ~exist('expDef', 'var'); params.expDef = 'all'; 
else, params.expDef = expDef;
end
if ~iscell(params.expDef); params.expDef = {params.expDef}; end

%Assign defaults for remaining params and check they aren't cells
if ~isfield('params', 'timeline2Check'); params.timeline2Check = {0}; end
if ~isfield('params', 'align2Check'); params.align2Check = {'*,*,*,*,*,*'}; end
if ~isfield('params', 'preproc2Check'); params.preproc2Check = {'*,*'}; end

if ~iscell(params.timeline2Check); params.timeline2Check = {params.timeline2Check}; end
if ~iscell(params.align2Check); params.align2Check = {params.align2Check}; end
if ~iscell(params.preproc2Check); params.preproc2Check = {params.preproc2Check}; end

fieldNames = fields(params); 
standardFields = {...
    'subject';...
    'expDate';...
    'expDef';...
    'timeline2Check';...
    'align2Check';...
    'preproc2Check'};
extraFields = fieldNames(~contains(fieldNames, standardFields));
params = orderfields(params, [standardFields; extraFields]);

paramLengths = structfun(@length, params);
if ~all(paramLengths == paramLengths(1) | 1)
    error('All params should be length one or same as number of mice')
end

for i = 1:length(fieldNames)
    if paramLengths(i) == paramLengths(1); continue; end
    params.(fieldNames{i}) = repmat(params.(fieldNames{i}),paramLengths(1),1);
end

params.mice2check = params.subject;
params.days2check = params.expDate;
params.expDef2check = params.expDef;