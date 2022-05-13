function params = addDefaultParam(params, newField, newValue)
nSubjects = length(params.subject);
if ~iscell(newValue); newValue = {newValue}; end
if ~isfield(params, newField)
    params.(newField) = repmat(newValue,nSubjects,1); 
end
end