function params = parseInputParams(paramsDef,paramsIn)
    
    fieldNamesDef = fieldnames(paramsDef);
    
    for f = 1:numel(fieldNamesDef)
        fieldName = fieldNamesDef{f};
        if isfield(paramsIn,fieldName)
            params.(fieldName) = paramsIn.(fieldName);
        else
            params.(fieldName) = paramsDef.(fieldName);
        end
    end
    