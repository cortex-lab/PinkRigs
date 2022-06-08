function filtered = filterStructRows(unfiltered, criterion)
filtered = unfiltered;
fieldNames = fields(unfiltered);
for fieldName = fieldNames'
    
    if isstruct(unfiltered.(fieldName{1}))
        filtered.(fieldName{1}) = filterStructRows(unfiltered.(fieldName{1}), criterion);
    else, filtered.(fieldName{1}) = unfiltered.(fieldName{1})(criterion,:,:);
    end
end
end
