function filtered = filterStructRows(unfiltered, criterion)
    %% Filters rows of a structure based on the logical "criterion"
    %  WARNING: Fields that don't have the same length as criterion will be
    %  ignored!
    %
    % Parameters:
    % -------------------
    % unfiltered: struct
    %   Structure to filter
    % criterion: indices
    %   A specific criterion on which to filter the above structure.
    %
    % Returns: 
    % -------------------
    % filtered: struct
    %   Filtered structure.

    filtered = unfiltered;
    fieldNames = fields(unfiltered);
    for fieldName = fieldNames'
        if isstruct(unfiltered.(fieldName{1}))
            filtered.(fieldName{1}) = filterStructRows(unfiltered.(fieldName{1}), criterion);
        else
            if length(unfiltered.(fieldName{1}))==length(criterion)
                filtered.(fieldName{1}) = unfiltered.(fieldName{1})(criterion,:,:);
            else
                filtered.(fieldName{1}) = unfiltered.(fieldName{1});
            end
        end
    end
end
