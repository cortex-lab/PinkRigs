function newCell = unnestCell(nestedCell, complete)
%% Will unnest a nested cell array
%
% NOTE: An example of a "nested" cell would be {{{1}}}. If this was
% completely un-nested, it would become {1} and if it was un-nested by 1
% "level" it would become {{1}}
%
% Parameters:
% ------------
% nestedCell (required): Nested (potentially) cell array
%   A cell array that the user wants to "un-nest"
%
% complete (default=1): logical
%   If 1, will completely un-nest a cell array, but if 0, will only un-nest
%   by 1 "level"
%  
% 
% Returns: 
% -----------
% newCell: cell array
%   The un-nested cell array (complete, or by 1 "level"

if ~exist('complete', 'var'); complete = 1; end
if ~iscell(nestedCell); nestedCell = {nestedCell}; end

cellLengths = cellfun(@(x) iscell(x) & length(x)==1, nestedCell);
while (all(cellLengths) || complete) && any(cellfun(@iscell, nestedCell))
    newCell = {};
    for i = 1:length(nestedCell)
        if iscell(nestedCell{i})
            newCell = [newCell; nestedCell{i}(:)];
        else
            newCell = [newCell; nestedCell(i)];
        end
    end
    nestedCell = newCell;
    cellLengths = cellfun(@(x) iscell(x) & length(x)==1, nestedCell);
end
newCell = nestedCell;
if length(newCell) == 1; newCell = newCell{1}; end
if ~iscell(newCell); newCell = {newCell}; end
end
