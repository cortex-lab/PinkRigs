%% funciton to unnest a cell
function newCell = unnestCell(nestedCell, complete)
if ~exist('complete', 'var'); complete = 1; end
if ~iscell(nestedCell); nestedCell = {nestedCell}; end

cellLengths = cellfun(@(x) iscell(x) & length(x)==1, nestedCell);
while (all(cellLengths) || complete) && any(cellfun(@iscell, nestedCell))
    newCell = {};
    for i = 1:length(nestedCell)
        if iscell(nestedCell{i})
            newCell = [newCell; nestedCell{i}];
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
