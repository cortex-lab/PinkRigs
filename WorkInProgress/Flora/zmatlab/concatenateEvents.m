function [ev] = concatenateEvents(evCell)
% evCell is a cell of structs that contain envents where each cell's structs fields must be the same
names = fieldnames(evCell{1});
for k=1:numel(names)
    a = {horzcat(evCell{:}).(names{k})};
    ev.(names{k}) = vertcat(a{:});   
end
end 