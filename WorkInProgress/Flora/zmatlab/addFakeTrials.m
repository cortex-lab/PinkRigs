function [outEV] = addFakeTrials(events)

[visGrid, audGrid,respGrid] = meshgrid(unique(events.stim_visDiff),unique(events.stim_audDiff),unique(events.response_direction));
[v,a,r] = deal(reshape(visGrid,[],1),reshape(audGrid,[],1),reshape(respGrid,[],1));

% 

% create a new block with everything 0 that is of the length that you want
% 
% for n=1:numel(v)
%     
% end 

n = size(v,1); 
names = fieldnames(events);
for k=1:numel(names)
    % for each ev 
    event_tagged.(names{k}) = zeros(n,1); 
end
event_tagged.stim_visDiff = v; 
event_tagged.stim_audDiff = a; 
event_tagged.response_direction = r; 

% concantenate with events
dataEvents = {events;event_tagged}; 
for k=1:numel(names)
    a = {horzcat(dataEvents{:}).(names{k})};
    outEV.(names{k}) = vertcat(a{:});   
end

end 