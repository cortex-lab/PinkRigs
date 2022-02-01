function events = extractEventsFromBlock(blks, eventNames)
%% Extract events from block file(s)
if ~exist('blks', 'var'); error('Must provide block(s)'); end
if ~exist('eventNames', 'var'); error('Must provide eventName(s)'); end
if ~iscell(blks); blks = {blks}; end
if ~iscell(eventNames); eventNames = {eventNames}; end

for i = 1:length(blks)
    nTrials = length(blks{i}.events.newTrialValues);
    nEnded = length(blks{i}.events.endTrialTimes);
    for j = 1:length(eventNames)
        if ~contains(eventNames{j}, {'Values'; 'Times'})
            eventNames{j} = [eventNames{j} 'Values'];
        end
        
        tDat = blks{i}.events.(eventNames{j})';
        if ~any(length(tDat) == [nTrials nEnded])
            error('Cannot parse trial-based data');
        end
        if strcmp(eventNames{j}(end-5:end), 'Values')
            events(i).(eventNames{j}(1:end-6)) = tDat(1:nEnded);
        else
            events(i).(eventNames{j}(1:end-5)) = tDat(1:nEnded);
        end
    end
end
end