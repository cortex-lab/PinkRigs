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
            if length(tDat)== sum(blks{i}.events.repeatNumValues==1)
                repeatNums = blks{i}.events.repeatNumValues==1;
                tDat = repeatNums(:)*nan;
                tDat(repeatNums==1) = blks{i}.events.(eventNames{j})';
                repeatIdx = find(repeatNums~=1);
                repeatedIdx = arrayfun(@(x) find(~isnan(tDat(1:x)),1,'last'),repeatIdx);
                tDat(repeatIdx) = tDat(repeatedIdx);
            else
                error('Cannot parse trial-based data');
            end
        end
        if strcmp(eventNames{j}(end-5:end), 'Values')
            events(i,1).(eventNames{j}(1:end-6)) = tDat(1:nEnded);
        else
            events(i,1).(eventNames{j}(1:end-5)) = tDat(1:nEnded);
        end
    end
end
end