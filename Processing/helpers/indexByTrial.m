function sortedByTrial = indexByTrial(trialTimes, prmTimes, prmValues, timesToSubtract)
%% A helper function to split variables into cells that contain all the values for that trial
%Inputs(default)
%trialTimes               (required) is an nx2 vector of the form [trialStartTimes, trialEndTimes];
%prmTimes                 (required) is an nx1 vector of times for the values that are to be split by trial.
%prmValues                (prmTimes) is an nxm vector of values that should be split.
%timesToSubtract          (0*nx1) is a nx1 of times to subtract from the paramters (e.g. if you wanted timings relative to the stimulus onset)

%Outputs
%sortedByTrial is a cell array of the prmValues, sorted by the trialTimes they occured between (each cell is one trial) 

%%
%Set default values
if ~exist('prmValues', 'var') || isempty(prmValues); prmValues = prmTimes; end
if ~all(diff(prmTimes(~isnan(prmTimes)))>=0) 
%     warning('Input times are not monotonic so will be sorted');
    [~, srtIdx] = sort(prmTimes);
    prmTimes = prmTimes(srtIdx, :);
    prmValues = prmValues(srtIdx, :);    
end

%Use histcounts to find all the times that fall between trial start and end times--this is a computationally fast way to do this. We remove indices
%with 0 values because these are out of bounds. 
[eventCount, ~, trialIdx] = histcounts(prmTimes, sort([trialTimes(:,1);trialTimes(:,2)+realmin]));
prmValues(trialIdx==0,:) = []; trialIdx(trialIdx==0) = [];

%idxBounds finds the bounds where the trialIdx starts and ends, then removes all even indices are these lie between trials (after end and before
%start. We also remove the eventcounts corresponding to these inter-trial spaces
idxBounds = [find(diff([-10;trialIdx])>0) find(diff([trialIdx;1e6])>0)];
idxBounds(mod(unique(trialIdx),2)==0,:) = [];
eventCount(2:2:end) = [];

%Get the unique trial indices that aren't inter-trial spaces and pre-populate the sortedByTrial cell array according to this.
idx = 0;
sortedByTrial = cell(length(eventCount),1);
if ~exist('timesToSubtract', 'var'); timesToSubtract = repmat(0*prmValues(:,1), length(eventCount),1); end
for i = 1:length(eventCount)
    if eventCount(i)==0; continue; else; idx = idx+1; end
    subtractValues = repmat(timesToSubtract(i,:), eventCount(i), 1);
    sortedByTrial{i} = prmValues(idxBounds(idx,1):idxBounds(idx,2),:);
    if isempty(sortedByTrial{i}); continue; end
    sortedByTrial{i} = sortedByTrial{i} - subtractValues;
end
end