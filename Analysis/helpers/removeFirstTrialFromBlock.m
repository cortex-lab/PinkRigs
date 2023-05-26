function [truncatedBlock, truncatedGalvo] = removeFirstTrialFromBlock(block, galvoLog)
%% A funciton remove the first trial from block and galvolog files, because these trials often had associated issues.

%INPUTS(default values)
%block(required)---------The block file
%galvoLog(required)------The galvo log (for inactivation trials)

%OUTPUTS
%truncatedBlock----------The block file with the first trial removed
%truncatedGalvo----------The galvo log with the first trial removed

%%

fieldList = fieldnames(block.events);                           %List of the events saved in the block file
trial2Idx = max(find(block.events.repeatNumValues==1,2));       %Finding the index of the second new (not repeated) trial
trial2Start = block.events.endTrialTimes(trial2Idx-1)+0.0001;   %Find the end time of the previous trial (remove things before this time)

%Loop to iterate over the saved events and remove the trial information (exceptions are fields related to the total reward). Fields that are empty, or
%only have a single value (i.e. don't change with trials) are ignored. Note, we modify two fields in each iteration (the times and values for a
%particular event).
for i = 1:2:length(fieldList)
    if contains(fieldList{i}, {'rTotValues'; 'totalRewardValues'}); continue; end
    if isempty(block.events.(fieldList{i})); continue; end
    if length(block.events.(fieldList{i+1})) == 1; continue; end
    
    %Find the last value of the event that happened before the new start time. This is the cutOff time
    cutOff = find(block.events.(fieldList{i+1})<trial2Start, 1, 'last');
    
    %Event ratio is the number of columns of recorded events/number of times. This is >1 if a "value" actually has multiple values at each time.
    %There is a santity check here to make sure that the event ratio is an integer (which it always should be)
    eventRatio = size(block.events.(fieldList{i}),2)/size(block.events.(fieldList{i+1}),2);
    if round(eventRatio) ~= eventRatio
        keyboard; 
    end
    
    %Remove all events and times that happen before the cutOff
    block.events.(fieldList{i})(:,1:(cutOff*eventRatio)) = [];
    block.events.(fieldList{i+1})(:,1:cutOff) = [];
end

%%
%Perform the same process as above, but for the "outputs" field. i.e. times that sounds, rewards, etc. were produced
fieldList = fieldnames(block.outputs);
for i = 1:2:length(fieldList)
    if isempty(block.outputs.(fieldList{i})); continue; end
    if length(block.outputs.(fieldList{i+1})) == 1; continue; end
    cutOff = find(block.outputs.(fieldList{i+1})<trial2Start, 1, 'last');
    eventRatio = size(block.outputs.(fieldList{i}),2)/size(block.outputs.(fieldList{i+1}),2);
    if round(eventRatio) ~= eventRatio; keyboard; end
    block.outputs.(fieldList{i})(:,1:(cutOff*eventRatio)) = [];
    block.outputs.(fieldList{i+1})(:,1:cutOff) = [];
end
%%
%Perform the same process as above, but for the "params" field. i.e. times that paramters changed
block.events.trialNumValues = block.events.trialNumValues-trial2Idx+1;
block.paramsTimes(1:(trial2Idx-1)) = [];
block.paramsValues(1:(trial2Idx-1)) = [];

%%
%Perform the same process as above, but for the galvo log file (excluding the stereotaxic calibration)
if isstruct(galvoLog) && length(fieldnames(galvoLog))>1
    galvoIdx = find(galvoLog.trialNum>=trial2Idx, 1);
    fieldList = fieldnames(galvoLog);
    for i = 1:length(fieldList)
        if strcmp(fieldList{i}, 'stereotaxCalib'); continue; end
        galvoLog.(fieldList{i})(1:(galvoIdx-1),:) = [];
    end
    galvoLog.trialNum = galvoLog.trialNum-trial2Idx+1;
end

truncatedGalvo = galvoLog;
truncatedBlock = block;
end


