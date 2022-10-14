function [ev] = sparseNoise(timeline, block, alignmentBlock)
% bulk of the code is taken from kilotrode 
% note that there is no true alignment of block here -- we just find the
% crrect indices of Timeline photodiode times 
blockRefTimes = block.stimWindowUpdateTimes;

blk_ev = block.events;
stimArray = blk_ev.stimuliOnValues>0;
stimArray = reshape(stimArray, size(stimArray,1), size(stimArray,2)/numel(blockRefTimes), []);

% should find a way not to hardcode this (but is currently part of
% vis.checker)
xRange = [-132 132];
xPos = linspace(xRange(1), xRange(2), size(stimArray,2)+1);
xPos = xPos(1:end-1)+mean(diff(xPos))/2;
yRange = [-36 36];
yPos = linspace(yRange(1), yRange(2), size(stimArray,1)+1);
yPos = yPos(1:end-1)+mean(diff(yPos))/2;

stimArrayZeroPad = cat(3,zeros(size(stimArray,1), size(stimArray,2),1), stimArray);
stimTimeInds = {[], []};
stimPositions = {[], []};
for x = 1:size(stimArray,1)
    for y = 1:size(stimArray,2)
        stimEventTimes{x,y,1} = find(stimArrayZeroPad(x,y,1:end-1)==0 & ...
            stimArrayZeroPad(x,y,2:end)==1); % going from black to white
        stimEventTimes{x,y,2} = find(stimArrayZeroPad(x,y,1:end-1)==1 & ...
            stimArrayZeroPad(x,y,2:end)==0); % going from white to black
        stimTimeInds{1} = [stimTimeInds{1}; stimEventTimes{x,y,1}];
        stimTimeInds{2} = [stimTimeInds{2}; stimEventTimes{x,y,2}];
        
        nEv = length(stimEventTimes{x,y,1});
        stimPositions{1} = [stimPositions{1}; yPos(x)*ones(nEv,1) xPos(y)*ones(nEv,1)];
        nEv = length(stimEventTimes{x,y,2});
        stimPositions{2} = [stimPositions{2}; yPos(x)*ones(nEv,1) xPos(y)*ones(nEv,1)];
    end
end

timelineRefTimes = timeproc.getChanEventTime(timeline,'photoDiode');

if length(blockRefTimes) ~= length(timelineRefTimes)
    if (length(blockRefTimes)-length(timelineRefTimes))==1
       truncated_block = blockRefTimes(2:end);
       if sum((diff(truncated_block)-diff(timelineRefTimes))>0.25)==0
           blockRefTimes=truncated_block; 
           stimArray = stimArray(:,:,2:end);
%% I am starting to not understand this ---- if flips are thrown away the indexing is surely False????
       else
           [timelineRefTimes, blockRefTimes] = try2alignVectors(timelineRefTimes,blockRefTimes,0.25,1);
       end

    else
        [timelineRefTimes, blockRefTimes] = try2alignVectors(timelineRefTimes,blockRefTimes,0.25,1);
    end
elseif any(abs((blockRefTimes-blockRefTimes(1)) - (timelineRefTimes-timelineRefTimes(1)))>0.5)
    [timelineRefTimes, blockRefTimes] = try2alignVectors(timelineRefTimes, blockRefTimes,0.25,1);
end
block.alignment = 'photodiode';
if length(blockRefTimes) ~= length(timelineRefTimes)
    error('Photodiode alignment error');
end


% it happends sometimes that there is one more stimulus index loaded in
% block compared to timeline... most often this doesn't happen....
if max(stimTimeInds{1})>length(timelineRefTimes)
    ix_drop=find(stimTimeInds{1}>length(timelineRefTimes));
    stimTimeInds{1}(ix_drop)=[];
    stimPositions{1}(ix_drop,:)=[];
%     fromdrop=size(stimArray,3)-size(ix_drop,1)+1; 
%     stimArray(:,:,fromdrop:end)=[];
end 

%

stimTimes = timelineRefTimes(stimTimeInds{1}); 
stimPositions = stimPositions{1}; % stim positions in yx 
stimArrayTimes = timelineRefTimes;

stimArrayatOn = stimArray(:,:,stimTimeInds{1}); 
%% write event structure 
ev.squareOnTimes = stimTimes;
ev.squareElevation = stimPositions(:,1); 
ev.squareAzimuth = stimPositions(:,2); 
ev.stimulus = permute(stimArrayatOn,[3 1 2]);
%ev.stimArrayTimes = stimArrayTimes; 

end

