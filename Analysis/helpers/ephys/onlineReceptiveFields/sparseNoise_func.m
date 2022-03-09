function [ev]=sparseNoise_func(expPath,RID)
timeline = getTimeline(expPath);
block = getBlock(expPath);

% Get the appropriate ref for the exp def
expDef = block.expDef;
expDefRef = preproc.getExpDefRef(expDef);

% Call specific preprocessing function
ev = preproc.expDef.(expDefRef)(timeline,block,1);

% align to ephys probe
params.ephysPath={RID.path(1:end-1)}; 
[ephysRefTimes,timelineRefTimes,~]=preproc.align.ephys(expPath,params);
co=robustfit(timelineRefTimes{1},ephysRefTimes{1});
fitTimeline_times = @(t)t*co(2) + co(1);
ev.stimTimesMain=fitTimeline_times(ev.stimTimes); 
ev.stimArrayTimesMain=fitTimeline_times(ev.stimArrayTimes); 
end