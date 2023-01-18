function [ev]=sparseNoise_func(expPath,RID)
[subject, expDate, expNum] = parseExpPath(expPath);
load(fullfile(expPath, [expDate '_' expNum '_' subject '_Timeline.mat']),'Timeline');
load(fullfile(expPath, [expDate '_' expNum '_' subject '_block.mat']),'block');

% Get the appropriate ref for the exp def
expDef = block.expDef;
expDefRef = preproc.getExpDefRef(expDef);

% Call specific preprocessing function
ev = preproc.expDef.(expDefRef)(Timeline,block,1);

% align to ephys probe
params.ephysPath={RID.path(1:end-1)}; 
%[ephysRefTimes,timelineRefTimes,~]=preproc.align.ephys(expPath,params);
[mname, expDate, expNum, ~] = parseExpPath(expPath);
csv.checkForNewPinkRigRecordings('subject',mname); 
expInfo = csv.queryExp('subject',mname,'expDate',expDate,'expNum',expNum);  
[ephysRefTimes,timelineRefTimes,~,~] = preproc.align.ephys(expInfo);
co=robustfit(timelineRefTimes{1},ephysRefTimes{1});
fitTimeline_times = @(t)t*co(2) + co(1);
ev.stimTimesMain=fitTimeline_times(ev.squareOnTimes); 
%ev.stimArrayTimesMain=fitTimeline_times(ev.stimArrayTimes); 
end