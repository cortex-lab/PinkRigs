function Timeline = getTimeline(subject,expDate,expNum)
    %%% This function will load timeline.
    
    expPath = getExpPath(subject, expDate, expNum);
    load(fullfile(expPath, [expDate '_' num2str(expNum) '_' subject '_Timeline.mat']),'Timeline');