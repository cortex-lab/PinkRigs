function block = getBlock(subject,expDate,expNum)
    %%% This function will load timeline.
    
    expPath = getExpPath(subject, expDate, expNum);
    load(fullfile(expPath, [expDate '_' num2str(expNum) '_' subject '_block.mat']),'block');