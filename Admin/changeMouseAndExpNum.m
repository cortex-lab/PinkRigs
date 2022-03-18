function changeMouseAndExpNum(expPath,newMouse,newExpNum)
    %% When the wrong mouse name is chosen in Alyx
    
    newExpNum = num2str(newExpNum);
    [subject, expDate, expNum, server] = parseExpPath(expPath);
    d = dir(fullfile(expPath,['*' subject '*']));
    for ii = 1:numel(d)
        oldName = d(ii).name;
        newName = regexprep(oldName,['_' subject '_'],['_' newMouse '_']); % change mouse name
        newName = regexprep(newName,['_' expNum '_'],['_' newExpNum '_']);
        movefile(fullfile(expPath,oldName),fullfile(expPath,newName))
    end
    
end