function changeMouseAndExpNum(expPath,newMouse,newExpNum)
    %% Changes the mouse name and exp num in all files of a folder.
    % Usually happens when the wrong mouse name is chosen in MC. Better to
    % do it once everything has been copied on the server.
    %
    % Parameters:
    % -------------------
    % expPath: str
    %   Path of the experiment
    % newMouse: str
    %   New mouse name
    % newExpNum: str
    %   New exp num

    %% 
    
    newExpNum = num2str(newExpNum);
    [subject, ~, expNum, ~] = parseExpPath(expPath);
    d = dir(fullfile(expPath,['*' subject '*']));
    for ii = 1:numel(d)
        oldName = d(ii).name;
        newName = regexprep(oldName,['_' subject '_'],['_' newMouse '_']); % change mouse name
        newName = regexprep(newName,['_' expNum '_'],['_' newExpNum '_']);
        movefile(fullfile(expPath,oldName),fullfile(expPath,newName))
    end
    
end