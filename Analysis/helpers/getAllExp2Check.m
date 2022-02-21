function exp2checkList = getAllExp2Check(varargin)
    %%% This function will fetch all possible experiments to check for
    %%% computing alignment, preprocessing etc.
    
    %% Get parameters
    %%% Parameters to tell how far in the past to look for.
    params.days2Check = inf;
    params.mice2Check = 'active';
   
    if ~isempty(varargin)
        paramsIn = varargin{1};
        params = parseInputParams(params,paramsIn);
    end
    
    todayDate = datenum(date);
    
    %% --------------------------------------------------------
    %% Fetch exp
    % Get active mouse list from main csv
    mainCSVLoc = getCSVLocation('main');
    mouseList = readtable(mainCSVLoc);
    
    if strcmp(params.mice2Check,'active')
        mouse2checkList = mouseList.Subject(mouseList.IsActive>0);
    elseif strcmp(params.mice2Check,'all')
        mouse2checkList = mouseList.Subject;
    else
        % specific mice
        mouse2checkList = mouseList.Subject(ismember(mouseList.Subject,params.mice2Check));
    end
    
    % Loop through csv to look for experiments that weren't
    % aligned, or all if recompute isn't none.
    exp2checkList = table();
    for mm = 1:numel(mouse2checkList)
        % Loop through subjects
        subject = mouse2checkList{mm};
        
        expListMouse = getMouseExpList(subject);
        dates2Check = todayDate - datenum(expListMouse.expDate) <= params.days2Check;
        
        % Get list of exp for this mouse
        exp2checkList = [exp2checkList; expListMouse(dates2Check,:)];
    end