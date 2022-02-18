function exp2checkList = getAllExp2Check(varargin)
    %%% This function will fetch all possible experiments to check for
    %%% computing alignment, preprocessing etc.
    
    %% Get parameters
    %%% Parameters to tell how far in the past to look for.
    days2Check = inf;
    mice2Check = 'active';
    
    % This is not ideal
    if ~isempty(varargin)
        params = varargin{1};
        
        if ~isempty(params) && isfield(params, 'days2Check')
            days2Check = params.days2Check;
        end
        if ~isempty(params) && isfield(params, 'mice2Check')
            mice2Check = params.mice2Check;
        end
    end
    
    todayDate = datenum(date);
    
    %% --------------------------------------------------------
    %% Fetch exp
    % Get active mouse list from main csv
    mainCSVLoc = getCSVLocation('main');
    mouseList = readtable(mainCSVLoc);
    
    if strcmp(mice2Check,'active')
        mouse2checkList = mouseList.Subject(mouseList.IsActive>0);
    elseif strcmp(mice2Check,'all')
        mouse2checkList = mouseList.Subject;
    else
        % specific mice
        mouse2checkList = mouseList.Subject(ismember(mouseList.Subject,mice2Check));
    end
    
    % Loop through csv to look for experiments that weren't
    % aligned, or all if recompute isn't none.
    exp2checkList = table();
    for mm = 1:numel(mouse2checkList)
        % Loop through subjects
        subject = mouse2checkList{mm};
        
        expListMouse = getMouseExpList(subject);
        dates2Check = todayDate - datenum(expListMouse.expDate) <= days2Check;
        
        % Get list of exp for this mouse
        exp2checkList = [exp2checkList; expListMouse(dates2Check,:)];
    end