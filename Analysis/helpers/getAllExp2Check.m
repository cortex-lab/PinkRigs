function exp2checkList = getAllExp2Check(varargin)
    %%% This function will fetch all possible experiments to check for
    %%% computing alignment, preprocessing etc.
    
    %% Get parameters
    %%% Parameters to tell how far in the past to look for.
    params.days2Check = inf;
    params.mice2Check = 'active';
    params.expDef2Check = 'all';
    params.timeline2Check = 0;
    
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
        
        % Initialize indices of exp to keep
        exp2Check = true(size(expListMouse,1),1);
        
        % Get specific dates
        if isa(params.days2Check,'double')
            dates2Check = todayDate - datenum(expListMouse.expDate) <= params.days2Check;
        elseif isa(params.days2Check,'cell')
            dates2Check = ismember(datenum(expListMouse.expDate),datenum(params.days2Check));
        else
            warning('Couldn''t find correspond dates for mouse %s', subject)
            dates2Check = [];
        end
        exp2Check = exp2Check & dates2Check;
        
        % Get specific expDefs
        if ~strcmp(params.expDef2Check,'all')
            expDef2Check = contains(expListMouse.expDef,params.expDef2Check);
            exp2Check = exp2Check & expDef2Check;
        end
        
        if params.timeline2Check
            exp2Check = exp2Check & expListMouse.timeline>0;
        end

        % Get list of exp for this mouse
        exp2checkList = [exp2checkList; expListMouse(exp2Check,:)];
    end