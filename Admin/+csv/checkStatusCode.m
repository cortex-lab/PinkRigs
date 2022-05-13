function codeChecked = checkStatusCode(codes2Check,codeRef)
    %%% This function will check the status of experiment (taken from the
    %%% csv) against a specific request
    %%%
    %%% (a,a,a) will take these if any of the bits that are inside equal a
    %%% (equivalent to "any")
    %%% ~(a,a,a) will take these if any of the bits that are inside equal a
    %%% (equivalent to "not any")
    %%% x(a) will take these if at least one of the bit is different
    %%% from a (equivalent to "not all")
    
    
    % Check if grouping (for "any" or "not any")
    parenthesisCheck = regexp(codeRef,'(\(|\))','split');
    if numel(parenthesisCheck)>1
        if numel(parenthesisCheck)>3
            error('Too many "~(.)" propositions. Can only take one.')
        end
        % Check if not any
        if ~isempty(parenthesisCheck{1}) && strcmp(parenthesisCheck{1}(end),'~')
            % "not any"
            parStatus = 2;
            parenthesisCheck{1}(end) = [];
        elseif ~isempty(parenthesisCheck{1}) && strcmp(parenthesisCheck{1}(end),'x')
            % "not all"
            parStatus = 3;
            parenthesisCheck{1}(end) = [];
        else
            % "any"
            parStatus = 1;
        end
        
        % Get parenthesis position
        parStart = numel(regexp(parenthesisCheck{1},',','split'))-1; % Last bit of the regexp should be empty
        parEnd = numel(regexp(parenthesisCheck{end},',','split'))-1; % First bit of the regexp should be empty
        if isempty(parStart)
            parStart = 0;
        end
        if isempty(parEnd)
            parEnd = 0;
        end
    else
        parStatus = 0;
    end
    codeRef_noParenth = cell2mat(parenthesisCheck);
    codeRefIndivBits = regexp(codeRef_noParenth,',','split');
    
    % Change the shape of the code to check
    codes2Check_res = cellfun(@(x) regexp(x,',','split'),codes2Check,'UniformOutput',false);
    
    nBits = numel(codeRefIndivBits);
    
    % Check correspondance of input code with ref code
    align2Check = true(numel(codes2Check_res),nBits);
    alignStatus = cell(1,nBits);
    for ii = 1:nBits
        alignStatus{ii} = cellfun(@(x) x{ii}, codes2Check_res, 'UniformOutput', false);
        if ~strcmp(codeRefIndivBits{ii},'*')
            if contains(codeRefIndivBits{ii},'~')
                align2Check(:,ii) = ~strcmp(lower(string(alignStatus{ii})),lower(codeRefIndivBits{ii}(2:end)));
            else
                align2Check(:,ii) = strcmp(lower(string(alignStatus{ii})),lower(codeRefIndivBits{ii}));
            end
        else
            align2Check(:,ii) = true;
        end
    end
    
    switch parStatus
        case 0
            codeChecked = all(align2Check,2);
        case 1
            % There was a "any"
            codeChecked = all(align2Check(:,1:parStart),2) & ...
                any(align2Check(:,parStart+1:end-parEnd),2) & ...
                all(align2Check(:,end-parEnd+1:end),2);
        case 2
            % There was a "not any"
            codeChecked = all(align2Check(:,1:parStart),2) & ...
                ~any(align2Check(:,parStart+1:end-parEnd),2) & ...
                all(align2Check(:,end-parEnd+1:end),2);
        case 3
            % There was a "not all"
            codeChecked = all(align2Check(:,1:parStart),2) & ...
                ~all(align2Check(:,parStart+1:end-parEnd),2) & ...
                all(align2Check(:,end-parEnd+1:end),2);
    end
end