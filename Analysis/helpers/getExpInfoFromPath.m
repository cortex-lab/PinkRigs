function expInfoList = getExpInfoFromPath(expPathList, skipCSVUpdate)
    %%% This function will go fetch the exp info from the csv, given the
    %%% exp path.
       
    %% Get the exp info from list of paths
    % Will check if currently in the csv, and crash if not.
    
    subjectList = cell(1,numel(numel(expPathList)));
    expDateList = cell(1,numel(numel(expPathList)));
    expNumList = cell(1,numel(numel(expPathList)));
    for ee = 1:numel(expPathList)
        [subjectList{ee},expDateList{ee},expNumList{ee}] = parseExpPath(expPathList{ee});
    end
    
    expInfoList = table();
    subjects = unique(subjectList);
    for ss = 1:numel(subjects)
        expList = getMouseExpList(subjects{ss});
        idx4thisSubject = find(contains(subjectList,subjects{ss}));
        for idx = 1:numel(idx4thisSubject)
            expIdx = find(contains(cellstr(datestr(expList.expDate,29)),expDateList{idx4thisSubject(idx)}) & ...
                contains(expList.expNum,num2str(expNumList{idx4thisSubject(idx)})));
            if ~isempty(expIdx)
                expInfoList = [expInfoList; expList(expIdx,:)];
            else
                % Exp not in the csv. Update and recheck, or error.
                if ~skipCSVUpdate
                    days2Check = 3;
                    recompute = 0;
                    checkForNewAVRecordings(days2Check, recompute)
                    expList = getMouseExpList(subjects{ss});
                end
                
                % Retry
                expIdx = find(contains(cellstr(datestr(expList.expDate,29)),expDateList{idx4thisSubject(idx)}) & ...
                    contains(expList.expNum,num2str(expNumList{idx4thisSubject(idx)})));
                if ~isempty(expIdx)
                    expInfoList = [expInfoList; expList(expIdx,:)];
                else
                    error('Exp. ''%s'' not found in csv. Have a look?', expList(expIdx,:).path{1})
                end
            end
        end
    end