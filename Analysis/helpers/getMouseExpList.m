function expList = getMouseExpList(subject)
    %% Function to load the .csv file with the list of experiments for a particular mouse
    
    csvLocation = getCSVLocation(subject);
    expList = readtable(csvLocation);
    if isnumeric(expList.expNum)
        expList.expNum = arrayfun(@num2str, expList.expNum, 'uni', 0);
    end
end