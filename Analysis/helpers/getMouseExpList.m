function expList = getMouseExpList(subject)
%% Function to load the csv. file with the list of experiments for a particular mouse
csvLocation = ['\\zserver.cortexlab.net\Code\AVrig\' subject '.csv'];
expList = readtable(csvLocation);
end