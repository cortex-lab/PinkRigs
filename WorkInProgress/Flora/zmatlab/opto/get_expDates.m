
clc; clear all;
extracted = loadOptoData('balanceTrials',0,'sepMice',1,'reExtract',1, ...
    'sepHemispheres',0,'sepPowers',0,'sepDiffPowers',0,'whichSet', 'uni_all'); 

% get the date IDs of each session
load('C:\Users\Flora\Documents\Processed data\Audiovisual\opto\expDates.mat'); 

%% calculate for each animal
minSessionIDs = cellfun(@(x) min([x.sessionID]), extracted.data);
startSessDate = {ID_dates{1,minSessionIDs}}; 

maxSessionIDs = cellfun(@(x) max([x.sessionID]), extracted.data);
endSessDate = {ID_dates{1,maxSessionIDs}}; 

startEndDates = cat(2,extracted.subject,startSessDate',endSessDate'); 