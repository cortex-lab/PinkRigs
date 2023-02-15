clc; clear all; 
params.subject = {'FT010'};
params.expDate = '2021-03-16';
params.expNum = 7;
exp2checkList = csv.queryExp(params);
expInfo = exp2checkList(1,:);
expInfo = csv.loadData(expInfo, 'dataType',{'timeline','events'});
% load also the mic data (currently not implemented by csv)

expPathStub = strcat(expInfo.expDate, {'_'}, expInfo.expNum, {'_'}, expInfo.subject);
micPath = cell2mat([expInfo.expFolder '\' expPathStub '_mic.mat']);
load(micPath); 
%%
micdl=numel(micData);

timebin=60; % normally happens in the 1st 10s 
samplelenth=timebin*Fs;

st = 1000000;

snippet = double(micData(st-1000000:st+1000000)); 
snippet = double(micData);
figure; spectrogram(snippet,kaiser(1000,10),[],[],Fs);
