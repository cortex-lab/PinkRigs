clc; clear all; 
params.subject = {['AV030']};
params.expDate = '2022-12-08';
params.expNum = 2;
exp2checkList = csv.queryExp(params);

expInfo = exp2checkList(1,:);
expInfo = csv.loadData(expInfo, 'dataType',{'blk','events'});
% load also the mic data (currently not implemented by csv)
%%
expPathStub = strcat(expInfo.expDate, {'_'}, expInfo.expNum, {'_'}, expInfo.subject);
micPath = cell2mat([expInfo.expFolder '\' expPathStub '_mic.mat']);
load(micPath); 
%%
micdl=numel(micData);

timebin=60; % normally happens in the 1st 10s 
samplelenth=timebin*Fs;

st = micdl-10000001;

snippet = double(micData(st-10000000:st+10000000)); 

figure; spectrogram(snippet,kaiser(500,10),[],[],Fs,'yaxis');
