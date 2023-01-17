close all

%% Get exp ref

clear params
% 
params.subject = {['all']};
csvData = csv.inputValidation(params);
csvData = csvData.mainCSV{1};
% select data of SC implant (based on AP position) 
is_SC = cellfun(@(x)str2double(x), csvData.P0_AP)<-3.7;
SC_subject = csvData.Subject(is_SC);
SC_subject = SC_subject';
%params.expDate = '2022-12-09';
%%
clear params
%params.subject = SC_subject;
params.subject = {['AV025']};
params.expDate = {'2022-11-08'};
params.expDef = 'sparseNoise';
%params.expNum = 3; 

% params.checkAlignCam = {2};
%params.checkEvents = 2;
exp2checkList = csv.queryExp(params);
exp2checkList(cell2mat(exp2checkList.daysSinceImplant)<0,:)=[];
%% Just run alignment
preproc.align.main(exp2checkList,'recompute',{'ephys'});
%% Just run preprocessing
preproc.extractExpData(exp2checkList,'recompute',{['spikes']}); % here spk does not work 
%% Or run all
% params.paramsAlign.recompute = {'all'};
% param.paramPreproc.recompute = {'all'};
% preproc.main(params,exp2checkList)

%% Do some processing

%%% TODO
