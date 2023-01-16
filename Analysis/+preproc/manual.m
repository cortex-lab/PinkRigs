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

params.subject = {'AV015'};
% params.expDate = 'last4';
params.expDate = {'2022-07-19'};
params.expDef = 's';
% params.checkAlignCam = {2};
% params.checkSpikes = 2;
exp2checkList = csv.queryExp(params);
 
%% Just run alignment
preproc.align.main(exp2checkList,'recompute',{'ephys'})

%% Just run preprocessing
preproc.extractExpData(exp2checkList,'recompute',{'events'});

%% Or run all
% params.paramsAlign.recompute = {'all'};
% param.paramPreproc.recompute = {'all'};
% preproc.main(params,exp2checkList)

%% Do some processing

%%% TODO
