close all

%% Get exp ref

clear params
% 

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
