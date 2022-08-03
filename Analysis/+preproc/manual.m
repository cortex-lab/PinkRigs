close all

%% Get exp ref

clear params
% 

params.subject = {'AV015'};
% params.expDate = 'last4';
% params.expDate = {'2022-07-28'};
params.expDef = 'p';
params.checkAlignEphys = 1;
% params.checkSpikes = 0;
exp2checkList = csv.queryExp(params);
 
%% Just run alignment
preproc.align.main(exp2checkList,'recompute',{'ephys'})

%% Just run preprocessing
preproc.extractExpData(exp2checkList,'recompute',{'spikes'});

%% Or run all
% params.paramsAlign.recompute = {'all'};
% param.paramPreproc.recompute = {'all'};
% preproc.main(params,exp2checkList)

%% Do some processing

%%% TODO
