clear all
close all

%% Get exp ref

params.mice2Check = {'AV007','AV008','AV009'};
% params.days2Check = 3;
% params.days2Check = {'2022-03-23','2022-03-24'};
% params.expDef2Check = 'imageWorld';
params.timeline2Check = 1;
% params.align2Check = '*,*,*,*,*,~1'; 
params.preproc2Check = '1,*';
exp2checkList = csv.queryExp(params);

%% Just run alignment
params.recompute = {'none'};
preproc.align.main(params,exp2checkList)

%% Just run preprocessing
params.recompute = {'spk'};
preproc.extractExpData(params, exp2checkList)

%% Or run all
% params.paramsAlign.recompute = {'all'};
% param.paramPreproc.recompute = 1;
% preproc.main(params,exp2checkList)

%% Do some processing

%%% TODO
