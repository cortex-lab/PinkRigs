clear all
close all

%% Get exp ref

params.mice2Check = 'AV009';
% params.days2Check = {'2021-11-22','2021-11-23'};
% params.expDef2Check = 'imageWorld';
params.timeline2Check = 1;
params.align2Check = '*,*,*,*,*,NaN';
params.preproc2Check = '*,*';
exp2checkList = queryExp(params);

%% Just run alignment
params.recompute = {'all'};
preproc.align.main(params,exp2checkList)

%% Just run preprocessing
params.recompute = 1;
preproc.extractExpData(params, exp2checkList)

%% Or run all
% params.paramsAlign.recompute = {'all'};
% param.paramPreproc.recompute = 1;
% preproc.main(params,exp2checkList)

%% Do some processing

%%% TODO
