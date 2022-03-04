clear all
close all

%% Get exp ref

params.mice2Check = 'FT035';
params.days2Check = {'2021-12-14'};
% params.expDef2Check = 'imageWorld';
params.timeline2Check = 1;
params.align2Check = '*,*,*,*,*'; 
params.preproc2Check = '*,*';
exp2checkList = queryExp(params);

%% Just run alignment
params.recompute = {'all'};
preproc.align.main(params,exp2checkList)

%% Just run preprocessing
params.recompute = {'all'};
preproc.extractExpData(params, exp2checkList)

%% Or run all
% params.paramsAlign.recompute = {'all'};
% param.paramPreproc.recompute = 1;
% preproc.main(params,exp2checkList)

%% Do some processing

%%% TODO
