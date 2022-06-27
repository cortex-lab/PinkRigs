close all

%% Get exp ref

clear params
% 

params.mice2Check = {'FT008'};
%params.days2Check = 3;
%params.days2Check = {{'2021-11-30'}};
%params.expDef2Check = 'sparseNoise';
% params.timeline2Check = 1;
% params.align2Check = '*,*,*,*,*,~1'; % "any 0"
params.preproc2Check = '*,*';
exp2checkList = csv.queryExp(params);

%% Just run alignment
preproc.align.main(exp2checkList(:,1:3),'recompute',{'ephys'})

%% Just run preprocessing
preproc.extractExpData(exp2checkList(:,1:3),'recompute',{'all'});

%% Or run all
% params.paramsAlign.recompute = {'all'};
% param.paramPreproc.recompute = {'all'};
% preproc.main(params,exp2checkList)

%% Do some processing

%%% TODO
