close all

%% Get exp ref

clear params
% 

params.subject = {'FT008'};
%params.expDate = 3;
%params.expDate = {{'2021-11-30'}};
%params.expDef = 'sparseNoise';
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
