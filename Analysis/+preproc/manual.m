close all

%% Get exp ref

clear params
params.mice2Check = {'AV005'};
params.days2Check = 1;
params.days2Check = {{'2022-05-27'}};
% params.expDef2Check = 'AVPassive_ckeckerboard_postactive';
% params.timeline2Check = 1;
% params.align2Check = '*,*,*,*,*,~1'; % "any 0"
params.preproc2Check = '2,*';
exp2checkList = csv.queryExp(params);

%% Just run alignment
params.recompute = {'ephys'};
preproc.align.main(params,exp2checkList)

%% Just run preprocessing
params.recompute = {'spk'};
%params.recompute = {'ev'};
preproc.extractExpData(params, exp2checkList)

%% Or run all
% params.paramsAlign.recompute = {'all'};
% param.paramPreproc.recompute = {'all'};
% preproc.main(params,exp2checkList)

%% Do some processing

%%% TODO
