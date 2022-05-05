clear all
close all

%% Get exp ref

clear params
params.mice2Check = 'AV009';
<<<<<<< HEAD
params.days2Check = 10;
% params.days2Check = {'2022-04-04'};
% params.expDef2Check = 'AVPassive_ckeckerboard_postactive';
params.timeline2Check = 1;
% params.align2Check = '*,*,*,*,*,~1'; % "any 0"
params.preproc2Check = '2,*';
=======
% params.days2Check = 10000;
params.days2Check = '2022-03-14';
params.expDef2Check = 'multiSpaceWorld_checker_training';
% params.timeline2Check = 1;
% params.align2Check = '*,*,*,*,*,~1'; % "any 0"
% params.preproc2Check = '*,2';
>>>>>>> 8198bf9f46ebc8570edeee894ae8d7412530b836
exp2checkList = csv.queryExp(params);

%%
clear params
params.mice2Check = 'AV008';
params.days2Check = '2022-03-11';
params.expDef2Check = 'multiSpaceWorld_checker_training';
exp2checkList = csv.queryExp(params);
params.recompute = {'ev'};
%%
preproc.extractExpData(params, exp2checkList)


%% Just run alignment
params.recompute = {'ephys'};
preproc.align.main(params,exp2checkList)

%% Just run preprocessing
<<<<<<< HEAD
params.recompute = {'all'};
=======
params.recompute = {'spk'};
>>>>>>> 8198bf9f46ebc8570edeee894ae8d7412530b836
preproc.extractExpData(params, exp2checkList)

%% Or run all
% params.paramsAlign.recompute = {'all'};
% param.paramPreproc.recompute = {'all'};
% preproc.main(params,exp2checkList)

%% Do some processing

%%% TODO
