close all

%% Get exp ref

clear params
% params.mice2Check = {'AV014'};
% params.days2Check = 1;
params.days2Check = {{'2021-12-18'}};
% params.expDef2Check = 'AVPassive_ckeckerboard_postactive';
% params.timeline2Check = 1;
% params.align2Check = '*,*,*,*,*,~1'; % "any 0"
params.preproc2Check = '2,*';
exp2checkList = csv.queryExp(params);

%% Just run alignment
preproc.align.main(exp2checkList(:,1:3),'recompute',{'ephys'})

%% Just run preprocessing
preproc.extractExpData(exp2checkList(:,1:3),'recompute',{'spk'})

%% Or run all
% params.paramsAlign.recompute = {'all'};
% param.paramPreproc.recompute = {'all'};
% preproc.main(params,exp2checkList)

%% Do some processing

%%% TODO
