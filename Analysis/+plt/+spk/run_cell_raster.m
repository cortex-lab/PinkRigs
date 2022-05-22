clc; clear; 
params.mice2Check = 'FT031';
params.days2Check = {'2021-12-03'};
params.expDef2Check = 't';
exp2checkList = csv.queryExp(params);

expInfo = exp2checkList(1,:);
expPath = dir([expInfo.expFolder{1} '\' '*_preprocData.mat']); 
load([expPath.folder '\' expPath.name]); 

%%
opt.paramTag = 'default';
plt.spk.cellRaster(spk,ev,[],opt); 
