clc; clear; 
params.mice2Check = 'AV008';
params.days2Check = {'2022-03-11'};
params.expDef2Check = 'multiSpaceWorld_checker_training';
exp2checkList = csv.queryExp(params);

expInfo = exp2checkList(1,:);
expPath = dir([expInfo.expFolder{1} '\' '*_preprocData.mat']); 
load([expPath.folder '\' expPath.name]); 

%%
plt.spk.cellRaster(spk,ev); 
%%
choices = ev.timeline_choiceMoveDir; 
choices = sign(choices-1.5); 
choices(isnan(choices))=0; 
%% aud onsets 
plt.spk.cellRaster(spk, ev.timeline_audPeriodOnOff(:,1),[sign(ev.stim_audAzimuth),choices]);
%% vis onsets
plt.spk.cellRaster(spk, ev.timeline_visPeriodOnOff(:,1),[sign(ev.stim_visAzimuth),choices]);
%% movements 
plt.spk.cellRaster(spk, ev.timeline_choiceMoveOn(:,1),choices);
%% passive? 