clc; clear; 
params.subject = 'AV008';
params.expDate = {'2022-03-10'};
params.expDef = 't';
exp2checkList = csv.queryExp(params);

expInfo = exp2checkList(1,:);
expPath = dir([expInfo.expFolder{1} '\' '*_preprocData.mat']); 
load([expPath.folder '\' expPath.name]); 

%%
opt.paramTag = 'default';
plt.spk.cellRaster(spk,ev,[],opt); 

%%
t =  ev.timeline_choiceMoveOn;%(ev.timeline_choiceMoveDir==2);
%%
movie = vidproc.getVideoAtTime(expPath(1).folder,t(9),'sideCam',[-0.5,1]); 
implay(movie); 
%% 
