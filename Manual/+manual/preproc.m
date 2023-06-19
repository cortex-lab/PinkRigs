close all

clear params
params.subject = {'AV025'};
params.expDate = {['2022-11-07']}; 
params.expDef = 'multiSpaceWorld_checker_training'; 

%
% preproc.extractExpData(exp2checkList,'recompute',{['events']});
%params.checkAlignEphys = '2'; 
%params.expNum = '1'; 
%params.expDate = {'2021-03-16'}; 
exp2checkList = csv.queryExp(params);

%% Just run alignment
preproc.align.main(exp2checkList,'recompute',{'ephys'});

%% Just run preprocessing
preproc.extractExpData(exp2checkList,'recompute',{['spikes']},'process',{['spikes']}); % here spk does not work 


