close all

clear params
params.subject = {'AV024'};
params.expDate = {'2022-10-12'}; 
params.expDef = 'p'; 
%params.checkAlignEphys = '2'; 
params.expNum = '1'; 
%params.expDate = {'2021-03-16'}; 
exp2checkList = csv.queryExp(params);

%% Just run alignment
preproc.align.main(exp2checkList,'recompute',{'block'});

%% Just run preprocessing
preproc.extractExpData(exp2checkList,'recompute',{['events']}); % here spk does not work 
