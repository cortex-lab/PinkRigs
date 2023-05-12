close all

clear params
params.subject = {'AV038'};
params.expDate = {'2023-03-08'}; 
params.expDef = 'm'; 
preproc.extractExpData(exp2checkList,'recompute',{['events']});
%params.checkAlignEphys = '2'; 
%params.expNum = '1'; 
%params.expDate = {'2021-03-16'}; 
exp2checkList = csv.queryExp(params);

%% Just run alignment
preproc.align.main(exp2checkList,'recompute',{'ephys'});

%% Just run preprocessing
preproc.extractExpData(exp2checkList,'recompute',{['events']},'process',{['events']}); % here spk does not work 


