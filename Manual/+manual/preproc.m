close all

clear params
params.subject = {'all'};
params.expDate = {'all'}; 
params.expDef = 'all'; 
params.checkAlignEphys = '2'; 

%params.expDate = {'2021-03-16'}; 
exp2checkList = csv.queryExp(params);

%% Just run alignment
preproc.align.main(exp2checkList,'recompute',{'ephys'});

%% Just run preprocessing
preproc.extractExpData(exp2checkList,'recompute',{['spikes']}); % here spk does not work 
