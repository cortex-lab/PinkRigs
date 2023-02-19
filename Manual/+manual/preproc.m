close all

clear params
params.subject = {'all'};
%params.expDate = {'2021-03-16'}; 
params.expDate = 'postImplant';

exp2checkList = csv.queryExp(params);

%% Just run alignment
preproc.align.main(exp2checkList,'recompute',{'ephys'});

%% Just run preprocessing
preproc.extractExpData(exp2checkList,'recompute',{['spikes']}); % here spk does not work 
