close all

clear params
params.subject = {'default'};


%params.expDate = {'2021-03-16'}; 
exp2checkList = csv.queryExp(params);

%% Just run alignment
preproc.align.main(exp2checkList,'recompute',{'block'});

%% Just run preprocessing
preproc.extractExpData(exp2checkList,'recompute',{['all']}); % here spk does not work 
