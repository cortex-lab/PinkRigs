close all

clear params
params.subject = {'FT008','FT009','FT010'};

exp2checkList = csv.queryExp(params);

%% Just run alignment
preproc.align.main(exp2checkList,'recompute',{'ephys'});

%% Just run preprocessing
preproc.extractExpData(exp2checkList,'recompute',{['spikes']}); % here spk does not work 
