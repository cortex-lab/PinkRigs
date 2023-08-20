close all

clear params
params.subject = {['all']};
params.expDate = {['2023-08-08']}; 
params.expDef = 't'; 

%

% preproc.extractExpData(exp2checkList,'recompute',{['ephys']});
%params.checkAlignEphys = '2'; 
%params.expNum = '1'; 
%params.expDate = {'2021-03-16'}; 
exp2checkList = csv.queryExp(params);

%% Just run alignment
preproc.align.main(exp2checkList,'recompute',{'ephys'});

%% Just run preprocessing
preproc.extractExpData(exp2checkList,'recompute',{['events']},'process',{['events']}); % here spk does not work 


