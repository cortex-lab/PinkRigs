close all

clear params
params.subject = {'FT038';'FT039'};
params.expDate = {['postImplant']}; 
params.expDef = 'p'; 

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


