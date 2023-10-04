close all

clear params
%params.subject = {['AV029'];['AV033'];['AV031'];['AV036'];['AV038'];['AV046'];['AV041'];['AV047'];['AV044']};
params.subject = {['AV047']};
params.expDate = {['2023-09-11']}; 
%params.expDef = 't'; 
    
%

% preproc.extractExpData(exp2checkList,'recompute',{['ephys']});
%params.checkAlignEphys = '2'; 
%params.expNum = '1'; 
%params.expDate = {'2021-03-16'}; 
exp2checkList = csv.queryExp(params);


%% Just run alignment
preproc.align.main(exp2checkList,'recompute',{'topCam'});

%% Just run preprocessing
preproc.extractExpData(exp2checkList,'recompute',{['events']},'process',{['events']}); % here spk does not work 


