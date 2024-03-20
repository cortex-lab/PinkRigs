close all

clear params
%params.subject = {['AV029'];['AV033'];['AV031'];['AV036'];['AV038'];['AV046'];['AV041'];['AV047'];['AV044']};
%params.subject = {['FT008'],['FT009'],['FT010'],['FT011'],['FT022'],['FT019'],['FT025'],['FT027']};
params.subject = {['AV043']}; 
params.expDate = {['2024-03-06']};
%params.expNum ={'1'}; 

%params.expDef = 't'; 
    
%

% preproc.extractExpData(exp2checkList,'recompute',{['ephys']});
%params.checkAlignEphys = '2'; 
%params.expNum = '1'; 
%params.expDate = {'2021-03-16'}; 
exp2checkList = csv.queryExp(params);


%% Just run alignment
preproc.align.main(exp2checkList,'recompute',{'alignment'});

%% Just run preprocessing
preproc.extractExpData(exp2checkList,'recompute',{['spikes']},'process',{['spikes']}); % here spk does not work 


