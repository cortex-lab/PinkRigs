close all

clear params
%params.subject = {['FT008'],['FT009'],['FT010'],['FT011'],['FT038'],['FT039'],['AV024'],['AV028'],['FT019'],['FT022'],['FT025'],['FT027']};
%params.subject ={['FT039']};
params.subject  = {['FT008']};
%
% params.expDate = {['2021-12-01']}; 
%params.expNum = '3'; 
params.expDef = 'n'; 
%params.checkAlignEphys = '2'; 
%params.expNum = '1'; 
%params.expDate = {'2021-03-16'}; 
exp2checkList = csv.queryExp(params);

%% Just run alignment
preproc.align.main(exp2checkList,'recompute',{'ephys'});

%% Just run preprocessing
preproc.extractExpData(exp2checkList,'recompute',{['events']},'process',{['events']}); % here spk does not work 

