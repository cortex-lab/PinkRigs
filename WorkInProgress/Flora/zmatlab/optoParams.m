function [params] = optoParams()
    %params.subject  = {['AV029'];['AV033'];['AV031'];['AV036'];['AV038'];['AV046'];['AV041'];['AV047'];['AV044']};
    %params.subject  = {['AV036'];['AV038'];['AV046'];['AV041'];['AV047']};
    params.subject  = {['AV041']};
    params.expDef = 't'; 
    params.checkEvents = '1';
    params.expDate = {['2022-07-01:2023-08-20']}; 
    params.reverse_opto = 0;
    params.combMice = 0; 
    params.selPowers = [10,20,34];
    params.selHemispheres = [0];
    params.sepPlots= 1; % customary!!
    params.minN = 600; 
end 