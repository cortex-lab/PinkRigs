function [params] = optoParams()
    params.subject  = {['AV029'];['AV033'];['AV031'];['AV036'];['AV038'];['AV046'];['AV041'];['AV047'];['AV044']};
    %params.subject  = {['AV036'];['AV038'];['AV046'];['AV041'];['AV047']};
    %params.subject  = {['AV036'];['AV038']};
    %params.subject  = {['AV041'];['AV047'];['AV044']};
    %params.subject  = {['AV041']};
    params.expDef = 't'; 
    params.checkEvents = '1';
    params.expDate = {['2022-07-01:2023-09-31']}; 
    params.reverse_opto =1;
    params.combMice = 0; 
    params.selPowers = [10];
    params.selHemispheres = [-1,1];
    params.sepPlots= 1; % customary!!
    params.minN = 600; 
    params.includeNoGo = 0;  
end 