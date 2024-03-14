function [params] = optoParams(which)
    params.subject  = {['AV029'];['AV033'];['AV031'];['AV036'];['AV038'];['AV046'];['AV041'];['AV047'];['AV044']};
    %params.subject  = {['AV036'];['AV038'];['AV046'];['AV041'];['AV044'];['AV047']};
    %params.subject  = {['AV036'];['AV038']};
    %params.subject  = {['AV041'];['AV047'];['AV044']};
    %params.subject  = {['AV047'];['AV044']};
    %params.subject  = {['AV044']};
    params.expDef = 't'; 
    params.checkEvents = '1';
    params.expDate = {['2022-09-04:2023-12-04']}; 
    params.combMice = 0; 
    params.minN = 400; 
    params.includeNoGo = 0; 
    params.sepPlots= 1; % customary!!

    % write optionals for this
    if strcmp('bi_high',which)
        params.reverse_opto =0;
        params.selPowers = [34];
        params.selHemispheres = [0];

    elseif strcmp('bi_low',which)
        params.reverse_opto =1;
        params.selPowers = [20];
        params.selHemispheres = [0];    

    elseif strcmp('uni_high',which)
        params.reverse_opto =1;
        params.selPowers = [17];
        params.selHemispheres = [-1,1];
    
    elseif strcmp('uni_low',which)
        params.reverse_opto =1;
        params.selPowers = [10];
        params.selHemispheres = [-1,1];

    elseif strcmp('uni_all',which)
        params.reverse_opto =1;
        params.selPowers = [10,30];
        params.selHemispheres = [-1,1];

    elseif strcmp('all_one_p',which)
        params.reverse_opto =0;
        params.selPowers = [10,20,15,17,30];
        params.selHemispheres = [-1,0,1];
    end 
end 