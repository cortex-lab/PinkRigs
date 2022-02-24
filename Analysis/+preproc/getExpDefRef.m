function expDefRef = getExpDefRef(expDef)
    %%% This function will get the exp def reference file name for each
    %%% specific expDef.
    
    if contains(expDef,'imageWorld')
        expDefRef = 'imageWorld';
        
    elseif contains(expDef,'spontaneousActivity')
        expDefRef = 'spontaneous';
        
    elseif contains(expDef,'sparseNoise')
        expDefRef = 'sparseNoise';
        
    elseif contains(expDef,'multiSpaceWorld_checker_training') || ...
            contains(expDef, 'AVPassive_checkerboard_postactive')
        expDefRef = 'AVprotocol';
        
    else
        %%% TODO fill in that part with you own expDefs...
    end
