function expDefRef = getExpDefRef(expDef)
    %%% This function will get the exp def reference file name for each
    %%% specific expDef.
    %%% TODO fill in that part with you own expDefs...
            
    if contains(expDef,'imageWorld')
        expDefRef = 'imageWorld';
        
    elseif contains(expDef,'spontaneousActivity')
        expDefRef = 'spontaneous';
        
    elseif contains(expDef,'sparseNoise')
        expDefRef = 'sparseNoise';
        
    elseif contains(expDef,'multiSpaceWorld_checker_training') || ...
            contains(expDef, 'multiSpaceWorld_checker')
%         expDefRef = 'AVprotocol';
        expDefRef = 'AVprotocol2';
    else
        error('ExpDef reference does not exist for expDef %s',expDef)
    end
